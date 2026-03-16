import os
import pickle
import numpy as np
import lmdb
import multiprocessing
import logging
from PIL import Image
from tqdm import tqdm
from .mrc_preprocess import mrcs_resize, mrcs_to_int8, window_mask, raw_csdata_process_from_cryosparc_dir
from . import fft, mrc
import gc

logger = logging.getLogger(__name__)


def _open_lmdb_set(base_path, map_size, generate_processed_data, save_raw_data, generate_ft_data):
    """Build paths, create directories, and open LMDB environments for one dataset location."""
    lmdb_paths = {}
    if generate_processed_data:
        lmdb_paths['processed'] = os.path.join(base_path, 'lmdb_processed')
    if save_raw_data:
        lmdb_paths['raw'] = os.path.join(base_path, 'lmdb_raw')
    if generate_ft_data:
        lmdb_paths['FT'] = os.path.join(base_path, 'lmdb_FT')
    for path in lmdb_paths.values():
        os.makedirs(path, exist_ok=True)
    return {name: lmdb.open(path, map_size=map_size[name], readonly=False, create=True, max_readers=128)
            for name, path in lmdb_paths.items()}


def _write_to_lmdb(envs, processed_data_by_type, item_index, num_items):
    """Write a batch of processed items into their respective LMDB environments."""
    for data_type, data_list in processed_data_by_type.items():
        if data_type in envs:
            with envs[data_type].begin(write=True) as txn:
                for i in range(num_items):
                    key = f"{item_index + i}".encode()
                    txn.put(key, data_list[i])


def create_lmdb_dataset(image_path_list, save_data_path, map_size,
                        # MODIFIED: 新增开关
                        split_by_protein=True,
                        num_processes=None, chunksize=1, resize=224, raw_resize=None, is_to_int8=True,
                        window=True, window_r=0.85,
                        generate_processed_data=True, generate_ft_data=False, save_raw_data=False,

                        num_resample_mrcs_per_dataset=None):
    # 全局元数据变量 (所有模式下共用)
    path_id_data_list, protein_id_list, protein_id_dict, mean_std_states_sum = [], [], {}, {}
    if num_processes is None: num_processes = 16

    with multiprocessing.Pool(processes=num_processes) as pool:
        protein_id_counter = 0
        tasks = [(idx, data_path, resize, raw_resize, is_to_int8, window, window_r,
                  generate_processed_data, generate_ft_data, save_raw_data,
                  num_resample_mrcs_per_dataset[idx] if num_resample_mrcs_per_dataset else None)
                 for idx, data_path in enumerate(image_path_list)]
        print(f"Starting parallel processing of {len(tasks)} source files with {num_processes} workers...")

        results_buffer = {}
        next_to_process = 0

        # ==============================================================================
        # MODIFIED: 根据 split_by_protein 开关选择不同的写入逻辑
        # ==============================================================================
        if split_by_protein:
            print("INFO: LMDB datasets will be split by protein name.")
            # 按蛋白分割的模式下，动态管理环境和计数器
            protein_envs = {}
            protein_item_counters = {}
        else:
            print("INFO: Creating a single combined LMDB dataset.")
            global_envs = _open_lmdb_set(
                os.path.join(save_data_path, 'lmdb_data'), map_size,
                generate_processed_data, save_raw_data, generate_ft_data)
            global_item_index = 0

        with tqdm(total=len(tasks), desc="Processing source files") as pbar:
            results_iterator = pool.imap_unordered(lmdb_process_item, tasks, chunksize=chunksize)
            for result in results_iterator:
                original_idx = result[0]
                results_buffer[original_idx] = result

                while next_to_process in results_buffer:
                    _, path_id_data, processed_data_by_type, mean_std_stats = results_buffer.pop(next_to_process)
                    protein_name = tasks[next_to_process][1].split('/')[-3]

                    if path_id_data and mean_std_stats is not None:
                        num_items_in_batch = len(path_id_data)

                        # --- Write branch ---
                        if split_by_protein:
                            if protein_name not in protein_envs:
                                protein_base_path = os.path.join(save_data_path, 'lmdb_data', protein_name)
                                protein_envs[protein_name] = _open_lmdb_set(
                                    protein_base_path, map_size,
                                    generate_processed_data, save_raw_data, generate_ft_data)
                                protein_item_counters[protein_name] = 0
                            current_item_index = protein_item_counters[protein_name]
                            _write_to_lmdb(protein_envs[protein_name], processed_data_by_type,
                                           current_item_index, num_items_in_batch)
                            protein_item_counters[protein_name] += num_items_in_batch
                        else:
                            _write_to_lmdb(global_envs, processed_data_by_type,
                                           global_item_index, num_items_in_batch)
                            global_item_index += num_items_in_batch

                        # --- 全局元数据聚合 (所有模式下共用) ---
                        path_id_data_list.extend(path_id_data)
                        if protein_name not in protein_id_dict:
                            protein_id_dict[protein_name] = protein_id_counter
                            mean_std_states_sum[protein_name] = []
                            protein_id_counter += 1
                        current_protein_id = protein_id_dict[protein_name]
                        protein_id_list.extend([current_protein_id] * len(path_id_data))
                        mean_std_states_sum[protein_name].append(mean_std_stats)

                    pbar.update(1)
                    next_to_process += 1

    # --- 关闭所有LMDB环境 ---
    if split_by_protein:
        for env_dict in protein_envs.values():
            for env in env_dict.values():
                env.close()
    else:
        for env in global_envs.values():
            env.close()

    # --- 后续元数据处理逻辑 (不变) ---

    mean_std_id_dict = {}
    for protein_name, stats_list in mean_std_states_sum.items():
        mrcs_sum = {'FT': {'sum': 0.0, 'sq_sum': 0.0, 'count': 0}, 'processed': {'sum': 0.0, 'sq_sum': 0.0, 'count': 0},
                    'raw': {'sum': 0.0, 'sq_sum': 0.0, 'count': 0}}

        for mrcs_sates in stats_list:
            for key in mrcs_sum.keys():
                if key in mrcs_sates:
                    mrcs_sum[key]['sum'] += mrcs_sates[key]['sum']
                    mrcs_sum[key]['sq_sum'] += mrcs_sates[key]['sq_sum']
                    mrcs_sum[key]['count'] += mrcs_sates[key]['count']

        mean_std_results = {}
        for key in ['raw', 'processed', 'FT']:
            s = mrcs_sum[key]
            if s['count'] > 0:
                mean = s['sum'] / s['count']
                variance = (s['sq_sum'] / s['count']) - (mean ** 2)
                std = np.sqrt(max(0, variance))
                mean_std_results[key] = (mean, std)
            else:
                mean_std_results[key] = (0.0, 0.0)
        mean_std_id_dict[protein_id_dict[protein_name]] = mean_std_results

    with open(os.path.join(save_data_path, 'mean_std_id_dict.data'), 'wb') as f:
        pickle.dump(mean_std_id_dict, f)

    if protein_id_dict:
        with open(os.path.join(save_data_path, 'protein_id_dict.data'), 'wb') as f:
            pickle.dump(protein_id_dict, f)
    if protein_id_list:
        with open(os.path.join(save_data_path, 'protein_id_list.data'), 'wb') as f:
            pickle.dump(protein_id_list, f)
    print("\nLMDB dataset creation and metadata saving finished.")


def lmdb_process_item(args):
    idx, data_path, resize, raw_resize, is_to_int8, window, window_r, \
        generate_processed_data, generate_ft_data, save_raw_data, num_resample_mrcs = args
    try:
        # with mrcfile.open(data_path, permissive=True) as mrc:
        #     np_image_raw = mrc.data.astype(np.float32)
        np_image_raw, _ = mrc.parse_mrc(data_path)
        np_image_raw = np_image_raw.astype(np.float32)
        n_total = np_image_raw.shape[0]

        np_image_processed = None
        if generate_processed_data:
            processed_mrcs = np_image_raw
            if np_image_raw.shape[1] != resize:
                processed_mrcs = mrcs_resize(processed_mrcs, resize, resize)
            if is_to_int8:
                processed_mrcs = mrcs_to_int8(processed_mrcs)
            np_image_processed = processed_mrcs

        np_image_raw_processed = None
        if save_raw_data:
            np_image_raw_processed = np.copy(np_image_raw)  # 使用copy避免后续操作影响
            if window:
                win_mask = window_mask(np_image_raw_processed.shape[-1], window_r, .99)
                np_image_raw_processed *= win_mask
            # if raw_resize is not None and raw_resize != np_image_raw_processed.shape[1]:
            if raw_resize is not None and raw_resize < np_image_raw_processed.shape[1]:
                np_image_raw_processed = mrcs_resize(np_image_raw_processed, raw_resize, raw_resize)
            np_image_raw_processed = np_image_raw_processed.astype(np.float32)

        np_image_FT = None
        if generate_ft_data:
            if np_image_raw_processed is None:
                ft_input_stack = np.copy(np_image_raw)  # 从原始数据副本开始
                if window:
                    ft_input_stack *= window_mask(ft_input_stack.shape[-1], window_r, .99)
                if raw_resize is not None and raw_resize < ft_input_stack.shape[1]:
                    ft_input_stack = mrcs_resize(ft_input_stack, raw_resize, raw_resize)
            else:
                ft_input_stack = np.copy(np_image_raw_processed)

            particles = [fft.ht2_center(img) for img in ft_input_stack]
            np_image_FT = np.asarray(particles, dtype=np.float32)
            np_image_FT = fft.symmetrize_ht(np_image_FT)
            if np_image_FT.ndim == 2:
                np_image_FT = np.expand_dims(np_image_FT, axis=0)
            np_image_FT = np_image_FT.astype(np.float32)
            del ft_input_stack

        mean_std_stats = {
            'raw': {'sum': 0.0, 'sq_sum': 0.0, 'count': 0},
            'processed': {'sum': 0.0, 'sq_sum': 0.0, 'count': 0},
            'FT': {'sum': 0.0, 'sq_sum': 0.0, 'count': 0}
        }
        if num_resample_mrcs is not None and n_total > 0:
            sample_size = min(n_total, num_resample_mrcs)
            resample_id = np.random.choice(n_total, size=sample_size, replace=False)
            if generate_processed_data:
                resample_processed = np_image_processed[resample_id]
                mean_std_stats['processed']['sum'] = np.sum(resample_processed)
                mean_std_stats['processed']['sq_sum'] = np.sum(np.square(resample_processed))
                mean_std_stats['processed']['count'] = resample_processed.size
            if save_raw_data:
                resample_raw = np_image_raw_processed[resample_id]
                mean_std_stats['raw']['sum'] = np.sum(resample_raw)
                mean_std_stats['raw']['sq_sum'] = np.sum(np.square(resample_raw))
                mean_std_stats['raw']['count'] = resample_raw.size
            if generate_ft_data:
                resample_ft = np_image_FT[resample_id]
                mean_std_stats['FT']['sum'] = np.sum(resample_ft)
                mean_std_stats['FT']['sq_sum'] = np.sum(np.square(resample_ft))
                mean_std_stats['FT']['count'] = resample_ft.size

        processed_data_by_type = {}
        if generate_processed_data: processed_data_by_type['processed'] = []
        if save_raw_data: processed_data_by_type['raw'] = []
        if generate_ft_data: processed_data_by_type['FT'] = []
        path_id_my_data = []
        for i in range(n_total):
            if generate_processed_data:
                img_pil = Image.fromarray(np_image_processed[i]).convert('L')
                processed_data_by_type['processed'].append(pickle.dumps(img_pil, protocol=pickle.HIGHEST_PROTOCOL))
            if save_raw_data:
                processed_data_by_type['raw'].append(
                    pickle.dumps(np_image_raw_processed[i], protocol=pickle.HIGHEST_PROTOCOL))
            if generate_ft_data:
                processed_data_by_type['FT'].append(pickle.dumps(np_image_FT[i], protocol=pickle.HIGHEST_PROTOCOL))
            path_id_my_data.append(os.path.join(data_path.split('/')[-1], str(i + 1).zfill(6)))

        del np_image_raw, np_image_processed, np_image_raw_processed, np_image_FT
        gc.collect()
        return (idx, path_id_my_data, processed_data_by_type, mean_std_stats)
    except Exception as e:
        logger.error(f"Error processing {data_path}: {e}", exc_info=True)
        gc.collect()
        return (idx, [], {}, None)


def process_one_dataset_paths(dir_one_dataset, num_resample_per_dataset=40000):
    mrc_dir_list, mrcs_names_list_process, num_resample_mrcs_per_dataset = [], [], []
    if os.path.isdir(dir_one_dataset):
        try:
            mrc_dir, mrcs_names_list_temp = get_mrcs_names_list_cs(dir_one_dataset)
            mrc_dir_list.extend([mrc_dir] * len(mrcs_names_list_temp))
            mrcs_names_list_process.extend(mrcs_names_list_temp)
            if mrcs_names_list_temp:
                num_resample_mrcs_per_dataset.extend(
                    [int(num_resample_per_dataset / len(mrcs_names_list_temp))] * len(mrcs_names_list_temp))
        except Exception as e:
            logger.warning(f"Could not process directory {dir_one_dataset}: {e}")
    return mrc_dir_list, mrcs_names_list_process, num_resample_mrcs_per_dataset


def get_mrcs_names_list_cs(mrcfile_path):
    cs_data, mrc_dir = raw_csdata_process_from_cryosparc_dir(mrcfile_path)
    blob_path_list = cs_data['blob/path'].tolist()
    mrcs_names_list = [path.split('/')[-1] for path in blob_path_list]
    return mrc_dir, list(dict.fromkeys(mrcs_names_list))
