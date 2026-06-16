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
STACK_FILE_EXTENSIONS = ('.mrcs', '.mrc')
STACK_SUBDIR_CANDIDATES = ('mics', 'tiltseries')
LMDB_MIN_PARTICLE_CHUNK_SIZE = 32
LMDB_MAX_PARTICLE_CHUNK_SIZE = 512
LMDB_TARGET_CHUNK_BYTES = 256 * 1024 * 1024
MAX_LMDB_MAP_RESIZE_ATTEMPTS = 6


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
            _write_batch_with_map_resize(
                envs[data_type], item_index, data_list, num_items, data_type)


def _write_batch_with_map_resize(env, item_index, data_list, num_items, data_type,
                                 max_resize_attempts=MAX_LMDB_MAP_RESIZE_ATTEMPTS):
    if not data_list:
        return

    for attempt in range(max_resize_attempts + 1):
        try:
            with env.begin(write=True) as txn:
                for i in range(num_items):
                    key = f"{item_index + i}".encode()
                    txn.put(key, data_list[i])
            return
        except lmdb.MapFullError as exc:
            if attempt >= max_resize_attempts:
                raise RuntimeError(
                    f"LMDB map resize failed for {data_type} after "
                    f"{max_resize_attempts + 1} attempts."
                ) from exc
            old_map_size = env.info()['map_size']
            new_map_size = old_map_size * 2
            print(
                f"WARNING: LMDB map full for {data_type}. Growing map size from "
                f"{old_map_size / (1024 ** 3):.2f}GB to "
                f"{new_map_size / (1024 ** 3):.2f}GB and retrying."
            )
            env.set_mapsize(new_map_size)


def _zero_mean_std_stats():
    return {
        'raw': {'sum': 0.0, 'sq_sum': 0.0, 'count': 0},
        'processed': {'sum': 0.0, 'sq_sum': 0.0, 'count': 0},
        'FT': {'sum': 0.0, 'sq_sum': 0.0, 'count': 0}
    }


def _count_particles_in_stack(stack_path):
    header = mrc.parse_header(stack_path)
    return int(header.fields['nz'])


def _load_mrc_particle_slice(data_path, particle_start=0, particle_stop=None):
    header = mrc.parse_header(data_path)
    fields = header.fields
    nx = int(fields['nx'])
    ny = int(fields['ny'])
    nz = int(fields['nz'])
    extbytes = int(fields.get('next', 0) or 0)
    dtype = np.dtype(header.dtype)

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"Invalid MRC dimensions for {data_path}: {(nz, ny, nx)}")

    particle_start = max(0, int(particle_start))
    particle_stop = nz if particle_stop is None else min(int(particle_stop), nz)
    if particle_start >= particle_stop:
        return np.empty((0, ny, nx), dtype=dtype), header

    expected_bytes = int(1024 + extbytes + nx * ny * nz * dtype.itemsize)
    actual_bytes = int(os.path.getsize(data_path))
    if actual_bytes < expected_bytes:
        raise ValueError(
            f"Invalid MRC stack {data_path}: truncated_payload "
            f"(actual_bytes={actual_bytes}, expected_bytes={expected_bytes}, "
            f"shape={(nz, ny, nx)}, dtype={dtype})"
        )

    offset = 1024 + extbytes + particle_start * ny * nx * dtype.itemsize
    particle_count = particle_stop - particle_start
    np_image_raw = np.memmap(
        data_path,
        dtype=dtype,
        mode='r',
        offset=offset,
        shape=(particle_count, ny, nx),
    )
    return np_image_raw, header


def _estimate_lmdb_particle_working_set_bytes(source_side, resize, raw_resize,
                                              generate_processed_data,
                                              generate_ft_data, save_raw_data):
    source_side = max(1, int(source_side))
    resize = max(1, int(resize or source_side))
    raw_side = max(1, int(raw_resize or source_side))

    estimated_bytes = source_side * source_side * 4
    if generate_processed_data:
        estimated_bytes += resize * resize * 5
    if save_raw_data:
        estimated_bytes += raw_side * raw_side * 4
    if generate_ft_data:
        estimated_bytes += raw_side * raw_side * 12
    return max(1, int(estimated_bytes))


def _compute_adaptive_lmdb_chunk_size(total_particles, source_side, resize, raw_resize,
                                      generate_processed_data, generate_ft_data,
                                      save_raw_data):
    total_particles = max(1, int(total_particles))
    estimated_bytes_per_particle = _estimate_lmdb_particle_working_set_bytes(
        source_side, resize, raw_resize, generate_processed_data,
        generate_ft_data, save_raw_data)
    adaptive_chunk_size = LMDB_TARGET_CHUNK_BYTES // estimated_bytes_per_particle
    adaptive_chunk_size = max(LMDB_MIN_PARTICLE_CHUNK_SIZE, int(adaptive_chunk_size))
    adaptive_chunk_size = min(LMDB_MAX_PARTICLE_CHUNK_SIZE, adaptive_chunk_size)
    adaptive_chunk_size = min(total_particles, adaptive_chunk_size)
    return max(1, adaptive_chunk_size)


def _normalize_lmdb_task(args):
    if isinstance(args, dict):
        task = dict(args)
    else:
        idx, data_path, resize, raw_resize, is_to_int8, window, window_r, \
            generate_processed_data, generate_ft_data, save_raw_data, num_resample_mrcs = args
        total_particles = _count_particles_in_stack(data_path)
        task = {
            'task_index': idx,
            'original_idx': idx,
            'stack_id': idx,
            'data_path': data_path,
            'stack_name': os.path.basename(data_path),
            'protein_name': os.path.normpath(data_path).split(os.sep)[-3],
            'resize': resize,
            'raw_resize': raw_resize,
            'is_to_int8': is_to_int8,
            'window': window,
            'window_r': window_r,
            'generate_processed_data': generate_processed_data,
            'generate_ft_data': generate_ft_data,
            'save_raw_data': save_raw_data,
            'num_resample_mrcs': num_resample_mrcs,
            'particle_start': 0,
            'particle_stop': total_particles,
            'total_particles': total_particles,
        }

    total_particles = int(task.get('total_particles') or _count_particles_in_stack(task['data_path']))
    task.setdefault('task_index', int(task.get('original_idx', 0)))
    task.setdefault('original_idx', int(task['task_index']))
    task.setdefault('stack_id', int(task['original_idx']))
    task.setdefault('stack_name', os.path.basename(task['data_path']))
    task.setdefault('protein_name', os.path.normpath(task['data_path']).split(os.sep)[-3])
    task.setdefault('particle_start', 0)
    task.setdefault('particle_stop', total_particles)
    task.setdefault('num_resample_mrcs', None)
    task['particle_start'] = int(task['particle_start'])
    task['particle_stop'] = int(task['particle_stop'])
    task['total_particles'] = total_particles
    return task


def _build_lmdb_tasks(image_path_list, resize, raw_resize, is_to_int8, window, window_r,
                      generate_processed_data, generate_ft_data, save_raw_data,
                      num_resample_mrcs_per_dataset=None, particle_chunk_size=None):
    if particle_chunk_size is not None:
        particle_chunk_size = int(particle_chunk_size)
        if particle_chunk_size <= 0:
            raise ValueError('particle_chunk_size must be a positive integer or None.')

    tasks = []
    for original_idx, data_path in enumerate(image_path_list):
        header = mrc.parse_header(data_path)
        total_particles = int(header.fields['nz'])
        ny = int(header.fields['ny'])
        nx = int(header.fields['nx'])
        source_side = min(ny, nx)
        if total_particles <= 0:
            continue

        if particle_chunk_size is None:
            chunk_particle_count = _compute_adaptive_lmdb_chunk_size(
                total_particles, source_side, resize, raw_resize,
                generate_processed_data, generate_ft_data, save_raw_data)
        else:
            chunk_particle_count = min(total_particles, particle_chunk_size)

        sample_target = None
        if num_resample_mrcs_per_dataset:
            sample_target = min(total_particles, int(num_resample_mrcs_per_dataset[original_idx]))

        for chunk_index, particle_start in enumerate(range(0, total_particles, chunk_particle_count)):
            particle_stop = min(particle_start + chunk_particle_count, total_particles)
            chunk_sample_target = None
            if sample_target is not None:
                chunk_sample_target = (
                    (particle_stop * sample_target) // total_particles
                    - (particle_start * sample_target) // total_particles
                )
            tasks.append({
                'task_index': len(tasks),
                'original_idx': original_idx,
                'stack_id': original_idx,
                'chunk_index': chunk_index,
                'data_path': data_path,
                'stack_name': os.path.basename(data_path),
                'protein_name': os.path.normpath(data_path).split(os.sep)[-3],
                'resize': resize,
                'raw_resize': raw_resize,
                'is_to_int8': is_to_int8,
                'window': window,
                'window_r': window_r,
                'generate_processed_data': generate_processed_data,
                'generate_ft_data': generate_ft_data,
                'save_raw_data': save_raw_data,
                'num_resample_mrcs': chunk_sample_target,
                'particle_start': particle_start,
                'particle_stop': particle_stop,
                'total_particles': total_particles,
            })
    return tasks


def create_lmdb_dataset(image_path_list, save_data_path, map_size,
                        # MODIFIED: 新增开关
                        split_by_protein=True,
                        num_processes=None, chunksize=1, resize=224, raw_resize=None, is_to_int8=True,
                        window=True, window_r=0.85,
                        generate_processed_data=True, generate_ft_data=False, save_raw_data=False,

                        num_resample_mrcs_per_dataset=None, particle_chunk_size=None):
    # 全局元数据变量 (所有模式下共用)
    path_id_data_list, protein_id_list, protein_id_dict, mean_std_states_sum = [], [], {}, {}
    if num_processes is None: num_processes = 16

    protein_id_counter = 0
    tasks = _build_lmdb_tasks(
        image_path_list, resize, raw_resize, is_to_int8, window, window_r,
        generate_processed_data, generate_ft_data, save_raw_data,
        num_resample_mrcs_per_dataset=num_resample_mrcs_per_dataset,
        particle_chunk_size=particle_chunk_size)
    print(
        f"Starting parallel processing of {len(image_path_list)} source files "
        f"as {len(tasks)} chunk tasks with {num_processes} workers...")

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

    def _process_result(result, task):
        nonlocal protein_id_counter
        nonlocal global_item_index
        _, path_id_data, processed_data_by_type, mean_std_stats = result
        protein_name = task['protein_name']

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

    if num_processes and num_processes > 1:
        with multiprocessing.Pool(processes=num_processes) as pool:
            with tqdm(total=len(tasks), desc="Processing source chunks") as pbar:
                results_iterator = pool.imap(lmdb_process_item, tasks, chunksize=chunksize)
                for task, result in zip(tasks, results_iterator):
                    _process_result(result, task)
                    pbar.update(1)
    else:
        with tqdm(total=len(tasks), desc="Processing source chunks") as pbar:
            for task in tasks:
                result = lmdb_process_item(task)
                _process_result(result, task)
                pbar.update(1)

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
    task = _normalize_lmdb_task(args)
    idx = task['task_index']
    data_path = task['data_path']
    resize = task['resize']
    raw_resize = task['raw_resize']
    is_to_int8 = task['is_to_int8']
    window = task['window']
    window_r = task['window_r']
    generate_processed_data = task['generate_processed_data']
    generate_ft_data = task['generate_ft_data']
    save_raw_data = task['save_raw_data']
    num_resample_mrcs = task['num_resample_mrcs']
    particle_start = task['particle_start']
    particle_stop = task['particle_stop']
    try:
        # with mrcfile.open(data_path, permissive=True) as mrc:
        #     np_image_raw = mrc.data.astype(np.float32)
        np_image_raw, _ = _load_mrc_particle_slice(data_path, particle_start, particle_stop)
        np_image_raw = np.asarray(np_image_raw, dtype=np.float32)
        n_total = np_image_raw.shape[0]

        np_image_processed = None
        if generate_processed_data:
            processed_mrcs = np_image_raw
            if resize is not None and np_image_raw.shape[1] != resize:
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

        mean_std_stats = _zero_mean_std_stats()
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
                if is_to_int8:
                    payload = Image.fromarray(np_image_processed[i]).convert('L')
                else:
                    payload = np.asarray(np_image_processed[i], dtype=np.float32)
                processed_data_by_type['processed'].append(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
            if save_raw_data:
                processed_data_by_type['raw'].append(
                    pickle.dumps(np_image_raw_processed[i], protocol=pickle.HIGHEST_PROTOCOL))
            if generate_ft_data:
                processed_data_by_type['FT'].append(pickle.dumps(np_image_FT[i], protocol=pickle.HIGHEST_PROTOCOL))
            path_id_my_data.append(os.path.join(task['stack_name'], str(particle_start + i + 1).zfill(6)))

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
    mrcfile_path = os.fspath(mrcfile_path)
    try:
        cs_data, mrc_dir = raw_csdata_process_from_cryosparc_dir(mrcfile_path)
    except Exception:
        cs_data = None
        mrc_dir = os.path.dirname(mrcfile_path) if mrcfile_path.endswith(STACK_FILE_EXTENSIONS) else mrcfile_path

    if cs_data is not None:
        blob_path_list = cs_data['blob/path'].tolist()
        mrcs_names_list = [path.split('/')[-1] for path in blob_path_list]
    elif mrcfile_path.endswith(STACK_FILE_EXTENSIONS):
        mrcs_names_list = [os.path.basename(mrcfile_path)]
    else:
        for subdir_name in STACK_SUBDIR_CANDIDATES:
            full_path = os.path.join(mrc_dir, subdir_name)
            if os.path.isdir(full_path):
                mrc_dir = full_path
                break
        mrcs_names_list = sorted(os.listdir(mrc_dir))
        mrcs_names_list = [name for name in mrcs_names_list if name.endswith(STACK_FILE_EXTENSIONS)]
    return mrc_dir, list(dict.fromkeys(mrcs_names_list))
