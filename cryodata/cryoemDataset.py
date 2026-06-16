# from models.get_transformers import to_int8
from cryodata.data_preprocess.mrc_preprocess import to_int8
from bisect import bisect_left
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os
import pickle
import random
import torch
import json
import lmdb
from annoy import AnnoyIndex
import logging

# Add a module-level logger
logger = logging.getLogger(__name__)

# Pixels threshold above which an image is classified as a micrograph rather than a particle
MICROGRAPH_SIZE_THRESHOLD = 384
LMDB_REFERENCE_MANIFEST_FILENAME = 'lmdb_reference_manifest.json'
CALCULATED_SCORE_SOURCES = frozenset((0, 3))
DATA_SOURCE_LABEL_FILENAME = 'labels_data_source.data'
SOURCE_MRCS_GROUP_ID_FILENAME = 'source_mrcs_group_id.data'
SOURCE_MRCS_GROUP_KEY_FILENAME = 'source_mrcs_group_key.data'
DATA_SOURCE_PTCLS = 'ptcls'
DATA_SOURCE_MICS = 'mics'
DATA_SOURCE_ET_TILTS = 'et_tilts'
DATA_SOURCE_ET_PTCLS = 'et_ptcls'
SUPPORTED_DATA_SOURCE_TYPES = (
    DATA_SOURCE_PTCLS,
    DATA_SOURCE_MICS,
    DATA_SOURCE_ET_TILTS,
    DATA_SOURCE_ET_PTCLS,
)
MICROGRAPH_LIKE_DATA_SOURCES = frozenset((DATA_SOURCE_MICS, DATA_SOURCE_ET_TILTS))


def _derive_used_default_score_from_source(score_source_values):
    return [0 if int(value) in CALCULATED_SCORE_SOURCES else 1 for value in score_source_values]


def _normalize_score_source_labels(score_source_values, legacy_default_values, expected_length):
    if score_source_values is not None:
        score_source_list = list(score_source_values)
        if len(score_source_list) == 0:
            return [0] * expected_length
        if len(score_source_list) != expected_length:
            logger.warning(
                f"labels_score_source.data has {len(score_source_list)} entries; expected {expected_length}. Falling back to zeros."
            )
            return [0] * expected_length
        return [int(value) for value in score_source_list]

    if legacy_default_values is not None:
        legacy_default_list = list(legacy_default_values)
        if len(legacy_default_list) == 0:
            return [0] * expected_length
        if len(legacy_default_list) != expected_length:
            logger.warning(
                f"labels_used_default_score.data has {len(legacy_default_list)} entries; expected {expected_length}. Falling back to zeros."
            )
            return [0] * expected_length
        return [0 if int(flag) == 0 else 1 for flag in legacy_default_list]

    return [0] * expected_length


def _normalize_data_source_labels(data_source_values, expected_length, data_path):
    """Load per-particle source labels, defaulting legacy datasets to cryo-EM particles."""
    if data_source_values is None:
        return [DATA_SOURCE_PTCLS] * expected_length

    labels = [str(value) for value in data_source_values]
    if len(labels) != expected_length:
        raise ValueError(
            f"{DATA_SOURCE_LABEL_FILENAME} length mismatch: expected {expected_length}, "
            f"found {len(labels)} in {data_path}."
        )
    invalid = sorted({label for label in labels if label not in SUPPORTED_DATA_SOURCE_TYPES})
    if invalid:
        raise ValueError(
            f"{DATA_SOURCE_LABEL_FILENAME} contains invalid data source labels: {invalid}. "
            f"Supported values: {list(SUPPORTED_DATA_SOURCE_TYPES)}."
        )
    return labels


def _normalize_source_mrcs_group_ids(group_id_values, expected_length, data_path):
    if group_id_values is None:
        return None

    group_ids = list(group_id_values)
    if len(group_ids) != expected_length:
        raise ValueError(
            f"{SOURCE_MRCS_GROUP_ID_FILENAME} length mismatch: expected {expected_length}, "
            f"found {len(group_ids)} in {data_path}."
        )

    normalized_group_ids = []
    for index, group_id in enumerate(group_ids):
        try:
            normalized_group_ids.append(int(group_id))
        except (TypeError, ValueError):
            raise ValueError(
                f"{SOURCE_MRCS_GROUP_ID_FILENAME} contains non-integer group id "
                f"at index {index}: {group_id!r}."
            )
    return normalized_group_ids


class MyEmFile(object):
    def __init__(self, emfile_path=None, selected_emfile_path=None, filetype='star'):
        self.filetype = filetype
        if emfile_path:
            if emfile_path.endswith(".star"):
                self.particles_file_content, self.particles_star_title, self.particles_id = self.read_star(emfile_path)
                if selected_emfile_path is not None and selected_emfile_path.endswith(".star"):
                    self.selected_particles_file_content, self.selected_particles_star_file_title, self.selected_particles_id = self.read_star(
                        selected_emfile_path)
                    self.unselected_particles_file_content, self.unselected_particles_id = self.divide_selected_unselected_particles_star(
                        self.particles_file_content, self.particles_id, self.selected_particles_id)
                else:
                    self.selected_particles_id = None

            if emfile_path.endswith(".cs"):
                self.filetype = 'cs'
                self.particles_file_content, self.particles_id = self.read_cs(emfile_path)
                if selected_emfile_path is not None and selected_emfile_path.endswith(".cs"):
                    self.selected_particles_csfile_content, self.selected_particles_id = self.read_cs(
                        selected_emfile_path)
                    self.unselected_particles_csfile_content, self.unselected_particles_id = self.divide_selected_unselected_particles_cs(
                        self.particles_file_content, self.particles_id, self.selected_particles_id)
                    # pass
                else:
                    self.selected_particles_id = None
        else:
            self.particles_id = None
            self.filetype = None
            self.selected_particles_id = None
        # pass

    def read_star(self, star_path):
        with open(star_path, "r") as starfile:
            star_data = starfile.readlines()
        for index, x in enumerate(star_data):
            if x == 'data_particles\n':
                for index2, x2 in enumerate(star_data[index:]):

                    splited_x = x2.split()
                    next_splited_x = star_data[index + index2 + 1].split()
                    if splited_x:
                        item_num = splited_x[-1].replace("#", "")
                        if item_num.isdigit():
                            if int(item_num) == len(next_splited_x) and int(item_num) != len(splited_x):
                                start_site = index + index2 + 1
                                break
        content = star_data[start_site:]
        title = star_data[:start_site]
        img_ids = self.get_star_image_id(content)
        return content, title, img_ids

    def read_cs(self, cs_path):
        cs_data = Dataset.load(cs_path)
        img_ids = cs_data['uid'].tolist()
        # mm=cs_data['blob/path'].tolist()
        # dd=cs_data['blob/idx'].tolist()
        return cs_data, img_ids

    def get_star_image_id(self, star_content):
        image_id = []
        for x in star_content:
            if len(x.strip()) > 0:
                image_id.append(x.strip().split()[5])
        return image_id

    def divide_selected_unselected_particles_star(self, particles_star_content, particles_id, selected_particles_id):
        unselected_particles_star_content = [particles_star_content[index] for index, x in enumerate(particles_id) if
                                             len(x) > 0 and x not in selected_particles_id]
        unselected_particles_id = self.get_star_image_id(unselected_particles_star_content)
        return unselected_particles_star_content, unselected_particles_id

    def divide_selected_unselected_particles_cs(self, particles_cs_content, particles_id, selected_particles_id):
        unselected_list = []
        unselected_particles_id = []
        for i, id in enumerate(particles_id):
            if id not in selected_particles_id:
                unselected_list.append(i)
                unselected_particles_id.append(id)
        unselected_particles_cs_content = particles_cs_content.take(unselected_list)
        return unselected_particles_cs_content, unselected_particles_id


class CryoMetaData(MyEmFile):
    def __init__(self, cfg=None, mrc_path=None, emfile_path=None, processed_data_path=None, selected_emfile_path=None,
                 tmp_data_save_path=None,
                 is_extra_valset=False, accelerator=None):
        super(CryoMetaData, self).__init__(emfile_path, selected_emfile_path)

        self.processed_data_path = processed_data_path
        self.id_index_dict = None
        self.id_score_source_dict = None
        self.id_used_default_score_dict = None
        self.id_data_source_dict = None

        assert processed_data_path is not None, "processed_data_path must be provided"
        self.load_preprocessed_data_path(data_path=processed_data_path)

    def _safe_load_json(self, path):
        try:
            with open(path, 'r') as fh:
                return json.load(fh)
        except Exception as e:
            logger.warning(f"Failed to load JSON file {path}: {e}")
            return None

    def close(self):
        """Close any open LMDB environments held by this metadata object."""
        # If lmdb_env or cached envs are present on the object, try to close them
        try:
            if hasattr(self, 'lmdb_env') and self.lmdb_env is not None:
                try:
                    self.lmdb_env.close()
                except Exception:
                    pass
                self.lmdb_env = None
        except Exception:
            pass

    def __del__(self):
        # Ensure resources are closed on garbage collection
        try:
            self.close()
        except Exception:
            pass

    def load_path(self):
        mrcs_path_list = []
        listdir(self.mrc_path, mrcs_path_list)
        self.mrcs_path_list = mrcs_path_list
        # if self.selected_particles_id is not None:
        # pass

    def load_preprocessed_data_path(self, data_path):
        def _validate_loaded_label_length(name, values, allow_empty_default=False):
            if values is None:
                return
            values_len = len(values)
            if allow_empty_default and values_len == 0:
                return
            if values_len != self.length:
                raise ValueError(
                    f"{name} length mismatch: expected {self.length}, found {values_len} in {data_path}."
                )

        self.lmdb_reference_manifest = None
        manifest_path = os.path.join(data_path, LMDB_REFERENCE_MANIFEST_FILENAME)
        if os.path.exists(manifest_path):
            self.lmdb_reference_manifest = self._safe_load_json(manifest_path)
            if self.lmdb_reference_manifest is None:
                raise ValueError(f"Failed to load {LMDB_REFERENCE_MANIFEST_FILENAME} from {data_path}.")

        if self.lmdb_reference_manifest is None and not os.path.exists(data_path + '/lmdb_data'):
            raise ValueError(
                f"{data_path} does not contain LMDB data."
            )

        with open(data_path + '/protein_id_list.data', 'rb') as filehandle:
            protein_id_list = pickle.load(filehandle)
        self.length = len(protein_id_list)
        if os.path.exists(data_path + '/lmdb_data'):
            self.lmdb_path = data_path + '/lmdb_data/'
        else:
            self.lmdb_path = None

        if os.path.exists(data_path + '/labels_for_clustering.data'):
            with open(data_path + '/labels_for_clustering.data', 'rb') as filehandle:
                self.labels_for_clustering = pickle.load(filehandle)
            _validate_loaded_label_length(
                'labels_for_clustering.data',
                self.labels_for_clustering,
                allow_empty_default=True,
            )
        else:
            self.labels_for_clustering = None

        if os.path.exists(data_path + '/labels_classification.data'):
            with open(data_path + '/labels_classification.data', 'rb') as filehandle:
                self.labels_classification = pickle.load(filehandle)
            if len(self.labels_classification) == 0:
                self.labels_classification = [1] * self.length
            else:
                _validate_loaded_label_length('labels_classification.data', self.labels_classification)
        else:
            self.labels_classification = [1] * self.length

        labels_score_source_raw = None
        if os.path.exists(data_path + '/labels_score_source.data'):
            with open(data_path + '/labels_score_source.data', 'rb') as filehandle:
                labels_score_source_raw = pickle.load(filehandle)

        legacy_default_score_labels = None
        if labels_score_source_raw is None and os.path.exists(data_path + '/labels_used_default_score.data'):
            with open(data_path + '/labels_used_default_score.data', 'rb') as filehandle:
                legacy_default_score_labels = pickle.load(filehandle)

        self.labels_score_source = _normalize_score_source_labels(
            labels_score_source_raw,
            legacy_default_score_labels,
            self.length,
        )
        self.labels_used_default_score = _derive_used_default_score_from_source(self.labels_score_source)
        labels_data_source_raw = None
        if os.path.exists(os.path.join(data_path, DATA_SOURCE_LABEL_FILENAME)):
            with open(os.path.join(data_path, DATA_SOURCE_LABEL_FILENAME), 'rb') as filehandle:
                labels_data_source_raw = pickle.load(filehandle)
        self.labels_data_source = _normalize_data_source_labels(
            labels_data_source_raw,
            self.length,
            data_path,
        )
        source_mrcs_group_id_raw = None
        if os.path.exists(os.path.join(data_path, SOURCE_MRCS_GROUP_ID_FILENAME)):
            with open(os.path.join(data_path, SOURCE_MRCS_GROUP_ID_FILENAME), 'rb') as filehandle:
                source_mrcs_group_id_raw = pickle.load(filehandle)
        self.source_mrcs_group_id = _normalize_source_mrcs_group_ids(
            source_mrcs_group_id_raw,
            self.length,
            data_path,
        )

        self.source_mrcs_group_key = None
        if os.path.exists(os.path.join(data_path, SOURCE_MRCS_GROUP_KEY_FILENAME)):
            with open(os.path.join(data_path, SOURCE_MRCS_GROUP_KEY_FILENAME), 'rb') as filehandle:
                self.source_mrcs_group_key = pickle.load(filehandle)

        # with open(path_out + 'output_tif_select_label.data', 'rb') as filehandle:
        #     self.tifs_selection_label = pickle.load(filehandle)

        if os.path.exists(data_path + '/means_stds.data'):
            with open(data_path + '/means_stds.data',
                      'rb') as filehandle:
                self.means_stds = pickle.load(filehandle)

        if os.path.exists(data_path + '/protein_id_list.data'):
            with open(data_path + '/protein_id_list.data',
                      'rb') as filehandle:
                self.protein_id_list = pickle.load(filehandle)
        else:
            self.protein_id_list = None

        if os.path.exists(data_path + '/protein_id_dict.data'):
            with open(data_path + '/protein_id_dict.data',
                      'rb') as filehandle:
                self.protein_id_dict = pickle.load(filehandle)
        else:
            self.protein_id_dict = None

        if os.path.exists(data_path + '/pretrain_data.json'):
            self.dataset_map = json.load(open(data_path + '/pretrain_data.json', 'r'))
        elif os.path.exists(data_path + '/finetune_data.json'):
            self.dataset_map = json.load(open(data_path + '/finetune_data.json', 'r'))
        else:
            self.dataset_map = None

        # if os.path.exists(data_path + '/mean_error_dict.json'):
        #     mean_error_dict = json.load(open(data_path + '/mean_error_dict.json', 'r'))
        #     mean_error_dis_dict = get_mean_error_distribution(mean_error_dict)
        #     self.mean_error_dis_dict = {self.protein_id_dict[key]: value for key, value in mean_error_dis_dict.items()}
        #     # self.mean_error_dis_dict = {'good':{self.protein_id_dict[key]: value for key, value in mean_error_dis_dict['good'].items()},
        #     #                             'bad':{self.protein_id_dict[key]: value for key, value in mean_error_dis_dict['bad'].items()}}
        # else:
        #     self.mean_error_dis_dict = None

        if os.path.exists(data_path + '/data_error_dict.json'):
            data_error_dict = json.load(open(data_path + '/data_error_dict.json', 'r'))
            self.data_error_dis_dict = {self.protein_id_dict[key]: np.array(value) for key, value in
                                        data_error_dict.items()}

        self.pose_id_map = None
        if os.path.exists(data_path + '/pose_id_map.data'):
            # self.pose_id_map = json.load(open(data_path + '/pose_id_map.data', 'r'))
            with open(data_path + '/pose_id_map.data',
                      'rb') as filehandle:
                pose_id_map = pickle.load(filehandle)
            if len(pose_id_map) > 0:
                self.pose_id_map = pose_id_map

        self.pose_id_map2 = None
        if os.path.exists(data_path + '/pose_id_map2.data'):
            # self.pose_id_map = json.load(open(data_path + '/pose_id_map.data', 'r'))
            with open(data_path + '/pose_id_map2.data',
                      'rb') as filehandle:
                pose_id_map2 = pickle.load(filehandle)
            if len(pose_id_map2) > 0:
                self.pose_id_map2 = pose_id_map2



    def preprocess_trainset_valset_index_finetune(self,
                                                  # valset_name=[],
                                                  # dataset_except_names=[],
                                                  ratio_balance_train=[0.35, 0.3, 0.35],
                                                  max_number_per_sample=None,
                                                  # is_valset=False,
                                                  is_balance=True,
                                                  middle_range_balance_train=[0.5, 0.85],
                                                  # data_error_dis_dict=None
                                                  ):
        id_index_dict_pos = {}
        id_index_dict_neg = {}
        id_index_dict_mid = {}

        # if data_error_dis_dict is not None:
        #     data_error_dis_dict_pos = {}
        #     data_error_dis_dict_neg = {}
        #     data_error_dis_dict_mid = {}

        protein_id_list_np = np.array(self.protein_id_list)
        labels_classification_np = np.array(self.labels_classification)
        labels_score_source_np = np.array(self.labels_score_source)
        labels_used_default_score_np = np.array(self.labels_used_default_score)
        labels_data_source_np = np.array(self.labels_data_source)
        for name, id in self.protein_id_dict.items():
            item_pos = {}
            item_neg = {}
            item_mid = {}
            # if name not in dataset_except_names:
            protein_index = np.where(protein_id_list_np == id)[0]
            if name.lower().endswith('good'):
                # if name in valset_name and is_valset:
                #     id_index_dict_pos[id] = np.where(protein_id_list_np == id)[0].tolist()
                # elif name not in valset_name and not is_valset:
                #     id_index_dict_pos[id] = np.where(protein_id_list_np == id)[0].tolist()
                #     if data_error_dis_dict is not None:
                #         data_error_dis_dict_pos[id] = data_error_dis_dict[id] / np.sum(data_error_dis_dict[id])
                item_pos['id'] = protein_index.tolist()
                item_pos['score'] = [1.0] * len(item_pos['id'])
                item_pos['score_source'] = labels_score_source_np[protein_index].tolist()
                item_pos['used_default_score'] = labels_used_default_score_np[protein_index].tolist()
                item_pos['data_source'] = labels_data_source_np[protein_index].tolist()
            elif name.lower().endswith('bad'):
                # if name in valset_name and is_valset:
                #     id_index_dict_neg[id] = np.where(protein_id_list_np == id)[0].tolist()
                # elif name not in valset_name and not is_valset:
                #     id_index_dict_neg[id] = np.where(protein_id_list_np == id)[0].tolist()
                #     if data_error_dis_dict is not None:
                #         data_error_dis_dict_neg[id] = data_error_dis_dict[id] / np.sum(data_error_dis_dict[id])
                item_neg['id'] = protein_index.tolist()
                item_neg['score'] = [0.0] * len(item_neg['id'])
                item_neg['score_source'] = labels_score_source_np[protein_index].tolist()
                item_neg['used_default_score'] = labels_used_default_score_np[protein_index].tolist()
                item_neg['data_source'] = labels_data_source_np[protein_index].tolist()
            else:
                pos_index = protein_index[labels_classification_np[protein_index] >= middle_range_balance_train[1]]
                # Discard data with a score less than 0
                neg_index = protein_index[(labels_classification_np[protein_index] < middle_range_balance_train[0]) & (
                            labels_classification_np[protein_index] >= 0)]
                # if data_error_dis_dict is not None:
                #     pos_dis = data_error_dis_dict[id][
                #         labels_classification_np[protein_index] >= middle_range_balance_train[1]]
                #     neg_dis = data_error_dis_dict[id][
                #         labels_classification_np[protein_index] < middle_range_balance_train[0]]
                if middle_range_balance_train[0] != middle_range_balance_train[1]:
                    mid_index = protein_index[
                        (labels_classification_np[protein_index] >= middle_range_balance_train[0]) & (
                                labels_classification_np[protein_index] < middle_range_balance_train[1])]
                    if len(mid_index) > 0:
                        # id_index_dict_mid[id] = mid_index.tolist()
                        item_mid['id'] = mid_index.tolist()
                        item_mid['score'] = labels_classification_np[mid_index].tolist()
                        item_mid['score_source'] = labels_score_source_np[mid_index].tolist()
                        item_mid['used_default_score'] = labels_used_default_score_np[mid_index].tolist()
                        item_mid['data_source'] = labels_data_source_np[mid_index].tolist()
                        # if data_error_dis_dict is not None:
                        #     mid_dis = data_error_dis_dict[id][
                        #         (labels_classification_np[protein_index] >= middle_range_balance_train[0]) & (
                        #                 labels_classification_np[protein_index] < middle_range_balance_train[1])]
                        #     data_error_dis_dict_mid[id] = mid_dis / np.sum(mid_dis)
                    # id_index_dict_mid[id] = mid_index.tolist()
                # if name in valset_name and is_valset:
                #     id_index_dict_pos[id] = pos_index.tolist()
                #     id_index_dict_neg[id] = neg_index.tolist()
                #
                # elif name not in valset_name and not is_valset:
                #     if len(pos_index) > 0:
                #         id_index_dict_pos[id] = pos_index.tolist()
                #         if data_error_dis_dict is not None:
                #             data_error_dis_dict_pos[id] = pos_dis / np.sum(pos_dis)
                #     if len(neg_index) > 0:
                #         id_index_dict_neg[id] = neg_index.tolist()
                #         if data_error_dis_dict is not None:
                #             data_error_dis_dict_neg[id] = neg_dis / np.sum(neg_dis)
                if len(pos_index) > 0:
                    item_pos['id'] = pos_index.tolist()
                    item_pos['score'] = labels_classification_np[pos_index].tolist()
                    item_pos['score_source'] = labels_score_source_np[pos_index].tolist()
                    item_pos['used_default_score'] = labels_used_default_score_np[pos_index].tolist()
                    item_pos['data_source'] = labels_data_source_np[pos_index].tolist()
                if len(neg_index) > 0:
                    item_neg['id'] = neg_index.tolist()
                    item_neg['score'] = labels_classification_np[neg_index].tolist()
                    item_neg['score_source'] = labels_score_source_np[neg_index].tolist()
                    item_neg['used_default_score'] = labels_used_default_score_np[neg_index].tolist()
                    item_neg['data_source'] = labels_data_source_np[neg_index].tolist()
            if len(item_pos) > 0:
                id_index_dict_pos[id] = item_pos
            if len(item_neg) > 0:
                id_index_dict_neg[id] = item_neg
            if len(item_mid) > 0:
                id_index_dict_mid[id] = item_mid
        if is_balance:
            resample_num_p = int(max_number_per_sample * ratio_balance_train[2])
            resample_num_n = int(max_number_per_sample * ratio_balance_train[0])
            resample_num_m = int(max_number_per_sample * ratio_balance_train[1]) if len(id_index_dict_mid) > 0 else 0
            # if len(id_index_dict_mid)==0:
            #     resample_num_p = int(max_number_per_sample * 4 * ratio_balance_train[1] * len(id_index_dict_neg) / (
            #             len(id_index_dict_pos) + len(id_index_dict_neg)))
            #     resample_num_n = int(max_number_per_sample * 2 - resample_num_p)
            #     resample_num_m=0
            # else:
            #     # resample_num_p=int(6 * ratio_balance_train[1] * max_number_per_sample * (len(id_index_dict_neg) * len(id_index_dict_mid)) / (len(id_index_dict_pos) * len(id_index_dict_neg) + len(id_index_dict_neg) * len(id_index_dict_mid) + len(id_index_dict_pos) * len(id_index_dict_mid)))
            #     # resample_num_n=int(3*(1-positive_ratio)*max_number_per_sample*(len(id_index_dict_pos)*len(id_index_dict_mid))/(len(id_index_dict_pos)*len(id_index_dict_neg)+len(id_index_dict_neg)*len(id_index_dict_mid)+len(id_index_dict_pos)*len(id_index_dict_mid)))
            #     # resample_num_m=2*max_number_per_sample-resample_num_p-resample_num_n
            #     # resample_num_p = int(2 * positive_ratio * max_number_per_sample)
            #     # resample_num_n = int( max_number_per_sample-resample_num_p/2)
            #     # resample_num_m = 2*max_number_per_sample-resample_num_p-resample_num_n
            #     # ratio_multi=[1,1,1]
            #     resample_num_p = int(max_number_per_sample* ratio_balance_train[2])
            #     resample_num_n = int(max_number_per_sample* ratio_balance_train[0])
            #     resample_num_m = int(max_number_per_sample* ratio_balance_train[1])

        else:
            resample_num_p = max_number_per_sample
            resample_num_n = max_number_per_sample
            resample_num_m = max_number_per_sample
        # data_error_dis_dict_all = {'good': data_error_dis_dict_pos, 'bad': data_error_dis_dict_neg,
        #                            'mid': data_error_dis_dict_mid} if data_error_dis_dict is not None else {
        #     'good': None, 'bad': None, 'mid': None}
        return id_index_dict_pos, id_index_dict_neg, id_index_dict_mid, (
            resample_num_p, resample_num_n, resample_num_m)

    def preprocess_trainset_index_pretrain(self, protein_id_dict=None, protein_id_list=None, id_map_for_filtering=None,
                                           score_bar=None, is_filtering=True):
        if id_map_for_filtering is not None:
            self.pose_id_map2 = id_map_for_filtering

        if score_bar is not None and self.pose_id_map2 is not None and self.labels_classification is not None:
            filtered_id_all = [key for key, value in self.pose_id_map2.items() if
                               self.labels_classification[key] > score_bar]
            self.pose_id_map2 = {key: i for i, key in enumerate(filtered_id_all)}
            # self.pose_id_map2 = {
            #     key: value
            #     for key, value in id_map_for_filtering.items()
            #     if self.labels_classification[key] > score_bar
            # }

        self.labels_class = [self.protein_id_list[i] for i in self.pose_id_map2.keys()] if (
                    self.pose_id_map2 is not None and is_filtering) else self.protein_id_list
        # aaa=[i for i in self.labels_classification if i >score_bar]
        if protein_id_dict is not None and protein_id_list is not None:
            target_protein_id_dict = protein_id_dict
            target_protein_id_list = protein_id_list
        else:
            target_protein_id_dict = self.protein_id_dict
            target_protein_id_list = self.protein_id_list
        bad_id_list_all = [target_protein_id_dict[name] for name in target_protein_id_dict.keys() if
                           name.lower().endswith('bad')]
        if self.dataset_map is None:

            # dataset_id_map=None
            id_map = None
            bad_id_list = None
        else:
            # id_index_dict = {target_protein_id_dict[name]: [] for name in self.dataset_map.keys()}
            id_map = {target_protein_id_dict[name]: target_protein_id_dict[name2] if name2 is not None else None for
                      name, name2 in self.dataset_map.items()}
            bad_id_list = [target_protein_id_dict[name] for name in self.dataset_map.keys() if
                           name.lower().endswith('bad')]

        dataset_id_map = {'id_map': id_map, 'bad_id_list': bad_id_list, 'bad_id_list_all': bad_id_list_all}
        # for i, id in enumerate(self.protein_id_list):
        #     id_index_dict[id].append(i)
        id_index_dict = {id: [] for id in target_protein_id_dict.values()}
        id_scores_dict = {}
        id_score_source_dict = {}
        id_used_default_score_dict = {}
        id_data_source_dict = {}
        scores_np = np.array(self.labels_classification)
        score_source_np = np.array(self.labels_score_source)
        used_default_score_np = np.array(self.labels_used_default_score)
        data_source_np = np.array(self.labels_data_source)
        protein_id_list_np = np.array(target_protein_id_list)
        for id in target_protein_id_dict.values():
            # aaa = np.where(protein_id_list_np == id)
            # id_index_dict[id] = np.where(protein_id_list_np == id)[0].tolist()
            id_selected = np.where(protein_id_list_np == id)[0].tolist()
            if self.pose_id_map2 is not None and is_filtering:
                id_index_dict[id] = [item for item in id_selected if item in self.pose_id_map2.keys()]
            else:
                id_index_dict[id] = id_selected
            id_scores_dict[id] = scores_np[id_index_dict[id]]
            id_score_source_dict[id] = score_source_np[id_index_dict[id]]
            id_used_default_score_dict[id] = used_default_score_np[id_index_dict[id]]
            id_data_source_dict[id] = data_source_np[id_index_dict[id]]
        self.id_index_dict = id_index_dict
        self.id_score_source_dict = id_score_source_dict
        self.id_used_default_score_dict = id_used_default_score_dict
        self.id_data_source_dict = id_data_source_dict
        return id_index_dict, dataset_id_map, id_scores_dict


class CryoEMDataset(Dataset):
    """自定义数据集"""

    def __init__(self, metadata: CryoMetaData, transform=None,
                 normal_scale=10, accelerator=None,
                 local_crops=None,
                 slice_setting=None,
                 mix_pos_setting=None,
                 weight_for_contrastive_classification_label=0.0,
                 use_triplex_labels=False, bar_score=0.0,
                 in_chans=1, needs_aug2=False,
                 pretrain_128=False,
                 patch_size=14,
                 mask_setting=None
                 ):
        self.pose_indices = AnnoyIndex(2, 'euclidean')
        self.tif_len = metadata.length
        self.lmdb_path = metadata.lmdb_path
        self.pretrain_128 = pretrain_128

        self.patch_size = patch_size
        self.mask_setting = mask_setting

        self.protein_id_list = metadata.protein_id_list
        self.protein_id_dict = metadata.protein_id_dict
        self.protein_id_dict_reverse = {v: k for k, v in self.protein_id_dict.items()}
        self.lmdb_reference_manifest = getattr(metadata, 'lmdb_reference_manifest', None)
        self.lmdb_reference_segments = {}
        self.protein_min_index = {}
        self.id_index_dict = metadata.id_index_dict or self._build_default_id_index_dict()

        if self.protein_id_list is not None:
            self.protein_min_index = self._build_protein_min_index()

        if self.lmdb_reference_manifest is not None:
            self.metadata = []
            self.cumulative_sizes = []
            self.worker_id = None
            self.env_processed = {}
            self.env_raw = {}
            self.env_FT = {}
            self._build_reference_lmdb_metadata()
        elif self.lmdb_path is not None:
            lmdb_dir_name_list = list(self.protein_id_dict.keys())
            lmdb_dir = self.lmdb_path
            # self.lmdb_dir = lmdb_dir

            # self.lmdb_paths = sorted(
            #     [os.path.join(lmdb_dir, name) for name in lmdb_dir_name_list if
            #      os.path.isdir(os.path.join(lmdb_dir, name))])
            self.lmdb_paths = [os.path.join(lmdb_dir, name) for name in lmdb_dir_name_list if
                               os.path.isdir(os.path.join(lmdb_dir, name))]
            if not self.lmdb_paths:
                raise ValueError(f"No LMDB directories found in {lmdb_dir}")

            self.metadata = []  # 存储每个LMDB的信息：(路径, 包含的样本数)
            self.cumulative_sizes = [0]  # 存储样本数量的累加和，用于快速定位全局索引

            # print("Scanning LMDB files and building index...")
            # 1. 遍历所有LMDB路径，只为获取样本数量，然后立刻关闭
            for path in self.lmdb_paths:
                try:
                    env = lmdb.open(os.path.join(path, 'lmdb_processed'), readonly=True, lock=False, readahead=False,
                                    meminit=False)
                    with env.begin() as txn:
                        num_samples = txn.stat()['entries']
                    env.close()

                    self.metadata.append((path, num_samples))
                    self.cumulative_sizes.append(self.cumulative_sizes[-1] + num_samples)
                except lmdb.Error as e:
                    print(f"Warning: Could not read LMDB at {path}. Skipping. Error: {e}")

            # 移除起始的0
            self.cumulative_sizes.pop(0)

            total_samples = self.cumulative_sizes[-1] if self.cumulative_sizes else 0
            # print(f"Found {len(self.lmdb_paths)} LMDBs with a total of {total_samples} samples.")

            # 2. 核心：不在这里打开任何env，只在需要时打开
            # self.open_envs = {}  # 用于缓存已打开的LMDB环境
            self.worker_id = None  # 用于多进程DataLoader
            self.env_processed = {}
            self.env_raw = {}
            self.env_FT = {}

        # if mrcdata.lmdb_path is not None:
        #     self.lmdb_env=lmdb.open(
        #         mrcdata.lmdb_path,
        #         readonly=True,
        #         # lock=False,
        #         # readahead=False
        #     )
        #     self.processed_tif_txn = self.lmdb_env.begin()
        # # if mrcdata.lmdb_env is not None:
        # #     self.processed_tif_txn = mrcdata.lmdb_env.begin()
        # else:
        #     self.processed_tif_txn = None
        # self.processed_tif_txn = mrcdata.processed_tif_txn
        self.labels_for_clustering = metadata.labels_for_clustering
        self.labels_classification = metadata.labels_classification
        self.labels_score_source = metadata.labels_score_source
        self.labels_used_default_score = metadata.labels_used_default_score
        self.labels_data_source = metadata.labels_data_source
        self.source_mrcs_group_id = getattr(metadata, 'source_mrcs_group_id', None)
        self.source_mrcs_group_key = getattr(metadata, 'source_mrcs_group_key', None)
        self.source_mrcs_group_index = self._build_source_mrcs_group_index()

        self.slice_setting = slice_setting
        self.mix_pos_setting = mix_pos_setting
        self.et_pose_search_N = self._get_et_pose_search_N()
        self.in_chans = in_chans
        self.needs_aug2 = needs_aug2

        self.use_triplex_labels = use_triplex_labels
        if bar_score < 0.5:
            self.bar_score = 1 - bar_score
        else:
            self.bar_score = bar_score

        if metadata.particles_id is not None:
            self.particles_id = metadata.particles_id
        else:
            self.particles_id = range(self.tif_len)
        # self.isnorm = is_Normalize
        # self.mean_std = mrcdata.means_stds
        self.normal_scale = normal_scale
        self.transform = transform
        self.local_crops_transform = None
        self.random_rotate_transform = None
        self.mic_transform = None
        self.mic_crop = None
        self.mix_pos_transforms = None
        self.random_rotate_transform_mix_pos = None
        self.transform_mix_pos = None
        self.accelerator = accelerator
        # self.train=True

        # self.labels_for_training = mrcdata.labels_for_training
        # self.probabilities_for_sampling = mrcdata.probabilities_for_sampling
        self.processed_data_path = metadata.processed_data_path

        if local_crops is not None:
            self.local_crops_number = local_crops['number']
            self.local_crops_only_mics = local_crops.get('only_mics', False)
        else:
            self.local_crops_number = 0
            self.local_crops_only_mics = False

        self.weight_for_contrastive_classification_label = weight_for_contrastive_classification_label
        if weight_for_contrastive_classification_label > 0:
            labels_classification_np = np.array(self.labels_classification)
            self.positive_items = np.where(labels_classification_np == 1)[0]
            self.negative_items = np.where(labels_classification_np == 0)[0]
        self.pose_id_map = metadata.pose_id_map

    def __len__(self):
        return self.tif_len

    def _build_default_id_index_dict(self):
        id_index_dict = {protein_id: [] for protein_id in self.protein_id_dict.values()}
        for index, protein_id in enumerate(self.protein_id_list):
            id_index_dict.setdefault(protein_id, []).append(index)
        return id_index_dict

    def _build_source_mrcs_group_index(self):
        if self.source_mrcs_group_id is None:
            return None
        source_mrcs_group_index = {}
        for index, group_id in enumerate(self.source_mrcs_group_id):
            source_mrcs_group_index.setdefault(group_id, []).append(index)
        return source_mrcs_group_index

    def _get_et_pose_search_N(self):
        if self.mix_pos_setting is None:
            return None
        value = self.mix_pos_setting.get('et_pose_search_N', self.mix_pos_setting.get('pose_search_N'))
        if value is None:
            return None
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise ValueError('et_pose_search_N must be a positive integer.')
        if value <= 0:
            raise ValueError('et_pose_search_N must be a positive integer.')
        return value

    def _build_protein_min_index(self):
        protein_min_index = {}
        for index, protein_id in enumerate(self.protein_id_list):
            protein_min_index.setdefault(protein_id, index)
        return protein_min_index

    def _get_lmdb_entry_count(self, lmdb_dir):
        env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False)
        try:
            with env.begin(write=False) as txn:
                return txn.stat()['entries']
        finally:
            env.close()

    def _build_reference_lmdb_metadata(self):
        manifest = self.lmdb_reference_manifest
        proteins = manifest.get('proteins') if isinstance(manifest, dict) else None
        if not isinstance(proteins, list):
            raise ValueError(f'{LMDB_REFERENCE_MANIFEST_FILENAME} must contain a proteins list.')

        seen_protein_ids = set()
        for protein_entry in proteins:
            protein_name = protein_entry.get('protein_name')
            protein_id = protein_entry.get('protein_id')
            if protein_name not in self.protein_id_dict:
                raise ValueError(
                    f"{LMDB_REFERENCE_MANIFEST_FILENAME} contains unknown protein '{protein_name}'."
                )
            expected_protein_id = self.protein_id_dict[protein_name]
            if protein_id != expected_protein_id:
                raise ValueError(
                    f"{LMDB_REFERENCE_MANIFEST_FILENAME} has protein_id={protein_id} for '{protein_name}', "
                    f'expected {expected_protein_id}.'
                )

            segments = protein_entry.get('segments')
            if not isinstance(segments, list) or not segments:
                raise ValueError(
                    f"{LMDB_REFERENCE_MANIFEST_FILENAME} contains no segments for '{protein_name}'."
                )

            normalized_segments = []
            expected_local_start = 0
            for segment in sorted(segments, key=lambda item: item.get('merged_local_start', -1)):
                merged_local_start = int(segment.get('merged_local_start', -1))
                count = int(segment.get('count', -1))
                if merged_local_start != expected_local_start:
                    raise ValueError(
                        f"{LMDB_REFERENCE_MANIFEST_FILENAME} has non-contiguous segments for '{protein_name}'."
                    )
                if count <= 0:
                    raise ValueError(
                        f"{LMDB_REFERENCE_MANIFEST_FILENAME} has invalid count={count} for '{protein_name}'."
                    )

                db_paths = segment.get('db_paths')
                if not isinstance(db_paths, dict) or 'lmdb_processed' not in db_paths:
                    raise ValueError(
                        f"{LMDB_REFERENCE_MANIFEST_FILENAME} segment for '{protein_name}' is missing lmdb_processed."
                    )

                processed_dir = db_paths['lmdb_processed']
                raw_dir = db_paths.get('lmdb_raw')
                ft_dir = db_paths.get('lmdb_FT')

                if not os.path.isdir(processed_dir):
                    raise FileNotFoundError(
                        f"Referenced LMDB directory does not exist for '{protein_name}': {processed_dir}"
                    )

                entry_count = self._get_lmdb_entry_count(processed_dir)
                source_local_indices = segment.get('source_local_indices')
                if source_local_indices is None:
                    if entry_count != count:
                        raise ValueError(
                            f"Referenced LMDB count mismatch for '{protein_name}' at {processed_dir}: "
                            f'expected {count}, found {entry_count}.'
                        )
                else:
                    if not isinstance(source_local_indices, list) or len(source_local_indices) != count:
                        raise ValueError(
                            f"{LMDB_REFERENCE_MANIFEST_FILENAME} source_local_indices length mismatch "
                            f"for '{protein_name}'."
                        )
                    normalized_source_indices = []
                    for source_index in source_local_indices:
                        source_index = int(source_index)
                        if source_index < 0 or source_index >= entry_count:
                            raise ValueError(
                                f"{LMDB_REFERENCE_MANIFEST_FILENAME} source index {source_index} "
                                f"for '{protein_name}' is outside referenced LMDB entry count {entry_count}."
                            )
                        normalized_source_indices.append(source_index)
                    source_local_indices = normalized_source_indices

                normalized_segments.append(
                    {
                        'merged_local_start': merged_local_start,
                        'count': count,
                        'processed_dir': processed_dir,
                        'raw_dir': raw_dir,
                        'ft_dir': ft_dir,
                        'source_local_indices': source_local_indices,
                    }
                )
                expected_local_start += count

            expected_count = len(self.id_index_dict.get(protein_id, []))
            if expected_local_start != expected_count:
                raise ValueError(
                    f"{LMDB_REFERENCE_MANIFEST_FILENAME} particle count mismatch for '{protein_name}': "
                    f'expected {expected_count}, found {expected_local_start}.'
                )

            self.lmdb_reference_segments[protein_id] = normalized_segments
            seen_protein_ids.add(protein_id)

        expected_protein_ids = set(self.protein_id_dict.values())
        if seen_protein_ids != expected_protein_ids:
            missing_ids = sorted(expected_protein_ids - seen_protein_ids)
            raise ValueError(
                f"{LMDB_REFERENCE_MANIFEST_FILENAME} is missing proteins for ids: {missing_ids}."
            )

    def _get_protein_local_index(self, protein_id, item):
        item_list = self.id_index_dict.get(protein_id, [])
        local_index = bisect_left(item_list, item)
        if local_index >= len(item_list) or item_list[local_index] != item:
            raise IndexError(item)
        return local_index

    def _get_manifest_segment(self, protein_id, local_index):
        segments = self.lmdb_reference_segments.get(protein_id)
        if not segments:
            raise ValueError(f'No LMDB reference segments found for protein id {protein_id}.')
        for segment in segments:
            start = segment['merged_local_start']
            end = start + segment['count']
            if start <= local_index < end:
                return segment
        raise IndexError(local_index)

    # def open_lmdb(self):
    #     # if mrcdata.lmdb_path is not None:
    #     self.lmdb_env = lmdb.open(
    #         self.lmdb_path,
    #         readonly=True,
    #         meminit=False,
    #         max_readers=1,
    #         lock=False,
    #         readahead=False
    #     )

    # @profile(precision=4)
    def __getitem__(self, item):
        '''get mrcdata1 and aug1'''
        mrcdata = self.get_mrcdata(item=item)

        weight = float(1.0)

        '''get labels data'''
        if self.labels_for_clustering is not None and len(self.labels_for_clustering) > item:
            label_for_clustering = self.labels_for_clustering[item]
        else:
            label_for_clustering = -1
        label_for_classification = self.labels_classification[item]
        label_score_source = self.labels_score_source[item]
        label_used_default_score = self.labels_used_default_score[item]
        label_data_source = self.labels_data_source[item]
        if self.use_triplex_labels:
            if label_for_classification > self.bar_score:
                label_for_classification = 1.0
            elif label_for_classification <= 1 - self.bar_score:
                label_for_classification = 0.0
            else:
                label_for_classification = 0.5

        protein_id = self.protein_id_list[item]

        '''get mrcdata2'''
        mrcdata_rotate1 = None  # 初始化
        mrcdata_rotate2 = None  # 初始化

        # labels_data_source.data records source modality per item:
        # ptcls/cryo-EM particles, mics/cryo-EM micrographs, et_tilts/cryo-ET tilts,
        # et_ptcls/extracted cryo-ET particles. Legacy datasets default to ptcls;
        # the image-size fallback only catches old unlabeled micrograph datasets.
        is_mics = label_data_source in MICROGRAPH_LIKE_DATA_SOURCES
        if label_data_source == DATA_SOURCE_PTCLS and getattr(mrcdata, 'size', None) is not None:
            is_mics = True if mrcdata.size[-1] > MICROGRAPH_SIZE_THRESHOLD else False

        if self.needs_aug2:

            is_random_rotate_transform =False
            is_mix_pos=False

            if is_mics:
                if self.mic_crop is not None:
                    mrcdata=self.mic_crop(mrcdata)
                mrcdata2 = mrcdata
            else:
                if self.random_rotate_transform is not None:
                    is_random_rotate_transform = True
                item2, weight, is_mix_pos = self.get_item2(item, label_data_source=label_data_source)
                if item2 is not None:
                    mrcdata2 = self.get_mrcdata(item=item2)
                else:
                    mrcdata2 = mrcdata

            # 接收返回的 rotate image
            aug1, mrcdata_rotate1 = self.mrcdata_aug(mrcdata, is_random_rotate_transform=is_random_rotate_transform,
                                                     is_mix_pos=is_mix_pos,
                                                     is_mics=is_mics)
            aug2, mrcdata_rotate2 = self.mrcdata_aug(mrcdata2,
                                                     is_random_rotate_transform=is_random_rotate_transform,
                                                     is_mix_pos=is_mix_pos,
                                                     is_mics=is_mics)
        else:
            # 接收返回的 rotate image
            if is_mics:
                if self.mic_crop is not None:
                    mrcdata = self.mic_crop(mrcdata)
            aug1, mrcdata_rotate1 = self.mrcdata_aug(mrcdata,is_mics=is_mics)
            aug2 = None
            mrcdata_rotate2 = None

        # === Generate Local Crops ===
        # Only generate local crops if enabled and (not only_mics OR data is micrographs)
        should_apply_local_crops = self.local_crops_number > 0 and (not self.local_crops_only_mics or is_mics)
        if should_apply_local_crops:
            local_crops1, local_crops2 = self.get_local_crops(mrcdata_rotate1, mrcdata_rotate2)
        else:
            local_crops1, local_crops2 = [], []
        # ===============================

        # Generate mask based on settings
        # Always generate a mask when MIM is enabled to ensure consistent DDP behavior
        # When only_mics=True and data is particles, generate an all-zero mask (no masking)
        if self.mask_setting is not None and self.mask_setting['is_add_mim_loss']:
            should_apply_masking = not self.mask_setting.get('only_mics', False) or is_mics
            if should_apply_masking:
                mask = self.get_mask(setting=self.mask_setting, W=aug1.shape[1] // self.patch_size,
                                     H=aug1.shape[2] // self.patch_size)
            else:
                # Generate all-zero mask for particles when only_mics=True
                # This ensures predictor_mim is always used consistently for DDP
                mask = np.zeros((aug1.shape[1] // self.patch_size, aug1.shape[2] // self.patch_size), dtype=np.int16)
        else:
            mask = []

        out = {
            'aug1': aug1,
            'aug2': aug2 if aug2 is not None else [],
            'weight': weight,
            'label_for_clustering': label_for_clustering,
            'label_for_classification': label_for_classification,
            'label_score_source': label_score_source,
            'label_used_default_score': label_used_default_score,
            'label_data_source': label_data_source,
            'mask': mask,
            'item': item,
            'local_crops1': local_crops1, 
            'local_crops2': local_crops2,
            'protein_id': protein_id
        }
        return out

    def get_mask(self,setting,W,H):
        mask = np.zeros(W*H, dtype=np.int16)

        if random.random() < setting['p']:

            mask_ratio=random.uniform(setting['mask_ratio'][0],setting['mask_ratio'][1] )
            num_mask=int(W*H*mask_ratio)
            mask_indices=random.sample(range(W*H),num_mask)
            for idx in mask_indices:
                mask[idx]=1
        mask=mask.reshape(W,H)

        return mask





    def get_item2(self, item, label_data_source=None):
        item2 = None
        weight = 1
        is_mix_pos = False
        if label_data_source is None and self.labels_data_source is not None:
            label_data_source = self.labels_data_source[item]
        if self.weight_for_contrastive_classification_label > 0:
            if random.random() < self.weight_for_contrastive_classification_label:
                if self.labels_classification[item] == 1:
                    item2 = np.random.choice(self.positive_items)
                else:
                    item2 = np.random.choice(self.negative_items)
        elif self.mix_pos_setting is not None and self.mix_pos_setting['p'] > 0 and self.pose_id_map is not None:
            if random.random() < self.mix_pos_setting['p']:
                if label_data_source == DATA_SOURCE_ET_PTCLS:
                    item2, weight = self.get_nearest_item_same_mrcs(item)
                    is_mix_pos = item2 is not None
                else:
                    # protein_id = self.protein_id_list[item]
                    # item_list=self.id_index_dict.get(protein_id, [])
                    # if len(item_list) > 1:
                    #     item2 = random.choice(item_list)
                    # protein_name=self.protein_id_dict_reverse[protein_id]
                    # self.pose_indices.load(os.path.join(self.processed_data_path, 'pose_data', protein_name + '_pose.ann'))
                    # nearest=self.pose_indices.get_nns_by_item(item-min(item_list), int(len(item_list)/20),include_distances=False)
                    # item2 = random.choice(nearest[1:])+ min(item_list)
                    nearest, min_id, protein_name, pose_items_id, item1_pose_id = self.get_nearest_item(item)
                    if nearest is not None and len(nearest) > 1:
                        item2_pose_id = weighted_random_choice_linear(nearest[1:], with_weight=False)
                        item2 = pose_items_id[item2_pose_id] + min_id

                        # if item-min(item_list)<0:
                        #     print('item is less than min(item.list): ' + str(item) + ' ' + str(min(item_list)))
                        #     print('protein_name: ' + protein_name)
                        # if item2-min(item_list)<0:
                        #     print('item2 is less than min(item.list): ' + str(item2) + ' ' + str(min(item_list)))
                        #     print('protein_name: ' + protein_name)
                        weight = sigmoid(self.mix_pos_setting['sigma'] * (
                                (3.5 - self.pose_indices.get_distance(item1_pose_id, item2_pose_id)) / 3.5
                                - self.mix_pos_setting['bias']))
                        self.pose_indices.unload()
                        is_mix_pos = True

                # if protein_name=='11307_J504_good':
                #     pass
        return item2, float(weight), is_mix_pos

    def get_nearest_item_same_mrcs(self, item, N=None):
        if N is None:
            N = self.et_pose_search_N
        if N is None:
            return None, 1
        N = int(N)
        if N <= 0:
            raise ValueError('et_pose_search_N must be a positive integer.')
        if self.source_mrcs_group_id is None or self.source_mrcs_group_index is None:
            return None, 1
        if item >= len(self.source_mrcs_group_id):
            return None, 1

        protein_id = self.protein_id_list[item]
        item_list = self.id_index_dict.get(protein_id, [])
        min_id = self.protein_min_index.get(protein_id, min(item_list) if item_list else 0)
        protein_pose_map = self.pose_id_map.get(protein_id) if isinstance(self.pose_id_map, dict) else None
        if protein_pose_map is None:
            return None, 1

        protein_name = self.protein_id_dict_reverse[protein_id]
        pose_file_path = os.path.join(self.processed_data_path, 'pose_data', protein_name + '_pose.ann')
        if not os.path.exists(pose_file_path):
            return None, 1

        local_item_id = item - min_id
        if local_item_id not in protein_pose_map:
            return None, 1
        item1_pose_id = protein_pose_map[local_item_id]

        group_id = self.source_mrcs_group_id[item]
        same_group_items = self.source_mrcs_group_index.get(group_id, [])
        candidate_distances = []
        self.pose_indices.load(pose_file_path)
        try:
            for candidate_item in same_group_items:
                if candidate_item == item:
                    continue
                if self.protein_id_list[candidate_item] != protein_id:
                    continue
                candidate_local_id = candidate_item - min_id
                if candidate_local_id not in protein_pose_map:
                    continue
                candidate_pose_id = protein_pose_map[candidate_local_id]
                distance = self.pose_indices.get_distance(item1_pose_id, candidate_pose_id)
                candidate_distances.append((distance, candidate_item, candidate_pose_id))
        finally:
            self.pose_indices.unload()

        if not candidate_distances:
            return None, 1

        candidate_distances.sort(key=lambda x: x[0])
        selected_distance, selected_item, _ = random.choice(candidate_distances[:N])
        weight = sigmoid(self.mix_pos_setting['sigma'] * (
                (3.5 - selected_distance) / 3.5
                - self.mix_pos_setting['bias']))
        return selected_item, float(weight)

    def get_nearest_item(self, item, N=None, pose_divide=None):

        if N is None:
            N = self.mix_pos_setting['pose_search_N']
        if pose_divide is None:
            pose_divide = self.mix_pos_setting['pose_search_divide']

        protein_id = self.protein_id_list[item]
        item_list = self.id_index_dict.get(protein_id, [])
        min_id = self.protein_min_index.get(protein_id, min(item_list) if item_list else 0)

        nearest = None
        protein_name = self.protein_id_dict_reverse[protein_id]
        pose_items_id = []
        item1_pose_id = None
        protein_pose_map = self.pose_id_map.get(protein_id) if isinstance(self.pose_id_map, dict) else None
        if protein_pose_map is None:
            return nearest, min_id, protein_name, pose_items_id, item1_pose_id

        pose_file_path = os.path.join(self.processed_data_path, 'pose_data', protein_name + '_pose.ann')
        if not os.path.exists(pose_file_path):
            return nearest, min_id, protein_name, pose_items_id, item1_pose_id

        local_item_id = item - min_id
        if local_item_id in protein_pose_map:
            item1_pose_id = protein_pose_map[local_item_id]
            self.pose_indices.load(pose_file_path)
            nearest = self.pose_indices.get_nns_by_item(
                item1_pose_id,
                int(len(item_list) / pose_divide) if N is None else N,
                include_distances=False,
            )
            pose_items_id = list(protein_pose_map.keys())
        return nearest, min_id, protein_name, pose_items_id, item1_pose_id

    def get_mrcdata(self, item=None, tif_path=None):
        mrcdata = None
        if tif_path is not None:
            raise ValueError('Direct particle paths are no longer supported; load particles from LMDB.')
        if item is not None:
            if self.lmdb_reference_manifest is not None:
                protein_id = self.protein_id_list[item]
                local_index = self._get_protein_local_index(protein_id, item)
                segment = self._get_manifest_segment(protein_id, local_index)
                segment_local_index = local_index - segment['merged_local_start']
                if segment.get('source_local_indices') is None:
                    source_local_index = segment_local_index
                else:
                    source_local_index = segment['source_local_indices'][segment_local_index]

                raw_env, processed_env, _ = self._get_env(
                    segment['processed_dir'],
                    raw_dir=segment.get('raw_dir'),
                    processed_dir=segment['processed_dir'],
                    ft_dir=segment.get('ft_dir'),
                    use_raw=self.pretrain_128,
                )
                lmdb_env = raw_env if self.pretrain_128 else processed_env
                with lmdb_env.begin(write=False) as txn:
                    key = f"{source_local_index}".encode()
                    value = txn.get(key)
                    if value is None:
                        raise KeyError(
                            f"Referenced LMDB {segment['processed_dir']} is missing expected key {source_local_index}."
                        )
                    data = pickle.loads(value)
                    mrcdata = data
                    tif_path = ''
                del data, value
            elif self.lmdb_path is not None:
                # if not hasattr(self, 'lmdb_env'):
                #     self.open_lmdb()
                index = item
                lmdb_idx = 0
                while index >= self.cumulative_sizes[lmdb_idx]:
                    lmdb_idx += 1

                lmdb_path, _ = self.metadata[lmdb_idx]

                # 2. 计算在该LMDB中的局部索引
                prev_size = self.cumulative_sizes[lmdb_idx - 1] if lmdb_idx > 0 else 0
                local_idx = index - prev_size

                # 3. 获取（可能需要懒加载）对应的LMDB环境
                lmdb_env_r, lmdb_env_p, _ = self._get_env(
                    lmdb_path,
                    use_raw=self.pretrain_128,
                )
                if self.pretrain_128:
                    lmdb_env = lmdb_env_r
                else:
                    lmdb_env = lmdb_env_p
                with lmdb_env.begin(write=False) as txn:
                    key = f"{local_idx}".encode()
                    value = txn.get(key)
                    if value is None:
                        raise KeyError(f"LMDB {lmdb_path} is missing expected key {local_idx}.")
                    data = pickle.loads(value)
                    mrcdata = data
                    tif_path = ''
                    # raw_tif_path = ''
                del data, value
            else:
                raise RuntimeError('CryoEMDataset requires LMDB-backed metadata.')
        return mrcdata

    def mrcdata_aug(self, mrcdata, is_random_rotate_transform=True, is_mix_pos=False,is_mics=False):
        if isinstance(mrcdata, np.ndarray):
            mrcdata = Image.fromarray(mrcdata)
        # # if mrcdata.mode != 'L':
        #     mrcdata = to_int8(mrcdata)
        mrcdata_rotate1 = mrcdata
        aug=mrcdata
        if is_mics:
            if self.mic_transform is not None:
                aug = self.mic_transform(mrcdata)
        else:

            # 1. 处理旋转
            if is_random_rotate_transform:
                if is_mix_pos and self.random_rotate_transform_mix_pos is not None:
                    mrcdata_rotate1 = self.random_rotate_transform_mix_pos(mrcdata)
                elif self.random_rotate_transform is not None:
                    mrcdata_rotate1 = self.random_rotate_transform(mrcdata)

            # 2. 生成 Global View (aug)
            if is_mix_pos:
                if self.transform_mix_pos is not None:
                    aug = self.transform_mix_pos(mrcdata_rotate1)
            elif self.transform is not None:
                aug = self.transform(mrcdata_rotate1)

            # 修改点：同时返回 最终aug tensor 和 中间状态的 rotate image
        return aug, mrcdata_rotate1

    def get_transforms(self, transforms, transforms_list_mix_pos=None):
        if transforms is None:
            self.transform = None
            self.local_crops_transform = None
            self.random_rotate_transform = None
            self.mic_transform = None
            self.mic_crop = None
            self.mix_pos_transforms = None
            self.random_rotate_transform_mix_pos = None 
            self.transform_mix_pos = None
        elif isinstance(transforms, (list, tuple)):
            # Backward-compatible transform layout used by older Cryo-IEF/CryoRanker
            # helpers: [particle_transform, random_rotate_transform, local_crops_transform].
            self.transform = transforms[0] if len(transforms) > 0 else None
            self.random_rotate_transform = transforms[1] if len(transforms) > 1 else None
            self.local_crops_transform = transforms[2] if len(transforms) > 2 else None
            self.mic_transform = None
            self.mic_crop = None
            if transforms_list_mix_pos is not None:
                self.random_rotate_transform_mix_pos = (
                    transforms_list_mix_pos[1] if len(transforms_list_mix_pos) > 1 else None
                )
                self.transform_mix_pos = (
                    transforms_list_mix_pos[0] if len(transforms_list_mix_pos) > 0 else None
                )
                self.mix_pos_transforms = transforms_list_mix_pos
            else:
                self.mix_pos_transforms = None
                self.random_rotate_transform_mix_pos = None
                self.transform_mix_pos = None
        else:
            self.transform = transforms['ptcls'] if 'ptcls' in transforms else None
            self.local_crops_transform = transforms['local_crops'] if 'local_crops' in transforms else None
            self.random_rotate_transform = transforms['random_rotate'] if 'random_rotate' in transforms else None
            self.mic_transform = transforms['mics'] if 'mics' in transforms else None
            self.mic_crop= transforms['random_resized_crop_all'] if 'random_resized_crop_all' in transforms else None
            if transforms_list_mix_pos is not None:
                # self.mix_pos_transforms = transforms_list_mix_pos
                self.random_rotate_transform_mix_pos = transforms['random_rotate'] if 'random_rotate' in transforms else None
                self.transform_mix_pos = transforms['ptcls'] if 'ptcls' in transforms else None
            else:
                self.mix_pos_transforms = None
                self.random_rotate_transform_mix_pos = None
                self.transform_mix_pos = None

    def get_local_crops(self, mrcdata_rotate1, mrcdata_rotate2=None):
        local_crops1 = []
        local_crops2 = []

        # 生成 crop
        for _ in range(self.local_crops_number):
            local_crops1.append(self.local_crops_transform(mrcdata_rotate1))
            if self.needs_aug2 and mrcdata_rotate2 is not None:
                local_crops2.append(self.local_crops_transform(mrcdata_rotate2))

        # 处理通道数 (如果输入不是单通道)
        if self.in_chans != 1:
            # 注意：这里只处理 local crops，aug1/aug2 的处理应在外部或 transform 中完成
            local_crops1 = [local_crop.repeat(self.in_chans, 1, 1) for local_crop in local_crops1]
            if self.needs_aug2 and mrcdata_rotate2 is not None:
                local_crops2 = [local_crop.repeat(self.in_chans, 1, 1) for local_crop in local_crops2]

        return local_crops1, local_crops2

    def _get_env(self, lmdb_path, raw_dir=None, processed_dir=None, ft_dir=None, use_raw=False, use_processed=True, use_FT=False):
        """
        懒加载和缓存LMDB环境的辅助函数。
        """
        # 在PyTorch DataLoader的多进程模式下，每个worker是独立的进程。
        # 我们需要在每个worker中维持自己的环境缓存。
        worker_info = torch.utils.data.get_worker_info()
        current_worker_id = worker_info.id if worker_info else 0

        # 如果切换了worker，清空旧的缓存
        if self.worker_id != current_worker_id:
            self.worker_id = current_worker_id
            # for env in self.open_envs.values():
            #     env.close()
            # self.open_envs.clear()
            # for env_raw, env_processed, env_FT in zip(self.env_raw.values(), self.env_processed.values(), self.env_FT.values()):
            #     env_raw.close()
            #     env_processed.close()
            #     env_FT.close()
            # self.env_raw.clear()
            # self.env_processed.clear()
            # self.env_FT.clear()
            if use_processed:
                for env_processed in self.env_processed.values():
                    env_processed.close()
                    self.env_processed.clear()
            if use_raw:
                for env_raw in self.env_raw.values():
                    env_raw.close()
                    self.env_raw.clear()
            if use_FT:
                for env_FT in self.env_FT.values():
                    env_FT.close()
                    self.env_FT.clear()

        processed_env_key = processed_dir if processed_dir is not None else os.path.join(lmdb_path, 'lmdb_processed')

        if use_raw:
            if raw_dir is not None:
                raw_env_key = raw_dir
            elif processed_dir is not None or ft_dir is not None:
                raise ValueError(
                    'LMDB reference manifest segment does not provide lmdb_raw for this protein segment.'
                )
            else:
                raw_env_key = os.path.join(lmdb_path, 'lmdb_raw')
        else:
            raw_env_key = None

        if use_FT:
            if ft_dir is not None:
                ft_env_key = ft_dir
            elif processed_dir is not None or raw_dir is not None:
                raise ValueError(
                    'LMDB reference manifest segment does not provide lmdb_FT for this protein segment.'
                )
            else:
                ft_env_key = os.path.join(lmdb_path, 'lmdb_FT')
        else:
            ft_env_key = None

        # 检查缓存中是否已有此LMDB的环境
        # if lmdb_path not in self.open_envs:
        #     # 如果没有，就打开它并存入缓存
        #     # readonly=True, lock=False 对于多进程读取是安全且高效的
        #     env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        #     self.open_envs[lmdb_path] = env
        if use_processed and processed_env_key not in self.env_processed:
            # 如果没有，就打开它并存入缓存
            # readonly=True, lock=False 对于多进程读取是安全且高效的
            env_processed = lmdb.open(processed_env_key, readonly=True, lock=False,
                                      readahead=False, meminit=False)
            self.env_processed[processed_env_key] = env_processed

        if use_raw and raw_env_key not in self.env_raw:
            env_raw = lmdb.open(raw_env_key, readonly=True, lock=False, readahead=False,
                                meminit=False)
            self.env_raw[raw_env_key] = env_raw

        if use_FT and ft_env_key not in self.env_FT:
            env_FT = lmdb.open(ft_env_key, readonly=True, lock=False, readahead=False,
                               meminit=False)
            self.env_FT[ft_env_key] = env_FT

        # return self.open_envs[lmdb_path]
        return (
            self.env_raw[raw_env_key] if use_raw else None,
            self.env_processed[processed_env_key] if use_processed else None,
            self.env_FT[ft_env_key] if use_FT else None,
        )


def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        elif os.path.splitext(file)[-1] == '.mrc' or os.path.splitext(file)[-1] == '.mrcs':
            list_name.append(file_path)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def weighted_random_choice_linear(my_list, with_weight=True):
    """
    使用线性递减的权重从列表中随机抽取一个元素。
    """

    if not my_list:
        return None

    if not with_weight:
        # 如果不需要权重，直接随机选择
        return random.choice(my_list)

    # 1. 生成权重列表 [len(my_list), len(my_list)-1, ..., 1]
    list_length = len(my_list)
    weights = list(range(list_length, 0, -1))
    # 或者使用列表推导式: weights = [list_length - i for i in range(list_length)]

    # 2. 使用 random.choices 进行加权抽样
    # k=1 表示只抽取一个元素，返回的是一个列表，所以用 [0] 获取该元素
    return random.choices(my_list, weights=weights, k=1)[0]
