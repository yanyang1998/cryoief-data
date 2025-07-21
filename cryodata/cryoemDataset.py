# from models.get_transformers import to_int8
from cryodata.data_preprocess.mrc_preprocess import to_int8
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
# from torchvision import transforms
import os
import pickle
import random
# import mrcfile
# from PIL import Image
import torch
from io import BytesIO
import json
import lmdb
from annoy import AnnoyIndex


# import time
# from torch.utils.data import DataLoader
# from memory_profiler import profile


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
                 is_extra_valset=False, accelerator=None, ctf_correction_averages=False,
                 ctf_correction_inference=False):
        super(CryoMetaData, self).__init__(emfile_path, selected_emfile_path)

        self.processed_data_path = processed_data_path
        self.id_index_dict = None


        assert processed_data_path is not None, "processed_data_path must be provided"
        self.load_preprocessed_data_path(data_path=processed_data_path,
                                         ctf_correction_averages=ctf_correction_averages,
                                         ctf_correction_train=ctf_correction_inference)

        # if processed_data_path is not None:
        #     # with open(os.path.join(processed_data_path, 'path_divided_by_labels.data'), 'rb') as filehandle:
        #     #     self.path_divided_by_labels = pickle.load(filehandle)
        #     # with open(os.path.join(processed_data_path, 'ids_divided_by_labels.data'), 'rb') as filehandle:
        #     #     self.ids_divided_by_labels = pickle.load(filehandle)
        #     self.load_preprocessed_data_path(data_path=processed_data_path,
        #                                      ctf_correction_averages=ctf_correction_averages,
        #                                      ctf_correction_train=ctf_correction_inference)
        # else:
        #     if is_extra_valset:
        #         tmp_data_save_path = tmp_data_save_path + '/tmp/preprocessed_data/extra_valset/'
        #     else:
        #         tmp_data_save_path = tmp_data_save_path + '/tmp/preprocessed_data/trainset/'
        #     self.mrc_path = mrc_path
        #     if not os.path.exists(tmp_data_save_path + '/output_tif_path.data') and accelerator.is_local_main_process:
        #         self.load_path()
        #         self.divided_into_single_mrc(tmp_data_save_path, resize=cfg['resize'], crop_ratio=cfg['crop_ratio'])
        #     accelerator.wait_for_everyone()
        #     self.load_preprocessed_data_path(data_path=tmp_data_save_path)

    def load_path(self):
        mrcs_path_list = []
        listdir(self.mrc_path, mrcs_path_list)
        self.mrcs_path_list = mrcs_path_list
        # if self.selected_particles_id is not None:
        # pass

    def load_preprocessed_data_path(self, data_path, ctf_correction_averages, ctf_correction_train):
        # path_out = path_result_dir + '/tmp/preprocessed_data/'
        if os.path.exists(data_path + '/output_tif_path.data'):
            with open(data_path + '/output_tif_path.data', 'rb') as filehandle:
                self.all_tif_path = pickle.load(filehandle)
        else:
            self.all_tif_path = None
        if ctf_correction_averages and os.path.exists(data_path + '/output_ctf_tif_path.data'):
            with open(data_path + '/output_ctf_tif_path.data', 'rb') as filehandle:
                self.all_tif_path_ctf_correction = pickle.load(filehandle)
        else:
            self.all_tif_path_ctf_correction = None

        if os.path.exists(data_path + '/lmdb_data'):
            # lmdb_env = lmdb.open(
            #     data_path + '/lmdb_data/lmdb_processed',
            #     readonly=True,
            #     lock=False,
            #     readahead=False
            # )
            # # self.lmdb_env = lmdb_env
            # processed_tif_txn = lmdb_env.begin()
            # self.length = processed_tif_txn.stat()['entries']
            # lmdb_env.close()
            with open(data_path + '/protein_id_list.data',
                      'rb') as filehandle:
                protein_id_list = pickle.load(filehandle)
            self.length = len(protein_id_list)
            self.lmdb_path = data_path + '/lmdb_data/'
            self.all_processed_tif_path = None
        else:
            # self.processed_tif_txn = None
            self.lmdb_env = None
            self.lmdb_path = None

            with open(data_path + '/output_processed_tif_path.data',
                      'rb') as filehandle:
                self.all_processed_tif_path = pickle.load(filehandle)
            self.length = len(self.all_processed_tif_path)
        if ctf_correction_train and os.path.exists(data_path + '/output_ctf_processed_tif_path.data'):
            with open(data_path + '/output_ctf_processed_tif_path.data', 'rb') as filehandle:
                self.all_processed_tif_path_ctf_correction = pickle.load(filehandle)
        else:
            self.all_processed_tif_path_ctf_correction = None

        if os.path.exists(data_path + '/labels_for_clustering.data'):
            with open(data_path + '/labels_for_clustering.data', 'rb') as filehandle:
                self.labels_for_clustering = pickle.load(filehandle)
        else:
            self.labels_for_clustering = None

        if os.path.exists(data_path + '/labels_classification.data'):
            with open(data_path + '/labels_classification.data', 'rb') as filehandle:
                self.labels_classification = pickle.load(filehandle)
            if len(self.labels_classification) == 0:
                self.labels_classification = [1] * self.length
        else:
            self.labels_classification = [1] * self.length

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



        # with open(data_path + '/labels_for_training.data',
        #           'rb') as filehandle:
        #     self.labels_for_training = pickle.load(filehandle)
        # with open(data_path + '/probabilities_for_sampling.data',
        #           'rb') as filehandle:
        #     self.probabilities_for_sampling = pickle.load(filehandle)

    # def preprocess_trainset_valset_index(self, valset_name=[], dataset_except_names=[], is_balance=False,
    #                                      max_resample_num=None, max_resample_num_val=None, ratio_balance_train=[0.35,0.3,0.35],
    #                                      max_number_per_sample=None):
    #     valset_name_id = [self.protein_id_dict[name] for name in valset_name]
    #     dataset_except_names_id = [self.protein_id_dict[name] for name in dataset_except_names]
    #     id_sum_dict = {id: 0 for id in self.protein_id_dict.values()}
    #     dataset_index = []
    #     valset_index = []
    #     positive_index = []
    #     middle_index = []
    #     negative_index = []
    #     resample_num_p = 0
    #     resample_num_m=0
    #     resample_num_n = 0
    #     if len(valset_name_id) > 0 or len(dataset_except_names_id) > 0:
    #         for i, protein_id in enumerate(self.protein_id_list):
    #             if protein_id in valset_name_id:
    #                 valset_index.append(i)
    #             elif protein_id not in dataset_except_names_id:
    #                 dataset_index.append(i)
    #     else:
    #         # dataset_index = list(range(len(self.all_processed_tif_path)))
    #         dataset_index = list(range(self.length))
    #     if is_balance:
    #         for i in dataset_index:
    #             if self.labels_classification[i] == 1:
    #                 if max_number_per_sample is not None:
    #                     if id_sum_dict[self.protein_id_list[i]] < max_number_per_sample:
    #                         positive_index.append(i)
    #                         id_sum_dict[self.protein_id_list[i]] += 1
    #                 else:
    #                     positive_index.append(i)
    #             else:
    #                 if max_number_per_sample is not None:
    #                     if id_sum_dict[self.protein_id_list[i]] < max_number_per_sample:
    #                         negative_index.append(i)
    #                         id_sum_dict[self.protein_id_list[i]] += 1
    #                 else:
    #                     negative_index.append(i)
    #         if len(positive_index) > len(negative_index):
    #             resample_num = len(negative_index)
    #             # positive_index=random.sample(positive_index,len(negative_index))
    #         else:
    #             resample_num = len(positive_index)
    #             # negative_index=random.sample(negative_index,len(positive_index))
    #         if max_resample_num is not None:
    #             resample_num_p = int(max_resample_num * ratio_balance_train[1]) if max_resample_num * ratio_balance_train[1] < len(
    #                 positive_index) else resample_num
    #             resample_num_n = max_resample_num - resample_num_p
    #         else:
    #             resample_num_p = resample_num
    #             resample_num_n = resample_num
    #
    #         sub_positive_index = random.sample(positive_index, resample_num_p)
    #         sub_negative_index = random.sample(negative_index, resample_num_n)
    #         # if len(positive_index)>len(negative_index):
    #         #     positive_index=random.sample(positive_index,len(negative_index))
    #         # else:
    #         #     negative_index=random.sample(negative_index,len(positive_index))
    #         dataset_index = sub_positive_index + sub_negative_index
    #         if max_resample_num_val is not None:
    #             if len(valset_index) > max_resample_num_val:
    #                 valset_index = random.sample(valset_index, max_resample_num_val)
    #     elif max_resample_num is not None:
    #         if len(dataset_index) > max_resample_num:
    #             dataset_index = random.sample(dataset_index, max_resample_num)
    #     return dataset_index, valset_index, positive_index, negative_index, (resample_num_p, resample_num_n)

    # def preprocess_trainset_valset_index_finetune(self, valset_name=[], dataset_except_names=[],
    #                                               positive_ratio=0.5,
    #                                               max_number_per_sample=None, is_valset=False):
    #     if is_valset:
    #         id_index_dict_pos = {id: [] for name, id in self.protein_id_dict.items() if name.lower().endswith(
    #             'good') and name in valset_name and name not in dataset_except_names}
    #         id_index_dict_neg = {id: [] for name, id in self.protein_id_dict.items() if name.lower().endswith(
    #             'bad') and name in valset_name and name not in dataset_except_names}
    #     else:
    #         id_index_dict_pos = {id: [] for name, id in self.protein_id_dict.items() if name.lower().endswith(
    #             'good') and name not in valset_name and name not in dataset_except_names}
    #         id_index_dict_neg = {id: [] for name, id in self.protein_id_dict.items() if name.lower().endswith(
    #             'bad') and name not in valset_name and name not in dataset_except_names}
    #     protein_id_list_np = np.array(self.protein_id_list)
    #     for name, id in self.protein_id_dict.items():
    #         if name.lower().endswith('good'):
    #             id_index_dict_pos[id] = np.where(protein_id_list_np == id)[0].tolist()
    #         elif name.lower().endswith('bad'):
    #             id_index_dict_neg[id] = np.where(protein_id_list_np == id)[0].tolist()
    #     resample_num_p = int(max_number_per_sample * 4 * positive_ratio * len(id_index_dict_neg) / (
    #                 len(id_index_dict_pos) + len(id_index_dict_neg)))
    #     resample_num_n = int(max_number_per_sample * 2 - resample_num_p)
    #     return id_index_dict_pos, id_index_dict_neg, (resample_num_p, resample_num_n)

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
        for name, id in self.protein_id_dict.items():
            item_pos = {}
            item_neg = {}
            item_mid = {}
            # if name not in dataset_except_names:
            if name.lower().endswith('good'):
                # if name in valset_name and is_valset:
                #     id_index_dict_pos[id] = np.where(protein_id_list_np == id)[0].tolist()
                # elif name not in valset_name and not is_valset:
                #     id_index_dict_pos[id] = np.where(protein_id_list_np == id)[0].tolist()
                #     if data_error_dis_dict is not None:
                #         data_error_dis_dict_pos[id] = data_error_dis_dict[id] / np.sum(data_error_dis_dict[id])
                item_pos['id'] = np.where(protein_id_list_np == id)[0].tolist()
                item_pos['score'] = [1.0] * len(item_pos['id'])
            elif name.lower().endswith('bad'):
                # if name in valset_name and is_valset:
                #     id_index_dict_neg[id] = np.where(protein_id_list_np == id)[0].tolist()
                # elif name not in valset_name and not is_valset:
                #     id_index_dict_neg[id] = np.where(protein_id_list_np == id)[0].tolist()
                #     if data_error_dis_dict is not None:
                #         data_error_dis_dict_neg[id] = data_error_dis_dict[id] / np.sum(data_error_dis_dict[id])
                item_neg['id'] = np.where(protein_id_list_np == id)[0].tolist()
                item_neg['score'] = [0.0] * len(item_neg['id'])
            else:
                protein_index = np.where(protein_id_list_np == id)[0]
                pos_index = protein_index[labels_classification_np[protein_index] >= middle_range_balance_train[1]]
                neg_index = protein_index[labels_classification_np[protein_index] < middle_range_balance_train[0]]
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
                if len(neg_index) > 0:
                    item_neg['id'] = neg_index.tolist()
                    item_neg['score'] = labels_classification_np[neg_index].tolist()
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
                                           score_bar=None,is_filtering=True):
        if id_map_for_filtering is not None:
            self.pose_id_map2 = id_map_for_filtering

        if score_bar is not None and self.pose_id_map2 is not None and self.labels_classification is not None:
            filtered_id_all=[key for key, value in self.pose_id_map2.items() if self.labels_classification[key] > score_bar]
            self.pose_id_map2 = {key: i for i, key in enumerate(filtered_id_all)}
            # self.pose_id_map2 = {
            #     key: value
            #     for key, value in id_map_for_filtering.items()
            #     if self.labels_classification[key] > score_bar
            # }

        self.labels_class=[self.protein_id_list[i] for i in self.pose_id_map2.keys()] if (self.pose_id_map2 is not None and is_filtering) else self.protein_id_list
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
        scores_np = np.array(self.labels_classification)
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
        self.id_index_dict = id_index_dict
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
                 pretrain_128=False
                 ):
        self.pose_indices = AnnoyIndex(2, 'euclidean')
        self.tif_len = metadata.length
        self.lmdb_path = metadata.lmdb_path
        self.pretrain_128 = pretrain_128

        self.protein_id_list = metadata.protein_id_list
        self.protein_id_dict = metadata.protein_id_dict
        self.protein_id_dict_reverse = {v: k for k, v in self.protein_id_dict.items()}


        if self.lmdb_path is not None:
            lmdb_dir_name_list=list(self.protein_id_dict.keys())
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
            self.env_raw= {}
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
        self.tif_path_list = metadata.all_processed_tif_path
        self.tif_path_list_ctf_correction = metadata.all_processed_tif_path_ctf_correction
        self.tif_path_list_raw = metadata.all_tif_path
        self.tif_path_list_raw_ctf_correction = metadata.all_tif_path_ctf_correction
        self.labels_for_clustering = metadata.labels_for_clustering
        self.labels_classification = metadata.labels_classification
        self.id_index_dict = metadata.id_index_dict

        self.slice_setting = slice_setting
        self.mix_pos_setting = mix_pos_setting
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
        self.accelerator = accelerator
        # self.train=True

        # self.labels_for_training = mrcdata.labels_for_training
        # self.probabilities_for_sampling = mrcdata.probabilities_for_sampling
        self.processed_data_path = metadata.processed_data_path

        if local_crops is not None:
            self.local_crops_number = local_crops['number']
        else:
            self.local_crops_number = 0

        self.weight_for_contrastive_classification_label = weight_for_contrastive_classification_label
        if weight_for_contrastive_classification_label > 0:
            labels_classification_np = np.array(self.labels_classification)
            self.positive_items = np.where(labels_classification_np == 1)[0]
            self.negative_items = np.where(labels_classification_np == 0)[0]
        if slice_setting is not None and (slice_setting['p'] > 0 or slice_setting['align_p'] > 0) and slice_setting[
            'processed_path_slice'] is not None:
            with open(slice_setting['processed_path_slice'] + '/output_processed_tif_path.data', 'rb') as filehandle:
                self.tif_path_list_slice = pickle.load(filehandle)
        else:
            self.tif_path_list_slice = None

        self.pose_id_map = metadata.pose_id_map



    def __len__(self):
        return self.tif_len
        # return len(self.tif_path_list)

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
        local_crops1 = []
        local_crops2 = []
        # end_git_item=time.time()

        '''get mrcdata1 and aug1'''
        mrcdata = self.get_mrcdata(item=item)

        weight = float(1.0)
        # label_for_training=self.labels_for_training[item]
        # gaussian_probabilities=self.probabilities_for_sampling[item]

        '''get labels data'''
        if self.labels_for_clustering is not None and len(self.labels_for_clustering) > item:
            label_for_clustering = self.labels_for_clustering[item]
        else:
            label_for_clustering = -1
        label_for_classification = self.labels_classification[item]
        if self.use_triplex_labels:
            if label_for_classification > self.bar_score:
                label_for_classification = 1.0
            elif label_for_classification <= 1 - self.bar_score:
                label_for_classification = 0.0
            else:
                label_for_classification = 0.5

        # particles_id = self.particles_id[item]
        protein_id = self.protein_id_list[item]

        '''get mrcdata2'''
        if self.needs_aug2:
            mrcdata2 = None
            is_random_rotate_transform = True if self.random_rotate_transform is not None else False
            item2, weight, is_mix_pos = self.get_item2(item)
            if item2 is not None:
                mrcdata2 = self.get_mrcdata(item=item2)
            elif self.slice_setting is not None and random.random() < self.slice_setting['p']:
                mrcdata2 = self.get_corr_slice(item)
            if mrcdata2 is None:
                mrcdata2 = mrcdata
            # else:
            #     is_random_rotate_transform = False
            aug1 = self.mrcdata_aug(mrcdata, is_random_rotate_transform=is_random_rotate_transform,
                                    is_mix_pos=is_mix_pos)
            aug2 = self.mrcdata_aug(mrcdata2, is_random_rotate_transform=is_random_rotate_transform,
                                    is_mix_pos=is_mix_pos)
        else:
            aug1 = self.mrcdata_aug(mrcdata)
            aug2 = None

        # img2tensor = transforms.ToTensor()
        # mrcdata = img2tensor(mrcdata)

        out = {
            # 'mrcdata': mrcdata,
            'aug1': aug1, 'aug2': aug2 if aug2 is not None else [],
            'weight': weight,
            'label_for_clustering': label_for_clustering,
            'label_for_classification': label_for_classification,
            # 'path': tif_path,
            # 'raw_path': raw_tif_path,
            # 'particles_id': str(particles_id),
            'item': item,
            'local_crops1': local_crops1,
            'local_crops2': local_crops2, 'protein_id': protein_id}
        return out

    def get_item2(self, item):
        item2 = None
        weight = 1
        is_mix_pos = False
        if self.weight_for_contrastive_classification_label > 0:
            if random.random() < self.weight_for_contrastive_classification_label:
                if self.labels_classification[item] == 1:
                    item2 = np.random.choice(self.positive_items)
                else:
                    item2 = np.random.choice(self.negative_items)
        elif self.mix_pos_setting is not None and self.mix_pos_setting['p'] > 0 and self.pose_id_map is not None:
            if random.random() < self.mix_pos_setting['p']:
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
                    #     print('item is less than min(item_list): ' + str(item) + ' ' + str(min(item_list)))
                    #     print('protein_name: ' + protein_name)
                    # if item2-min(item_list)<0:
                    #     print('item2 is less than min(item_list): ' + str(item2) + ' ' + str(min(item_list)))
                    #     print('protein_name: ' + protein_name)
                    weight = sigmoid(self.mix_pos_setting['sigma'] * (
                            (3.5 - self.pose_indices.get_distance(item1_pose_id, item2_pose_id)) / 3.5
                            - self.mix_pos_setting['bias']))
                    self.pose_indices.unload()
                    is_mix_pos = True

                # if protein_name=='11307_J504_good':
                #     pass
        return item2, float(weight), is_mix_pos

    def get_nearest_item(self, item, N=None, pose_divide=None):

        if N is None:
            N = self.mix_pos_setting['pose_search_N']
        if pose_divide is None:
            pose_divide = self.mix_pos_setting['pose_search_divide']

        protein_id = self.protein_id_list[item]
        item_list = self.id_index_dict.get(protein_id, [])
        min_id=self.cumulative_sizes[protein_id-1] if protein_id > 0 else 0

        # min_id = min(item_list)
        nearest = None
        protein_name = self.protein_id_dict_reverse[protein_id]
        pose_items_id = []
        item1_pose_id = None
        if item - min_id in self.pose_id_map[protein_id]:
            item1_pose_id = self.pose_id_map[protein_id][item - min_id]
            # if len(item_list) > 1:
            #     item2 = random.choice(item_list)

            self.pose_indices.load(os.path.join(self.processed_data_path, 'pose_data', protein_name + '_pose.ann'))
            nearest = self.pose_indices.get_nns_by_item(item1_pose_id,
                                                        int(len(item_list) / pose_divide) if N is None else N,
                                                        include_distances=False)
            pose_items_id = list(self.pose_id_map[protein_id].keys())
            # nearest=nearest[1:]
        return nearest, min_id, protein_name, pose_items_id, item1_pose_id

    def get_corr_slice(self, item):
        tif_path = self.tif_path_list[item]
        tif_path_split = tif_path.split('/')
        tif_path_split[-5] += '_slice'
        tif_path2 = '/'.join(tif_path_split)
        mrcdata2 = None
        if os.path.exists(tif_path2):
            try:
                with open(tif_path2,
                          'rb') as filehandle:
                    mrcdata2 = pickle.load(filehandle)

            except EOFError:
                print('error for path: ' + tif_path)
        return mrcdata2

    def get_mrcdata(self, item=None, tif_path=None):
        mrcdata = None
        if tif_path is not None:
            if os.path.exists(tif_path):
                try:
                    with open(tif_path,
                              'rb') as filehandle:
                        mrcdata = pickle.load(filehandle)

                except EOFError:
                    print('error for path: ' + tif_path)
        elif item is not None:
            if self.lmdb_path is not None:
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
                lmdb_env_r, lmdb_env_p, _ = self._get_env(lmdb_path,use_raw=self.pretrain_128)
                if self.pretrain_128:
                    lmdb_env = lmdb_env_r
                else:
                    lmdb_env = lmdb_env_p
                with lmdb_env.begin(write=False) as txn:
                    key = f"{local_idx}".encode()
                    value = txn.get(key)
                    # data = torch.load(BytesIO(value),weights_only=False)
                    data = pickle.loads(value)
                    mrcdata = data
                    tif_path = ''
                    # raw_tif_path = ''
                del data, value
            else:
                tif_path = self.tif_path_list[item]
                if self.slice_setting is not None and (random.random() < self.slice_setting['p']):
                    slice_path = random.choice(self.tif_path_list_slice)
                    if os.path.exists(slice_path):
                        tif_path = slice_path

                try:
                    with open(tif_path,
                              'rb') as filehandle:
                        mrcdata = pickle.load(filehandle)

                except EOFError:
                    print('error for path: ' + tif_path)
        return mrcdata

    def mrcdata_aug(self, mrcdata, is_random_rotate_transform=True, is_mix_pos=False):
        if isinstance(mrcdata,np.ndarray) :
            mrcdata =Image.fromarray(mrcdata)
        # # if mrcdata.mode != 'L':
        #     mrcdata = to_int8(mrcdata)
        if is_random_rotate_transform:
            if is_mix_pos and self.random_rotate_transform_mix_pos is not None:
                mrcdata_rotate1 = self.random_rotate_transform_mix_pos(mrcdata)
            elif self.random_rotate_transform is not None:
                mrcdata_rotate1 = self.random_rotate_transform(mrcdata)
            else:
                mrcdata_rotate1 = mrcdata
        else:
            mrcdata_rotate1 = mrcdata
        if is_mix_pos:
            aug = self.transform_mix_pos(mrcdata_rotate1)
        else:
            aug = self.transform(mrcdata_rotate1)
        return aug

    def get_transforms(self, transforms, transforms_list_mix_pos=None):
        self.transform = transforms[0]
        self.local_crops_transform = transforms[1]
        self.random_rotate_transform = transforms[2]
        if transforms_list_mix_pos is not None:
            # self.mix_pos_transforms = transforms_list_mix_pos
            self.random_rotate_transform_mix_pos = transforms_list_mix_pos[2]
            self.transform_mix_pos = transforms_list_mix_pos[0]
        else:
            self.mix_pos_transforms = None
            self.random_rotate_transform_mix_pos = None
            self.transform_mix_pos = None

    def get_local_crops(self, aug1, aug2, mrcdata_rotate1, mrcdata_rotate2):
        local_crops1 = []
        local_crops2 = []
        for _ in range(self.local_crops_number):
            local_crops1.append(self.local_crops_transform(mrcdata_rotate1))
            if self.needs_aug2:
                local_crops2.append(self.local_crops_transform(mrcdata_rotate2))
        # imgs_all = [aug1, aug2] + local_crops1 + local_crops2

        if self.in_chans != 1:
            aug1 = aug1.repeat(self.in_chans, 1, 1)
            local_crops1 = [local_crop.repeat(self.in_chans, 1, 1) for local_crop in local_crops1]
            if self.needs_aug2:
                aug2 = aug2.repeat(self.in_chans, 1, 1)
                local_crops2 = [local_crop.repeat(self.in_chans, 1, 1) for local_crop in local_crops2]
        return local_crops1, local_crops2

    def _get_env(self, lmdb_path, use_raw=False, use_processed=True, use_FT=False):
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

        # 检查缓存中是否已有此LMDB的环境
        # if lmdb_path not in self.open_envs:
        #     # 如果没有，就打开它并存入缓存
        #     # readonly=True, lock=False 对于多进程读取是安全且高效的
        #     env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        #     self.open_envs[lmdb_path] = env
        if os.path.join(lmdb_path, 'lmdb_processed') not in self.env_processed:
            # 如果没有，就打开它并存入缓存
            # readonly=True, lock=False 对于多进程读取是安全且高效的
            if use_processed:
                env_processed = lmdb.open(os.path.join(lmdb_path, 'lmdb_processed'), readonly=True, lock=False,
                                          readahead=False, meminit=False)
                self.env_processed[lmdb_path] = env_processed

            if use_raw:
                env_raw = lmdb.open(os.path.join(lmdb_path, 'lmdb_raw'), readonly=True, lock=False, readahead=False,
                                    meminit=False)
                self.env_raw[lmdb_path] = env_raw

            if use_FT:
                env_FT = lmdb.open(os.path.join(lmdb_path, 'lmdb_FT'), readonly=True, lock=False, readahead=False,
                                   meminit=False)
                self.env_FT[lmdb_path] = env_FT

        # return self.open_envs[lmdb_path]
        return self.env_raw[lmdb_path] if use_raw else None, self.env_processed[lmdb_path], self.env_FT[
            lmdb_path] if use_FT else None


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
