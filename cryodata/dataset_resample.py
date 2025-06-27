from torch.utils.data.sampler import Sampler
import random
import numpy as np


class MyResampleSampler(Sampler):
    def __init__(self, data, id_index_dict_pos, id_index_dict_mid, id_index_dict_neg, resample_num_pos,
                 resample_num_neg, resample_num_mid,
                 batch_size_all=None, shuffle_type=None, dataset_id_map=None, only_mixup_bad_particles=False,
                 error_balance=None, mean_error_dis_dict=None, data_error_dis_dict=None,
                 balance_per_interval=False,
                 index_list=[0.3]
                 ):
        self.data = data
        self.id_index_dict_pos = id_index_dict_pos
        self.id_index_dict_mid = id_index_dict_mid
        self.id_index_dict_neg = id_index_dict_neg
        self.resample_num_pos = resample_num_pos
        self.resample_num_mid = resample_num_mid
        self.resample_num_neg = resample_num_neg
        self.shuffle_type = shuffle_type
        self.batch_size_all = batch_size_all
        self.dataset_id_map = dataset_id_map
        self.my_seed = 0
        self.only_mixup_bad_particles = only_mixup_bad_particles
        self.error_balance = error_balance
        self.mean_error_dis_dict = mean_error_dis_dict
        self.data_error_dis_dict = data_error_dis_dict
        self.balance_per_interval = balance_per_interval
        self.index_list = index_list
        self.indices = resample_from_id_index_dict_finetune(self.id_index_dict_pos,
                                                            self.id_index_dict_mid,
                                                            self.id_index_dict_neg,
                                                            self.resample_num_pos,
                                                            self.resample_num_mid,
                                                            self.resample_num_neg, batch_size_all=self.batch_size_all,
                                                            shuffle_type=self.shuffle_type, my_seed=self.my_seed,
                                                            dataset_id_map=dataset_id_map,
                                                            only_mixup_bad_particles=only_mixup_bad_particles,
                                                            balance_per_interval=balance_per_interval,
                                                            index_list=index_list
                                                            # , error_balance=error_balance,
                                                            # mean_error_dis_dict=mean_error_dis_dict
                                                            # , data_error_dis_dict=data_error_dis_dict
                                                            )

    def __iter__(self):
        self.indices = resample_from_id_index_dict_finetune(self.id_index_dict_pos,
                                                            self.id_index_dict_mid,
                                                            self.id_index_dict_neg,
                                                            self.resample_num_pos,
                                                            self.resample_num_mid,
                                                            self.resample_num_neg, batch_size_all=self.batch_size_all,
                                                            shuffle_type=self.shuffle_type, my_seed=self.my_seed,
                                                            dataset_id_map=self.dataset_id_map,
                                                            only_mixup_bad_particles=self.only_mixup_bad_particles,
                                                            balance_per_interval=self.balance_per_interval,
                                                            index_list=self.index_list,
                                                            # , error_balance=self.error_balance,
                                                            # mean_error_dis_dict=self.mean_error_dis_dict
                                                            # , data_error_dis_dict=self.data_error_dis_dict
                                                            )
        self.my_seed += 1
        return iter(self.indices)

    def __len__(self):
        # return len(self.data)
        return len(self.indices)


class MyResampleSampler_pretrain(Sampler):
    def __init__(self, id_index_dict, batch_size_all, max_number_per_sample=None, shuffle_type=None,
                 shuffle_mix_up_ratio=0.2, dataset_id_map=None, bad_particles_ratio=0.1, combine_same_class=False,
                 only_mixup_bad_particles=False, id_scores_dict=None, scores_bar=0.8):
        self.id_index_dict = id_index_dict
        self.batch_size_all = batch_size_all
        self.max_number_per_sample = max_number_per_sample
        self.shuffle_type = shuffle_type
        self.shuffle_mix_up_ratio = shuffle_mix_up_ratio
        self.my_seed = 0
        self.dataset_id_map = dataset_id_map
        self.bad_particles_ratio = bad_particles_ratio
        self.combine_same_class = combine_same_class
        self.only_mixup_bad_particles = only_mixup_bad_particles
        self.id_scores_dict = id_scores_dict
        self.scores_bar = scores_bar
        self.indices = resample_from_id_index_dict(id_index_dict, max_number_per_sample, batch_size_all, shuffle_type,
                                                   shuffle_mix_up_ratio, self.my_seed, dataset_id_map,
                                                   bad_particles_ratio=bad_particles_ratio,
                                                   combine_same_class=combine_same_class,
                                                   only_mixup_bad_particles=only_mixup_bad_particles,
                                                   id_scores_dict=id_scores_dict,
                                                   scores_bar=scores_bar
                                                   )

    # @profile(precision=4)
    def __iter__(self):
        self.indices = resample_from_id_index_dict(self.id_index_dict, self.max_number_per_sample, self.batch_size_all,
                                                   self.shuffle_type, self.shuffle_mix_up_ratio, self.my_seed,
                                                   self.dataset_id_map, bad_particles_ratio=self.bad_particles_ratio,
                                                   combine_same_class=self.combine_same_class,
                                                   only_mixup_bad_particles=self.only_mixup_bad_particles,
                                                   id_scores_dict=self.id_scores_dict,
                                                   scores_bar=self.scores_bar
                                                   )
        # print(self.indices[0:80])
        self.my_seed += 1
        # print(sorted(self.indices))
        return iter(self.indices)

    def __len__(self):
        # return len(self.data)
        return len(self.indices)


class MyResampleSampler_pretrain_old(Sampler):
    def __init__(self, id_index_dict, max_number_per_sample, bad_particles_id=[]):
        self.id_index_dict = id_index_dict
        self.max_number_per_sample = max_number_per_sample
        self.indices = resample_from_id_index_dict_old(id_index_dict, max_number_per_sample, bad_particles_id)
        self.bad_particles_id = bad_particles_id

    def __iter__(self):
        self.indices = resample_from_id_index_dict_old(self.id_index_dict, self.max_number_per_sample,
                                                       self.bad_particles_id)

        return iter(self.indices)

    def __len__(self):
        # return len(self.data)
        return len(self.indices)


# # Optimize the resample_from_id_index_dict function
def resample_from_id_index_dict(id_index_dict, max_number_per_sample=None, batch_size_all=None, shuffle_type=None,
                                shuffle_mix_up_ratio=0.2, my_seed=0, dataset_id_map=None, bad_particles_ratio=0.1,
                                combine_same_class=False, only_mixup_bad_particles=False, error_balance=None,
                                mean_error_dis_dict=None, data_error_dis_dict=None,
                                id_scores_dict=None, scores_bar=0.0,
                                balance_per_interval=False,
                                interval_list=[0.3]
                                ):
    random.seed(my_seed)
    resampled_index_list = []
    final_resampled_index_list = []
    ids_list = list(id_index_dict.keys())
    mix_up_list = []
    ids_list_all = list(id_index_dict.keys())

    if dataset_id_map is not None:
        id_map = dataset_id_map['id_map']
        bad_id_list = dataset_id_map['bad_id_list']
        bad_id_list_all = dataset_id_map['bad_id_list_all']
        if id_map is not None:
            ids_list = list(id_map.keys())
        else:
            ids_list = None


    else:
        id_map = None
        bad_id_list = []
        bad_id_list_all = []

    # if shuffle_type == 'class':
    #     random.shuffle(ids_list)
    if shuffle_type == 'class':
        if ids_list is not None:
            random.shuffle(ids_list)
        random.shuffle(ids_list_all)

    if id_map is not None and combine_same_class:
        for my_id in ids_list:
            if id_map[my_id] is not None:
                max_number_per_sample_neg = int(max_number_per_sample * bad_particles_ratio)
                max_number_per_sample_pos = int(max_number_per_sample - max_number_per_sample_neg)

                selected_index_list1, mix_up_list_added1 = get_index_per_class(id_index_dict[my_id],
                                                                               max_number_per_sample_pos, shuffle_type,
                                                                               shuffle_mix_up_ratio,
                                                                               is_bad_class=my_id in bad_id_list,
                                                                               error_distribution=data_error_dis_dict[
                                                                                   my_id] if data_error_dis_dict is not None else None,
                                                                               interval_list=interval_list)

                selected_index_list2, mix_up_list_added2 = get_index_per_class(id_index_dict[id_map[my_id]],
                                                                               max_number_per_sample_neg, shuffle_type,
                                                                               shuffle_mix_up_ratio,
                                                                               error_distribution=data_error_dis_dict[
                                                                                   id_map[
                                                                                       my_id]] if data_error_dis_dict is not None else None,
                                                                               interval_list=interval_list)
                new_selected_index_list = selected_index_list1 + selected_index_list2
                # random.shuffle(new_selected_index_list)
                # resampled_index_list.append(new_selected_index_list)
                mix_up_list_added = mix_up_list_added1 + mix_up_list_added2
                # mix_up_list.extend(mix_up_list_added)
            else:
                if my_id in bad_id_list and not only_mixup_bad_particles:
                    max_number_per_sample_i = max_number_per_sample * bad_particles_ratio
                else:
                    max_number_per_sample_i = max_number_per_sample

                new_selected_index_list, mix_up_list_added = get_index_per_class(id_index_dict[my_id],
                                                                                 max_number_per_sample_i, shuffle_type,
                                                                                 shuffle_mix_up_ratio,
                                                                                 is_bad_class=my_id in bad_id_list
                                                                                 ,
                                                                                 error_distribution=data_error_dis_dict[
                                                                                     my_id] if data_error_dis_dict is not None else None,
                                                                                 interval_list=interval_list)
            if only_mixup_bad_particles:
                if my_id in bad_id_list:
                    new_selected_index_list = []
                else:
                    mix_up_list_added = []

            if len(new_selected_index_list) > 0:
                resampled_index_list.append(new_selected_index_list)
            mix_up_list.extend(mix_up_list_added)
    else:
        for my_id in ids_list_all:

            if my_id in bad_id_list_all and not only_mixup_bad_particles:
                max_number_per_sample_i = int(max_number_per_sample * bad_particles_ratio)
            else:
                max_number_per_sample_i = max_number_per_sample

            if error_balance is not None:
                if error_balance['data_mean_error']:
                    max_number_per_sample_i = int(max_number_per_sample_i * mean_error_dis_dict[my_id])

            new_selected_index_list, mix_up_list_added = get_index_per_class(id_index_dict[my_id],
                                                                             max_number_per_sample_i, shuffle_type,
                                                                             shuffle_mix_up_ratio,
                                                                             is_bad_class=my_id in bad_id_list_all,
                                                                             error_distribution=data_error_dis_dict[
                                                                                 my_id] if data_error_dis_dict is not None else None,
                                                                             # dataset_scores=np.array(id_scores_dict)[id_index_dict[my_id]] if id_scores_dict is not None else None,
                                                                             dataset_scores=id_scores_dict[
                                                                                 my_id] if id_scores_dict is not None else None,
                                                                             # bad_particles_ratio=bad_particles_ratio
                                                                             scores_bar=scores_bar,
                                                                             balance_per_interval=balance_per_interval,
                                                                             interval_list=interval_list
                                                                             )
            if only_mixup_bad_particles:
                if my_id in bad_id_list_all:
                    new_selected_index_list = []
                else:
                    mix_up_list_added = []

            if len(new_selected_index_list) > 0:
                resampled_index_list.append(new_selected_index_list)
            mix_up_list.extend(mix_up_list_added)

    if shuffle_type == 'batch':
        random.shuffle(mix_up_list)
        step = len(mix_up_list) // len(resampled_index_list)
        for i in range(len(resampled_index_list)):
            # step=batch_size_all-len(resampled_index_list[i])
            resampled_index_list[i].extend(mix_up_list[:step])
            random.shuffle(resampled_index_list[i])
            new_resampled_index_list_i = []
            for ii in range(len(resampled_index_list[i]) // batch_size_all + 1):
                if len(resampled_index_list[i]) >= batch_size_all:
                    new_resampled_index_list_i.append(resampled_index_list[i][:batch_size_all])
                    resampled_index_list[i] = resampled_index_list[i][batch_size_all:]
            final_resampled_index_list.extend(new_resampled_index_list_i)
            mix_up_list = mix_up_list[step:]

    if shuffle_type == 'class':
        random.shuffle(mix_up_list)
        step = len(mix_up_list) // len(resampled_index_list)
        for i in range(len(resampled_index_list)):
            # step=batch_size_all-len(resampled_index_list[i])
            resampled_index_list[i].extend(mix_up_list[:step])
            random.shuffle(resampled_index_list[i])
            mix_up_list = mix_up_list[step:]

    if shuffle_type == 'batch':
        random.shuffle(final_resampled_index_list)
        final_resampled_index_list = [item for sublist in final_resampled_index_list for item in sublist]
    else:
        final_resampled_index_list = [item for sublist in resampled_index_list for item in sublist]
        if shuffle_type == 'all':
            random.shuffle(final_resampled_index_list)
    return final_resampled_index_list


def resample_from_id_index_dict_old(id_index_dict, max_number_per_sample, bad_particles_id=[]):
    resampled_index_list = []
    for id, index_list in id_index_dict.items():
        if id in bad_particles_id:
            new_max_number_per_sample = max_number_per_sample // 10
            # new_max_number_per_sample=0
        else:
            new_max_number_per_sample = max_number_per_sample
        if len(index_list) > new_max_number_per_sample:
            resampled_index_list.extend(random.sample(index_list, new_max_number_per_sample))
        else:
            resampled_index_list.extend(index_list)
    return resampled_index_list


def get_index_per_class(index_list, max_number_per_sample=None, shuffle_type=None, shuffle_mix_up_ratio=0.2,
                        is_bad_class=False, error_distribution=None, dataset_scores=None, scores_bar=0.8,
                        balance_per_interval=False, interval_list=[0.3]):
    # index_list = id_index_dict[my_id]
    if balance_per_interval:
        index_list = balance_from_scores_interval(20, index_list['score'], index_list['id'], num_min_per_interval=128,
                                                  interval_list=interval_list)
    else:
        if isinstance(index_list, dict):
            index_list = index_list['id']

    if dataset_scores is not None and scores_bar > 0 and max(dataset_scores) != min(dataset_scores):
        index_list_good = np.array(index_list)[np.array(dataset_scores) >= scores_bar].tolist()
        index_list_bad = np.array(index_list)[np.array(dataset_scores) < scores_bar].tolist()
        index_list = index_list_good

    else:
        index_list_bad = None
    len_index_list = len(index_list)
    if max_number_per_sample is not None and len_index_list > max_number_per_sample:
        if error_distribution is not None:
            index_array = np.array(index_list)
            selected_indices = np.random.choice(index_array, size=max_number_per_sample, replace=False,
                                                p=error_distribution)
            selected_index_list = selected_indices.tolist()
        else:
            selected_index_list = random.sample(index_list, int(max_number_per_sample))

        if shuffle_type == 'batch' or shuffle_type == 'class':
            if is_bad_class:
                mix_up_list_added = selected_index_list[:int(len(selected_index_list) * shuffle_mix_up_ratio)]
                new_selected_index_list = []
            elif index_list_bad is not None:
                if len(index_list_bad) < int(len(selected_index_list) * shuffle_mix_up_ratio):
                    mix_up_list_added = index_list_bad
                else:
                    mix_up_list_added = random.sample(index_list_bad,
                                                      int(len(selected_index_list) * shuffle_mix_up_ratio))
                new_selected_index_list = selected_index_list[
                                          :max_number_per_sample - int(len(selected_index_list) * shuffle_mix_up_ratio)]
            else:
                new_selected_index_list = selected_index_list[
                                          :max_number_per_sample - int(len(selected_index_list) * shuffle_mix_up_ratio)]
                mix_up_list_added = (
                    selected_index_list[max_number_per_sample - int(len(selected_index_list) * shuffle_mix_up_ratio):])
        else:
            if is_bad_class and shuffle_mix_up_ratio > 0:
                mix_up_list_added = selected_index_list
                new_selected_index_list = []
            elif index_list_bad is not None and shuffle_mix_up_ratio > 0:
                if len(index_list_bad) < int(len(index_list_bad) * shuffle_mix_up_ratio):
                    mix_up_list_added = index_list_bad
                else:

                    mix_up_list_added = random.sample(index_list_bad, int(len(index_list_bad) * shuffle_mix_up_ratio))
                new_selected_index_list = selected_index_list[
                                          :max_number_per_sample - int(len(index_list_bad) * shuffle_mix_up_ratio)]
            else:
                new_selected_index_list = selected_index_list
                mix_up_list_added = []
    else:
        if is_bad_class:
            mix_up_list_added = index_list[:int(len(index_list) * shuffle_mix_up_ratio)]
            new_selected_index_list = []
        elif shuffle_type == 'batch' or shuffle_type == 'class':
            random.shuffle(index_list)
            new_selected_index_list = index_list[:len_index_list - int(len(index_list) * shuffle_mix_up_ratio)]
            mix_up_list_added = (index_list[len_index_list - int(len(index_list) * shuffle_mix_up_ratio):])
        else:
            if shuffle_type != 'valset':
                random.shuffle(index_list)
            new_selected_index_list = index_list
            mix_up_list_added = []
    return new_selected_index_list, mix_up_list_added


def resample_from_id_index_dict_finetune(id_index_dict_pos, id_index_dict_mid, id_index_dict_neg, resample_num_p,
                                         resample_num_mid, resample_num_n,
                                         batch_size_all=None, shuffle_type=None, shuffle_mix_up_ratio=0.2, my_seed=0,
                                         dataset_id_map=None, only_mixup_bad_particles=False,
                                         balance_per_interval=False,
                                         index_list=[0.3],
                                         # error_balance=None,
                                         # mean_error_dis_dict=None,
                                         # data_error_dis_dict=None
                                         ):
    random.seed(my_seed)
    if shuffle_type == 'batch':
        id_index_dict_all = {**id_index_dict_pos, **id_index_dict_neg}
        resampled_index_list = resample_from_id_index_dict(id_index_dict_all, resample_num_p + resample_num_n,
                                                           batch_size_all, shuffle_type,
                                                           shuffle_mix_up_ratio, my_seed, dataset_id_map=dataset_id_map,
                                                           only_mixup_bad_particles=only_mixup_bad_particles,
                                                           balance_per_interval=balance_per_interval,
                                                           interval_list=index_list
                                                           # error_balance=error_balance,
                                                           # mean_error_dis_dict=mean_error_dis_dict,
                                                           # data_error_dis_dict=data_error_dis_dict
                                                           # positive_ratio=resample_num_p /(resample_num_p + resample_num_n)
                                                           )

    else:
        resampled_index_list = []
        resampled_index_list.extend(
            resample_from_id_index_dict(id_index_dict_pos, resample_num_p, batch_size_all, shuffle_type,
                                        shuffle_mix_up_ratio, my_seed,
                                        only_mixup_bad_particles=only_mixup_bad_particles,
                                        balance_per_interval=balance_per_interval,
                                        interval_list=index_list
                                        # error_balance=error_balance,
                                        # mean_error_dis_dict=mean_error_dis_dict,
                                        # data_error_dis_dict=data_error_dis_dict['good']
                                        ))
        resampled_index_list.extend(
            resample_from_id_index_dict(id_index_dict_neg, resample_num_n, batch_size_all, shuffle_type,
                                        shuffle_mix_up_ratio, my_seed,
                                        only_mixup_bad_particles=only_mixup_bad_particles,
                                        balance_per_interval=balance_per_interval,
                                        interval_list=index_list
                                        # error_balance=error_balance,
                                        # mean_error_dis_dict=mean_error_dis_dict,
                                        # data_error_dis_dict=data_error_dis_dict['bad']
                                        ))
        if len(id_index_dict_mid) > 0:
            resampled_index_list.extend(
                resample_from_id_index_dict(id_index_dict_mid, resample_num_mid, batch_size_all, shuffle_type,
                                            shuffle_mix_up_ratio, my_seed,
                                            only_mixup_bad_particles=only_mixup_bad_particles,
                                            balance_per_interval=balance_per_interval,
                                            interval_list=index_list
                                            # error_balance=error_balance, mean_error_dis_dict=mean_error_dis_dict,
                                            # data_error_dis_dict=data_error_dis_dict['mid']
                                            ))
        if shuffle_type == 'all':
            random.shuffle(resampled_index_list)
    return resampled_index_list


def balance_from_scores_interval(interval_num, scores, ids, num_min_per_interval=0, interval_list=[0.3]):
    """
    Balance the dataset based on scores intervals.
    :param interval_num: Number of intervals to divide the scores into.
    :param scores: List of scores corresponding to each id.
    :param ids: List of ids corresponding to each score.
    :param num_min_per_interval: Minimum number of samples per interval.
    :return: Balanced ids and scores.
    """

    if len(scores) == 0:
        return [], []

    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return ids, scores  # No variation in scores, return original lists
    if len(interval_list)>1:
        interval_num = len(interval_list)
    interval_size = (max_score - min_score) / interval_num
    min_interval_len = len(scores)
    scores_np = np.array(scores)
    ids_np = np.array(ids)

    intervals = [[] for _ in range(interval_num)]
    for i in range(interval_num):
        lower_bound = min_score + i * interval_size
        upper_bound = min_score + (i + 1) * interval_size
        mask = (scores_np >= lower_bound) & (scores_np < upper_bound)
        interval_ids = ids_np[mask].tolist()
        intervals[i] = interval_ids
        if len(interval_ids) < min_interval_len:
            min_interval_len = len(interval_ids)

    if min_interval_len > num_min_per_interval:
        num_min_per_interval = min_interval_len
    balanced_ids = []
    # balanced_scores = []
    if interval_list is not None and len(interval_list) > 0:
        if len(interval_list) == 1:
            new_intervals = [1 - interval_list[0] + interval_list[0] / interval_num * i for i in range(interval_num)]
        else:
            new_intervals = interval_list
    else:
        new_intervals = [1 for _ in range(interval_num)]
    for i, interval in enumerate(intervals):
        if len(interval) < int(num_min_per_interval * new_intervals[i]):
            selected_ids = interval
        else:
            selected_ids = random.sample(interval, int(num_min_per_interval * new_intervals[i]))
        balanced_ids.extend(selected_ids)

    balanced_ids = sorted(balanced_ids)  # Sort the final list of ids

    return balanced_ids
