from torch.utils.data.sampler import Sampler
import random
import numpy as np

# Batch size divisors for micrograph datasets (mics_*).
# A smaller number for __init__ gives a larger starting batch size,
# while the tighter value in __iter__ keeps per-step memory usage lower.
MICS_BATCH_DIVISOR_INIT = 10
MICS_BATCH_DIVISOR_ITER = 8


def _validate_ratio(name, value):
    if value is None:
        return None
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f'{name} must be within [0.0, 1.0].')
    return value


def _validate_calculated_score_ratio(calculated_score_ratio):
    return _validate_ratio('calculated_score_ratio', calculated_score_ratio)


def _validate_missing_score_ratio(missing_score_ratio):
    return _validate_ratio('missing_score_ratio', missing_score_ratio)


def _validate_score_source_ratio_inputs(calculated_score_ratio=None, missing_score_ratio=None):
    calculated_score_ratio = _validate_calculated_score_ratio(calculated_score_ratio)
    missing_score_ratio = _validate_missing_score_ratio(missing_score_ratio)
    if (
        calculated_score_ratio is not None
        and missing_score_ratio is not None
        and calculated_score_ratio + missing_score_ratio > 1.0
    ):
        raise ValueError(
            'calculated_score_ratio and missing_score_ratio must sum to at most 1.0.'
        )
    return calculated_score_ratio, missing_score_ratio


def _coerce_optional_sequence(values, expected_length):
    if values is None:
        return None
    values_list = list(values)
    if len(values_list) != expected_length:
        return None
    return values_list


def _align_values_to_indices(selected_ids, source_ids, source_values):
    aligned_values = _coerce_optional_sequence(source_values, len(source_ids))
    if aligned_values is None:
        return None
    value_by_id = {source_id: aligned_values[idx] for idx, source_id in enumerate(source_ids)}
    try:
        return [value_by_id[item_id] for item_id in selected_ids]
    except KeyError:
        return None


def _derive_score_source_from_legacy(used_default_score, expected_length):
    legacy_values = _coerce_optional_sequence(used_default_score, expected_length)
    if legacy_values is None:
        return None
    return [0 if int(flag) == 0 else 1 for flag in legacy_values]


def _sample_without_replacement(items, sample_size, weights=None):
    items = list(items)
    sample_size = min(sample_size, len(items))
    if sample_size <= 0:
        return []
    if sample_size >= len(items):
        selected_items = list(items)
        random.shuffle(selected_items)
        return selected_items

    probability_weights = _coerce_optional_sequence(weights, len(items))
    if probability_weights is not None:
        weights_np = np.asarray(probability_weights, dtype=float)
        if np.all(np.isfinite(weights_np)) and np.any(weights_np > 0):
            weights_np = np.clip(weights_np, a_min=0.0, a_max=None)
            weight_sum = float(weights_np.sum())
            if weight_sum > 0:
                probabilities = weights_np / weight_sum
                selected_positions = np.random.choice(
                    len(items), size=sample_size, replace=False, p=probabilities
                ).tolist()
                return [items[position] for position in selected_positions]

    return random.sample(items, sample_size)


def _sample_ids_with_metadata(index_list, sample_size, used_default_score=None, score_source=None, weights=None):
    selected_ids = _sample_without_replacement(index_list, sample_size, weights=weights)
    selected_used_default_score = _align_values_to_indices(selected_ids, index_list, used_default_score)
    selected_score_source = _align_values_to_indices(selected_ids, index_list, score_source)
    selected_weights = _align_values_to_indices(selected_ids, index_list, weights)
    return selected_ids, selected_used_default_score, selected_score_source, selected_weights


def _shuffle_metadata(index_list, used_default_score=None, score_source=None, weights=None):
    if len(index_list) <= 1:
        return list(index_list), used_default_score, score_source, weights
    shuffled_order = list(range(len(index_list)))
    random.shuffle(shuffled_order)
    shuffled_index_list = [index_list[idx] for idx in shuffled_order]
    shuffled_used_default_score = (
        [used_default_score[idx] for idx in shuffled_order]
        if used_default_score is not None else None
    )
    shuffled_score_source = (
        [score_source[idx] for idx in shuffled_order]
        if score_source is not None else None
    )
    shuffled_weights = [weights[idx] for idx in shuffled_order] if weights is not None else None
    return shuffled_index_list, shuffled_used_default_score, shuffled_score_source, shuffled_weights


def _select_indices_with_score_source_ratios(index_list, sample_size, score_source,
                                             calculated_score_ratio=None, missing_score_ratio=None,
                                             weights=None):
    weights = _coerce_optional_sequence(weights, len(index_list))
    score_source = _coerce_optional_sequence(score_source, len(index_list))
    calculated_score_ratio, missing_score_ratio = _validate_score_source_ratio_inputs(
        calculated_score_ratio,
        missing_score_ratio,
    )
    if score_source is None or (calculated_score_ratio is None and missing_score_ratio is None):
        return _sample_without_replacement(index_list, sample_size, weights=weights)

    sample_size = min(sample_size, len(index_list))
    if sample_size <= 0:
        return []

    selected_positions = []
    selected_position_set = set()

    def _take_positions(target_positions, target_count):
        if target_count <= 0 or len(target_positions) == 0:
            return []
        position_weights = [weights[idx] for idx in target_positions] if weights is not None else None
        return _sample_without_replacement(target_positions, target_count, weights=position_weights)

    if calculated_score_ratio is not None:
        calculated_positions = [idx for idx, source in enumerate(score_source) if int(source) == 0]
        selected_calculated = _take_positions(
            calculated_positions,
            min(int(round(sample_size * calculated_score_ratio)), len(calculated_positions)),
        )
        selected_positions.extend(selected_calculated)
        selected_position_set.update(selected_calculated)

    if missing_score_ratio is not None:
        missing_positions = [idx for idx, source in enumerate(score_source) if int(source) == 2]
        selected_missing = _take_positions(
            [idx for idx in missing_positions if idx not in selected_position_set],
            min(int(round(sample_size * missing_score_ratio)), len(missing_positions)),
        )
        selected_positions.extend(selected_missing)
        selected_position_set.update(selected_missing)

    remaining_slots = sample_size - len(selected_positions)
    if remaining_slots > 0:
        remaining_positions = [idx for idx in range(len(index_list)) if idx not in selected_position_set]
        remaining_weights = [weights[idx] for idx in remaining_positions] if weights is not None else None
        selected_positions.extend(
            _sample_without_replacement(remaining_positions, remaining_slots, weights=remaining_weights)
        )

    random.shuffle(selected_positions)
    return [index_list[idx] for idx in selected_positions]


class MyResampleSampler(Sampler):
    def __init__(self, data, id_index_dict_pos, id_index_dict_mid, id_index_dict_neg, resample_num_pos,
                 resample_num_neg, resample_num_mid,
                 batch_size_all=None, shuffle_type=None, dataset_id_map=None, only_mixup_bad_particles=False,
                 error_balance=None, mean_error_dis_dict=None, data_error_dis_dict=None,
                 balance_per_interval=False,
                 index_list=None,
                 shuffle_mix_up_ratio=0.0,
                 calculated_score_ratio=None,
                 missing_score_ratio=None,
                 ):
        self.data = data
        if index_list is None:
            index_list = [0.3]
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
        self.shuffle_mix_up_ratio = shuffle_mix_up_ratio
        (
            self.calculated_score_ratio,
            self.missing_score_ratio,
        ) = _validate_score_source_ratio_inputs(calculated_score_ratio, missing_score_ratio)
        self.indices = resample_from_id_index_dict_finetune(self.id_index_dict_pos,
                                                            self.id_index_dict_mid,
                                                            self.id_index_dict_neg,
                                                            self.resample_num_pos,
                                                            self.resample_num_mid,
                                                            self.resample_num_neg,
                                                            shuffle_mix_up_ratio=shuffle_mix_up_ratio,
                                                            batch_size_all=self.batch_size_all,
                                                            shuffle_type=self.shuffle_type, my_seed=self.my_seed,
                                                            dataset_id_map=dataset_id_map,
                                                            only_mixup_bad_particles=only_mixup_bad_particles,
                                                            balance_per_interval=balance_per_interval,
                                                            index_list=index_list,
                                                            calculated_score_ratio=self.calculated_score_ratio,
                                                            missing_score_ratio=self.missing_score_ratio
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
                                                            calculated_score_ratio=self.calculated_score_ratio,
                                                            missing_score_ratio=self.missing_score_ratio,
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
                 only_mixup_bad_particles=False, id_scores_dict=None, id_used_default_score_dict=None,
                 id_score_source_dict=None, scores_bar=0.8, id_protein_name_dict=None, num_processes=1,
                 calculated_score_ratio=None, missing_score_ratio=None):
        # if isinstance(shuffle_type, int):
        #     shuffle_type=int(shuffle_type/2)

        if shuffle_type == 'batch':
            shuffle_type = 1
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
        self.id_used_default_score_dict = id_used_default_score_dict
        self.id_score_source_dict = id_score_source_dict
        self.scores_bar = scores_bar
        self.id_protein_name_dict=id_protein_name_dict
        (
            self.calculated_score_ratio,
            self.missing_score_ratio,
        ) = _validate_score_source_ratio_inputs(calculated_score_ratio, missing_score_ratio)
        self.batch_num=None
        self.num_processes=num_processes

        indices = resample_from_id_index_dict(id_index_dict, max_number_per_sample, int(batch_size_all * 1.2),
                                              int(shuffle_type / 2) if isinstance(shuffle_type, int) else shuffle_type,
                                              shuffle_mix_up_ratio, self.my_seed, dataset_id_map,
                                              bad_particles_ratio=bad_particles_ratio,
                                              combine_same_class=combine_same_class,
                                              only_mixup_bad_particles=only_mixup_bad_particles,
                                              id_scores_dict=id_scores_dict,
                                              id_used_default_score_dict=id_used_default_score_dict,
                                              id_score_source_dict=id_score_source_dict,
                                              scores_bar=scores_bar,
                                              id_protein_name_dict=self.id_protein_name_dict,
                                              calculated_score_ratio=self.calculated_score_ratio,
                                              missing_score_ratio=self.missing_score_ratio
                                              )
        if isinstance(self.shuffle_type, int):
            combined_resampled_index_list, indices = self._build_batched_index_list(
                indices, MICS_BATCH_DIVISOR_INIT)
        self.indices = indices

    # @profile(precision=4)
    def _build_batched_index_list(self, indices, mics_divisor):
        """Combine per-class index lists into shuffled batches of size batch_size_all / num_processes."""
        combined = []
        for i in range(len(indices)):
            batch_size_i = int(self.batch_size_all / (self.num_processes * self.shuffle_type))
            batch_i = []
            if (self.id_protein_name_dict is not None
                    and i in self.id_protein_name_dict
                    and self.id_protein_name_dict[i].startswith('mics_')):
                batch_size_i = int(batch_size_i / mics_divisor)
            if len(indices[i]) < batch_size_i * self.num_processes and len(indices[i]) > self.num_processes:
                batch_size_ii = int(len(indices[i]) / self.num_processes)
                for j in range(0, len(indices[i]), batch_size_ii):
                    batch = indices[i][j:j + batch_size_ii]
                    batch_i.append(batch)
                    if len(batch_i) == self.num_processes and isinstance(batch_i[0], list):
                        combined.append(indices[i])
                        batch_i = []
            else:
                for j in range(0, len(indices[i]), batch_size_i):
                    batch = indices[i][j:j + batch_size_i]
                    if len(batch) == batch_size_i:
                        batch_i.append(batch)
                    if len(batch_i) == self.num_processes and isinstance(batch_i[0], list):
                        combined.append(batch_i)
                        batch_i = []
        random.shuffle(combined)
        combined = [item for sublist in combined
                    for item in sublist
                    if isinstance(sublist, list) and isinstance(item, list) and len(item) > 0]
        self.batch_num = len(combined)
        flat_indices = [idx for batch in combined
                        if isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], int)
                        for idx in batch]
        return combined, flat_indices

    def __iter__(self):
        indices = resample_from_id_index_dict(self.id_index_dict, self.max_number_per_sample,
                                              int(1.2 * self.batch_size_all),
                                              int(self.shuffle_type / 2) if isinstance(self.shuffle_type,
                                                                                       int) else self.shuffle_type,
                                              self.shuffle_mix_up_ratio, self.my_seed,
                                              self.dataset_id_map, bad_particles_ratio=self.bad_particles_ratio,
                                              combine_same_class=self.combine_same_class,
                                              only_mixup_bad_particles=self.only_mixup_bad_particles,
                                              id_scores_dict=self.id_scores_dict,
                                              id_used_default_score_dict=self.id_used_default_score_dict,
                                              id_score_source_dict=self.id_score_source_dict,
                                              scores_bar=self.scores_bar,
                                              id_protein_name_dict=self.id_protein_name_dict,
                                              calculated_score_ratio=self.calculated_score_ratio,
                                              missing_score_ratio=self.missing_score_ratio
                                              )
        self.indices = indices
        self.my_seed += 1
        if isinstance(self.shuffle_type, int):
            combined_resampled_index_list, indices = self._build_batched_index_list(
                indices, MICS_BATCH_DIVISOR_ITER)
            self.indices = indices

            for batch in combined_resampled_index_list:
                yield batch

        else:
            for idx in self.indices:
                yield idx

    def __len__(self):
        # return len(self.data)
        if self.batch_num is not None:
            return self.batch_num
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
                                id_scores_dict=None, id_used_default_score_dict=None, id_score_source_dict=None,
                                scores_bar=0.0,
                                balance_per_interval=False,
                                interval_list=None,
                                per_batch_num=0,
                                id_protein_name_dict=None,
                                calculated_score_ratio=None,
                                missing_score_ratio=None
                                ):
    random.seed(my_seed)
    np.random.seed(my_seed)
    calculated_score_ratio, missing_score_ratio = _validate_score_source_ratio_inputs(
        calculated_score_ratio,
        missing_score_ratio,
    )
    if interval_list is None:
        interval_list = [0.3]
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
    if isinstance(shuffle_type, int):
        per_batch_num = shuffle_type
        shuffle_type = 'batch'

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
                                                                               dataset_used_default_score=id_used_default_score_dict[
                                                                                   my_id] if id_used_default_score_dict is not None else None,
                                                                               dataset_score_source=id_score_source_dict[
                                                                                   my_id] if id_score_source_dict is not None else None,
                                                                               interval_list=interval_list,
                                                                               calculated_score_ratio=calculated_score_ratio,
                                                                               missing_score_ratio=missing_score_ratio)

                selected_index_list2, mix_up_list_added2 = get_index_per_class(id_index_dict[id_map[my_id]],
                                                                               max_number_per_sample_neg, shuffle_type,
                                                                               shuffle_mix_up_ratio,
                                                                               error_distribution=data_error_dis_dict[
                                                                                   id_map[
                                                                                       my_id]] if data_error_dis_dict is not None else None,
                                                                               dataset_used_default_score=id_used_default_score_dict[
                                                                                   id_map[my_id]] if id_used_default_score_dict is not None else None,
                                                                               dataset_score_source=id_score_source_dict[
                                                                                   id_map[my_id]] if id_score_source_dict is not None else None,
                                                                               interval_list=interval_list,
                                                                               calculated_score_ratio=calculated_score_ratio,
                                                                               missing_score_ratio=missing_score_ratio)
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
                                                                                 dataset_used_default_score=id_used_default_score_dict[
                                                                                     my_id] if id_used_default_score_dict is not None else None,
                                                                                 dataset_score_source=id_score_source_dict[
                                                                                     my_id] if id_score_source_dict is not None else None,
                                                                                 interval_list=interval_list,
                                                                                 calculated_score_ratio=calculated_score_ratio,
                                                                                 missing_score_ratio=missing_score_ratio)
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
                                                                             dataset_used_default_score=id_used_default_score_dict[
                                                                                 my_id] if id_used_default_score_dict is not None else None,
                                                                             dataset_score_source=id_score_source_dict[
                                                                                 my_id] if id_score_source_dict is not None else None,
                                                                             # bad_particles_ratio=bad_particles_ratio
                                                                             scores_bar=scores_bar,
                                                                             balance_per_interval=balance_per_interval,
                                                                             interval_list=interval_list,
                                                                             bad_particles_ratio=bad_particles_ratio,
                                                                             calculated_score_ratio=calculated_score_ratio,
                                                                             missing_score_ratio=missing_score_ratio,

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
        batch_size_all_new = batch_size_all // (per_batch_num * 2) if per_batch_num > 0 else batch_size_all
        # step = len(mix_up_list)*batch_size_all_new // (len(resampled_index_list)*max_number_per_sample)
        step = len(mix_up_list) // len(resampled_index_list)

        for i in range(len(resampled_index_list)):
            # step=batch_size_all-len(resampled_index_list[i])
            if len(mix_up_list) >= step:
                resampled_index_list[i].extend(mix_up_list[:step])
            random.shuffle(resampled_index_list[i])
            # new_resampled_index_list_i = []
            # # for ii in range(len(resampled_index_list[i]) // batch_size_all + 1):
            # for ii in range(max_number_per_sample // batch_size_all_new + 1):
            #     if len(resampled_index_list[i]) >= batch_size_all_new:
            #         new_resampled_index_list_i.append(resampled_index_list[i][:batch_size_all_new])
            #         resampled_index_list[i] = resampled_index_list[i][batch_size_all_new:]
            #     else:
            #         if per_batch_num > 0:
            #             new_resampled_index_list_i.append(resampled_index_list[i])
            #         elif id_protein_name_dict is not None and id_protein_name_dict[i].startswith('mics_'):
            #             new_resampled_index_list_i.append(resampled_index_list[i]+random.choices(resampled_index_list[i],k=batch_size_all_new-len(resampled_index_list[i])))
            #         resampled_index_list[i] = []
            #
            # final_resampled_index_list.extend(new_resampled_index_list_i)
            if len(mix_up_list) >= step:
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
        # return final_resampled_index_list
        return resampled_index_list
        # if per_batch_num > 0:
        #     return final_resampled_index_list
        # random.shuffle(final_resampled_index_list)
        # result = [item for sublist in final_resampled_index_list for item in sublist]
    else:
        result = [item for sublist in resampled_index_list for item in sublist]
        if shuffle_type == 'all':
            random.shuffle(result)
    return result


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
                        bad_particles_ratio=0.2,
                        is_bad_class=False, error_distribution=None, dataset_scores=None, scores_bar=0.8,
                        balance_per_interval=False, interval_list=None, dataset_used_default_score=None,
                        dataset_score_source=None, calculated_score_ratio=None, missing_score_ratio=None):
    if interval_list is None:
        interval_list = [0.3]
    calculated_score_ratio, missing_score_ratio = _validate_score_source_ratio_inputs(
        calculated_score_ratio,
        missing_score_ratio,
    )

    if isinstance(index_list, dict):
        base_index_list = list(index_list.get('id', []))
        if dataset_scores is None:
            dataset_scores = index_list.get('score')
        if dataset_used_default_score is None:
            dataset_used_default_score = index_list.get('used_default_score')
        if dataset_score_source is None:
            dataset_score_source = index_list.get('score_source')
    else:
        base_index_list = list(index_list)

    dataset_scores = _coerce_optional_sequence(dataset_scores, len(base_index_list))
    dataset_used_default_score = _coerce_optional_sequence(dataset_used_default_score, len(base_index_list))
    dataset_score_source = _coerce_optional_sequence(dataset_score_source, len(base_index_list))
    if dataset_score_source is None:
        dataset_score_source = _derive_score_source_from_legacy(dataset_used_default_score, len(base_index_list))
    error_distribution = _coerce_optional_sequence(error_distribution, len(base_index_list))

    if balance_per_interval and dataset_scores is not None:
        index_list = balance_from_scores_interval(20, dataset_scores, base_index_list, num_min_per_interval=128,
                                                  interval_list=interval_list)
        dataset_scores = _align_values_to_indices(index_list, base_index_list, dataset_scores)
        dataset_used_default_score = _align_values_to_indices(index_list, base_index_list, dataset_used_default_score)
        dataset_score_source = _align_values_to_indices(index_list, base_index_list, dataset_score_source)
        error_distribution = _align_values_to_indices(index_list, base_index_list, error_distribution)
    else:
        index_list = base_index_list

    if dataset_scores is not None and len(index_list) > 0 and scores_bar > 0 and max(dataset_scores) != min(dataset_scores):
        scores_arr = np.asarray(dataset_scores)
        index_arr = np.asarray(index_list)
        good_mask = scores_arr >= scores_bar
        bad_mask = (scores_arr < scores_bar) & (scores_arr >= 0)

        index_list_good = index_arr[good_mask].tolist()
        index_list_bad = index_arr[bad_mask].tolist()
        used_default_good = None
        used_default_bad = None
        score_source_good = None
        score_source_bad = None
        error_distribution_good = None
        error_distribution_bad = None
        if dataset_used_default_score is not None:
            used_default_arr = np.asarray(dataset_used_default_score)
            used_default_good = used_default_arr[good_mask].tolist()
            used_default_bad = used_default_arr[bad_mask].tolist()
        if dataset_score_source is not None:
            score_source_arr = np.asarray(dataset_score_source)
            score_source_good = score_source_arr[good_mask].tolist()
            score_source_bad = score_source_arr[bad_mask].tolist()
        if error_distribution is not None:
            error_distribution_arr = np.asarray(error_distribution)
            error_distribution_good = error_distribution_arr[good_mask].tolist()
            error_distribution_bad = error_distribution_arr[bad_mask].tolist()

        index_list = index_list_good
        dataset_used_default_score = used_default_good
        dataset_score_source = score_source_good
        error_distribution = error_distribution_good
    else:
        index_list_bad = None
        used_default_bad = None
        score_source_bad = None
        error_distribution_bad = None
        if dataset_scores is not None:
            scores_arr = np.asarray(dataset_scores)
            valid_mask = scores_arr >= 0
            index_list = np.asarray(index_list)[valid_mask].tolist()
            if dataset_used_default_score is not None:
                dataset_used_default_score = np.asarray(dataset_used_default_score)[valid_mask].tolist()
            if dataset_score_source is not None:
                dataset_score_source = np.asarray(dataset_score_source)[valid_mask].tolist()
            if error_distribution is not None:
                error_distribution = np.asarray(error_distribution)[valid_mask].tolist()

    if bad_particles_ratio > 0 and index_list_bad is not None and len(index_list_bad) > 0:
        num_good_resample = int(len(index_list) * (1 - bad_particles_ratio))
        num_bad_resample = min(len(index_list_bad), int(len(index_list) * bad_particles_ratio))
        selected_good, selected_good_default, selected_good_score_source, selected_good_weights = _sample_ids_with_metadata(
            index_list, num_good_resample, dataset_used_default_score, dataset_score_source, error_distribution
        )
        selected_bad, selected_bad_default, selected_bad_score_source, selected_bad_weights = _sample_ids_with_metadata(
            index_list_bad, num_bad_resample, used_default_bad, score_source_bad, error_distribution_bad
        )
        index_list = selected_good + selected_bad
        if selected_good_default is not None or selected_bad_default is not None:
            dataset_used_default_score = (selected_good_default or []) + (selected_bad_default or [])
        else:
            dataset_used_default_score = None
        if selected_good_score_source is not None or selected_bad_score_source is not None:
            dataset_score_source = (selected_good_score_source or []) + (selected_bad_score_source or [])
        else:
            dataset_score_source = None
        if selected_good_weights is not None or selected_bad_weights is not None:
            error_distribution = (selected_good_weights or []) + (selected_bad_weights or [])
        else:
            error_distribution = None
        index_list, dataset_used_default_score, dataset_score_source, error_distribution = _shuffle_metadata(
            index_list, dataset_used_default_score, dataset_score_source, error_distribution
        )

    len_index_list = len(index_list)
    if max_number_per_sample is not None and len_index_list > max_number_per_sample:
        if dataset_score_source is not None and (
            calculated_score_ratio is not None or missing_score_ratio is not None
        ):
            selected_index_list = _select_indices_with_score_source_ratios(
                index_list, max_number_per_sample, dataset_score_source,
                calculated_score_ratio=calculated_score_ratio,
                missing_score_ratio=missing_score_ratio,
                weights=error_distribution
            )
        elif error_distribution is not None:
            selected_index_list = _sample_without_replacement(index_list, max_number_per_sample, weights=error_distribution)
        else:
            selected_index_list = random.sample(index_list, int(max_number_per_sample))

        if shuffle_type == 'batch' or shuffle_type == 'class':
            if is_bad_class:
                mix_up_list_added = selected_index_list[:int(len(selected_index_list) * shuffle_mix_up_ratio)]
                new_selected_index_list = []
            else:
                new_selected_index_list = selected_index_list[
                    :max_number_per_sample - int(len(selected_index_list) * shuffle_mix_up_ratio)]
                mix_up_list_added = (
                    selected_index_list[max_number_per_sample - int(len(selected_index_list) * shuffle_mix_up_ratio):])
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
                                         index_list=None,
                                         calculated_score_ratio=None,
                                         missing_score_ratio=None,

                                         # error_balance=None,
                                         # mean_error_dis_dict=None,
                                         # data_error_dis_dict=None
                                         ):
    random.seed(my_seed)
    np.random.seed(my_seed)
    calculated_score_ratio, missing_score_ratio = _validate_score_source_ratio_inputs(
        calculated_score_ratio,
        missing_score_ratio,
    )
    if index_list is None:
        index_list = [0.3]
    if isinstance(shuffle_type, int):
        per_batch_num = shuffle_type
        shuffle_type = 'batch'
    else:
        per_batch_num = 0
    if shuffle_type == 'batch':
        # id_index_dict_all = {**id_index_dict_pos, **id_index_dict_neg}
        resampled_index_list_P = resample_from_id_index_dict(id_index_dict_pos,
                                                             resample_num_p,
                                                             batch_size_all,
                                                             shuffle_type,
                                                             shuffle_mix_up_ratio, my_seed,
                                                             dataset_id_map=dataset_id_map,
                                                             only_mixup_bad_particles=only_mixup_bad_particles,
                                                             balance_per_interval=balance_per_interval,
                                                             interval_list=index_list,
                                                             per_batch_num=per_batch_num,
                                                             calculated_score_ratio=calculated_score_ratio,
                                                             missing_score_ratio=missing_score_ratio
                                                             # error_balance=error_balance,
                                                             # mean_error_dis_dict=mean_error_dis_dict,
                                                             # data_error_dis_dict=data_error_dis_dict
                                                             # positive_ratio=resample_num_p /(resample_num_p + resample_num_n)
                                                             )
        resampled_index_list_N = resample_from_id_index_dict(id_index_dict_neg,
                                                             resample_num_n,
                                                             batch_size_all,
                                                             shuffle_type,
                                                             shuffle_mix_up_ratio, my_seed,
                                                             dataset_id_map=dataset_id_map,
                                                             only_mixup_bad_particles=only_mixup_bad_particles,
                                                             balance_per_interval=balance_per_interval,
                                                             interval_list=index_list,
                                                             per_batch_num=per_batch_num,
                                                             calculated_score_ratio=calculated_score_ratio,
                                                             missing_score_ratio=missing_score_ratio
                                                             )
        if len(id_index_dict_mid) > 0:
            resampled_index_list_M = resample_from_id_index_dict(id_index_dict_mid, resample_num_mid,
                                                                 batch_size_all, shuffle_type,
                                                                 shuffle_mix_up_ratio, my_seed,
                                                                 only_mixup_bad_particles=only_mixup_bad_particles,
                                                                 balance_per_interval=balance_per_interval,
                                                                 interval_list=index_list,
                                                                 per_batch_num=per_batch_num,
                                                                 calculated_score_ratio=calculated_score_ratio,
                                                                 missing_score_ratio=missing_score_ratio
                                                                 )
            combined_resampled_index_list = [
                (resampled_index_list_P[i] if len(resampled_index_list_P) > i else []) + (resampled_index_list_N[i] if len(
                    resampled_index_list_N) > i else []) + (resampled_index_list_M[i] if len(
                    resampled_index_list_M) > i else []) for i in
                range(len(resampled_index_list_P))]
            # combined_resampled_index_list = [
            #     random.sample(combined_resampled_index_list[i], int(batch_size_all / per_batch_num)) if int(
            #         batch_size_all / per_batch_num) < len(combined_resampled_index_list[i]) else [] for i in
            #     range(len(resampled_index_list_P))]
        else:
            # combined_resampled_index_list = [random.sample(resampled_index_list_P[i] + resampled_index_list_N[i],
            #                                                int(batch_size_all / per_batch_num)) for i in
            #                                  range(len(resampled_index_list_P))]

            combined_resampled_index_list = [
                (resampled_index_list_P[i] if len(resampled_index_list_P)>i else []) + (resampled_index_list_N[i] if len(resampled_index_list_N)>i else []) for i in
                range(len(resampled_index_list_P))]
        combined_resampled_index_list = [
            random.sample(combined_resampled_index_list[i], int(batch_size_all / per_batch_num)) if int(
                batch_size_all / per_batch_num) <= len(combined_resampled_index_list[i]) else [] for i in
            range(len(resampled_index_list_P))]
        random.shuffle(combined_resampled_index_list)
        resampled_index_list = [item for sublist in combined_resampled_index_list for item in sublist]

    else:
        resampled_index_list = []
        resampled_index_list.extend(
            resample_from_id_index_dict(id_index_dict_pos, resample_num_p,
                                        batch_size_all,
                                        shuffle_type,
                                        shuffle_mix_up_ratio, my_seed,
                                        only_mixup_bad_particles=only_mixup_bad_particles,
                                        balance_per_interval=balance_per_interval,
                                        interval_list=index_list,
                                        calculated_score_ratio=calculated_score_ratio,
                                        missing_score_ratio=missing_score_ratio
                                        # error_balance=error_balance,
                                        # mean_error_dis_dict=mean_error_dis_dict,
                                        # data_error_dis_dict=data_error_dis_dict['good']
                                        ))
        resampled_index_list.extend(
            resample_from_id_index_dict(id_index_dict_neg, resample_num_n,
                                        batch_size_all,
                                        shuffle_type,
                                        shuffle_mix_up_ratio, my_seed,
                                        only_mixup_bad_particles=only_mixup_bad_particles,
                                        balance_per_interval=balance_per_interval,
                                        interval_list=index_list,
                                        calculated_score_ratio=calculated_score_ratio,
                                        missing_score_ratio=missing_score_ratio
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
                                            interval_list=index_list,
                                            calculated_score_ratio=calculated_score_ratio,
                                            missing_score_ratio=missing_score_ratio
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
    if len(interval_list) > 1:
        interval_num = len(interval_list)
    interval_size = (max_score - min_score) / interval_num
    min_interval_len = len(scores)
    scores_np = np.array(scores)
    ids_np = np.array(ids)

    intervals = [[] for _ in range(interval_num)]
    for i in range(interval_num):
        lower_bound = min_score + i * interval_size
        upper_bound = min_score + (i + 1) * interval_size
        mask = (scores_np >= lower_bound) & (scores_np < upper_bound) & (scores_np >= 0)
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
