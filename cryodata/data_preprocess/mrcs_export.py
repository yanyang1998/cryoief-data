from pathlib import Path

import numpy as np
import torch
from cryosparc.dataset import Dataset

from cryodata.cs_star_translate.cs2star import cs2star
from cryodata.cs_star_translate.pyem.mrc import ZSliceWriter


class CryoMRCSSaver:
    """Online writer for dataloader-produced particles and aligned metadata."""

    def __init__(
        self,
        save_path,
        particles_per_mrcs_file,
        reference_cs_path=None,
        orig_min=-5.0,
        orig_max=5.0,
        dataset_transform=None,
        normalize_mean=None,
        normalize_std=None,
        output_prefix='generated_particles',
    ):
        if particles_per_mrcs_file <= 0:
            raise ValueError('particles_per_mrcs_file must be a positive integer.')
        if orig_max <= orig_min:
            raise ValueError('orig_max must be greater than orig_min.')

        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.particles_per_mrcs_file = int(particles_per_mrcs_file)
        self.reference_cs_path = reference_cs_path
        self.reference_cs = Dataset.load(reference_cs_path) if reference_cs_path is not None else None
        self.orig_min = float(orig_min)
        self.orig_max = float(orig_max)
        self.output_prefix = output_prefix
        self.normalize_ops = self._resolve_normalize_ops(
            dataset_transform=dataset_transform,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )

        self._writer = None
        self._current_file_index = -1
        self._current_file_count = 0
        self._total_count = 0
        self._closed = False
        self._reference_indices = []
        self._blob_paths = []
        self._blob_indices = []
        self._blob_shapes = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @staticmethod
    def _resolve_normalize_ops(dataset_transform=None, normalize_mean=None, normalize_std=None):
        if normalize_mean is not None or normalize_std is not None:
            if normalize_mean is None or normalize_std is None:
                raise ValueError('normalize_mean and normalize_std must be provided together.')
            return [(normalize_mean, normalize_std)]
        return CryoMRCSSaver._find_normalize_ops(dataset_transform)

    @staticmethod
    def _find_normalize_ops(transform):
        if transform is None:
            return []

        ops = []
        if hasattr(transform, 'transforms'):
            for child in transform.transforms:
                ops.extend(CryoMRCSSaver._find_normalize_ops(child))
            return ops

        if isinstance(transform, (list, tuple)):
            for child in transform:
                ops.extend(CryoMRCSSaver._find_normalize_ops(child))
            return ops

        if transform.__class__.__name__ == 'Normalize' and hasattr(transform, 'mean') and hasattr(transform, 'std'):
            ops.append((transform.mean, transform.std))
        return ops

    @staticmethod
    def _single_channel_value(value, name):
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must contain exactly one value for single-channel particle export.')
        return float(arr[0])

    def _prepare_images(self, images):
        if torch.is_tensor(images):
            images_np = images.detach().cpu().float().numpy()
        else:
            images_np = np.asarray(images, dtype=np.float32)

        if images_np.ndim == 2:
            images_np = images_np[np.newaxis, :, :]
        elif images_np.ndim == 3:
            pass
        elif images_np.ndim == 4:
            if images_np.shape[1] != 1:
                raise ValueError('CryoMRCSSaver only supports single-channel image batches.')
            images_np = images_np[:, 0, :, :]
        else:
            raise ValueError('images must have shape (H, W), (B, H, W), or (B, 1, H, W).')

        images_np = images_np.astype(np.float32, copy=False)
        for mean, std in reversed(self.normalize_ops):
            mean_value = self._single_channel_value(mean, 'Normalize mean')
            std_value = self._single_channel_value(std, 'Normalize std')
            images_np = images_np * std_value + mean_value

        images_np = np.clip(images_np, 0.0, 1.0)
        images_np = images_np * (self.orig_max - self.orig_min) + self.orig_min
        return images_np.astype(np.float32, copy=False)

    @staticmethod
    def _prepare_item_indices(item_indices):
        if torch.is_tensor(item_indices):
            return item_indices.detach().cpu().numpy().astype(np.int64).reshape(-1).tolist()
        return np.asarray(item_indices, dtype=np.int64).reshape(-1).tolist()

    def _output_filename(self, file_index):
        return f'{self.output_prefix}_{file_index:06d}.mrcs'

    def _scaled_psize(self, reference_index, new_box_size):
        if self.reference_cs is None or 'blob/psize_A' not in self.reference_cs or 'blob/shape' not in self.reference_cs:
            return 1.0
        old_shape = np.asarray(self.reference_cs['blob/shape'][reference_index], dtype=np.float32).reshape(-1)
        if old_shape.size == 0 or old_shape[0] <= 0:
            return float(self.reference_cs['blob/psize_A'][reference_index])
        return float(self.reference_cs['blob/psize_A'][reference_index]) * float(old_shape[0]) / float(new_box_size)

    def _open_next_writer(self, image_shape, reference_index=None):
        if self._writer is not None:
            self._writer.close()
        self._current_file_index += 1
        self._current_file_count = 0
        filename = self._output_filename(self._current_file_index)
        psize = self._scaled_psize(reference_index, image_shape[0]) if reference_index is not None else 1.0
        self._writer = ZSliceWriter(str(self.save_path / filename), shape=image_shape, dtype=np.float32, psz=psize)

    def write_batch(self, images, item_indices):
        if self._closed:
            raise RuntimeError('Cannot write to a closed CryoMRCSSaver.')

        particles = self._prepare_images(images)
        indices = self._prepare_item_indices(item_indices)
        if len(indices) != particles.shape[0]:
            raise ValueError('item_indices length must match batch size.')

        if self.reference_cs is not None:
            max_index = len(self.reference_cs) - 1
            invalid = [idx for idx in indices if idx < 0 or idx > max_index]
            if invalid:
                raise IndexError(f'item_indices contain values outside reference .cs range: {invalid[:5]}')

        for particle, reference_index in zip(particles, indices):
            if self._writer is None or self._current_file_count >= self.particles_per_mrcs_file:
                self._open_next_writer(particle.shape, reference_index=reference_index)
            elif particle.shape != tuple(self._writer.shape):
                raise ValueError('All exported particles in one saver must have the same image shape.')

            self._writer.write(particle)

            filename = self._output_filename(self._current_file_index)
            self._reference_indices.append(reference_index)
            self._blob_paths.append(filename)
            self._blob_indices.append(self._current_file_count)
            self._blob_shapes.append([particle.shape[0], particle.shape[1]])
            self._current_file_count += 1
            self._total_count += 1

    def close(self):
        if self._closed:
            return
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self.reference_cs is not None and self._total_count > 0:
            self._write_metadata()
        self._closed = True

    def _write_metadata(self):
        exported_cs = self.reference_cs.take(np.asarray(self._reference_indices, dtype=np.int64))
        self._ensure_blob_fields(exported_cs)

        exported_cs['blob/path'] = np.asarray(self._blob_paths, dtype=object)
        exported_cs['blob/idx'] = np.asarray(self._blob_indices, dtype=exported_cs['blob/idx'].dtype)
        exported_cs['blob/shape'] = np.asarray(self._blob_shapes, dtype=exported_cs['blob/shape'].dtype)
        self._update_pixel_size_fields(exported_cs)
        self._update_shift_fields(exported_cs)

        cs_path = self.save_path / f'{self.output_prefix}.cs'
        star_path = self.save_path / f'{self.output_prefix}.star'
        exported_cs.save(cs_path)
        result = cs2star(str(cs_path), str(star_path))
        if result == 1:
            raise RuntimeError(f'Failed to convert {cs_path} to STAR metadata.')

    @staticmethod
    def _ensure_blob_fields(cs_data):
        fields_to_add = []
        dtypes_to_add = []
        if 'blob/path' not in cs_data:
            fields_to_add.append('blob/path')
            dtypes_to_add.append('O')
        if 'blob/idx' not in cs_data:
            fields_to_add.append('blob/idx')
            dtypes_to_add.append('uint32')
        if 'blob/shape' not in cs_data:
            fields_to_add.append('blob/shape')
            dtypes_to_add.append(('uint32', (2,)))
        if 'blob/psize_A' not in cs_data:
            fields_to_add.append('blob/psize_A')
            dtypes_to_add.append('float32')
        if fields_to_add:
            cs_data.add_fields(fields_to_add, dtypes_to_add)

    def _scale_factors(self):
        ref_shapes = np.asarray(self.reference_cs['blob/shape'][self._reference_indices], dtype=np.float32)
        new_shapes = np.asarray(self._blob_shapes, dtype=np.float32)
        old_box = ref_shapes[:, 0]
        new_box = new_shapes[:, 0]
        scale = np.ones_like(new_box, dtype=np.float32)
        valid = old_box > 0
        scale[valid] = new_box[valid] / old_box[valid]
        return scale

    def _update_pixel_size_fields(self, cs_data):
        if 'blob/psize_A' not in self.reference_cs or 'blob/shape' not in self.reference_cs:
            return

        scale = self._scale_factors()
        old_psize = np.asarray(self.reference_cs['blob/psize_A'][self._reference_indices], dtype=np.float32)
        new_psize = old_psize / scale
        cs_data['blob/psize_A'] = new_psize.astype(cs_data['blob/psize_A'].dtype)
        for field in ('alignments2D/psize_A', 'alignments3D/psize_A'):
            if field in cs_data:
                cs_data[field] = new_psize.astype(cs_data[field].dtype)

    def _update_shift_fields(self, cs_data):
        if 'blob/shape' not in self.reference_cs:
            return
        scale = self._scale_factors()
        for field in ('alignments2D/shift', 'alignments3D/shift'):
            if field in cs_data:
                shifts = np.asarray(cs_data[field], dtype=np.float32) * scale[:, np.newaxis]
                cs_data[field] = shifts.astype(cs_data[field].dtype)
