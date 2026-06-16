# cryodata

[![PyPI version](https://badge.fury.io/py/cryodata.svg)](https://badge.fury.io/py/cryodata)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
<a href="https://doi.org/10.1038/s41592-025-02916-8"><img src="https://img.shields.io/badge/Paper-Nature%20Methods-blue" style="max-width: 100%;"></a>
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/cryodata)](https://pepy.tech/project/cryodata)

Cryo-EM data processing tools for deep learning. This package provides a full pipeline for converting raw cryo-EM particle data from [cryoSPARC](https://cryosparc.com/) into PyTorch-ready datasets, as used by [cryo-IEF](https://github.com/westlake-repl/Cryo-IEF), [CryoDECO](https://github.com/yanyang1998/CryoDECO) and [CryoWizard](https://github.com/SMART-StructBio-AI/CryoWizard).

## Features


- **Preprocessing pipeline** — resize, normalize, and window-mask cryo-EM particles from cryoSPARC jobs
- **LMDB dataset creation** — fast multi-process conversion of MRC stacks into LMDB databases for efficient training I/O
- **PyTorch dataset & sampler** — `CryoEMDataset` and `CryoMetaData` classes with support for balanced resampling
- **Fourier-space representations** — optional FFT/Hilbert-transform outputs alongside real-space images
- **Online MRCS export** — write dataloader-generated particles back to `.mrcs` stacks with aligned `.cs` and `.star` metadata
- **Format conversion** — convert cryoSPARC `.cs` files to RELION `.star` format

## Installation

```bash
pip install cryodata
```

For development:

```bash
git clone https://github.com/SMART-StructBio-AI/cryoief-data
cd cryoief-data
pip install -e .
```

## Quick Start

```python
from cryodata.data_preprocess.mrc_preprocess import raw_data_preprocess
from cryodata.cryoemDataset import CryoEMDataset, CryoMetaData
import torchvision.transforms as transforms
import torch

PRETRAIN_MEAN_STD = (0.549995056189533, 0.11784453744259035) # mean and std calculated from the training set of Cryo-IEF
raw_data_path = 'path/to/cryosparc/particles/job'
processed_data_path = 'path/to/processed/data'

# Step 1: Preprocess raw cryoSPARC particle data
new_cs_data = raw_data_preprocess(
    raw_data_path,
    processed_data_path,
    resize=224,          # resize particles to 224×224
    is_to_int8=True,     # convert to uint8 for storage efficiency
)

# Step 2: Load the dataset
meta_data = CryoMetaData(processed_data_path=processed_data_path)
cryodataset = CryoEMDataset(metadata=meta_data)

# Step 3: Set up data augmentation transforms
base_transforms ={'ptcls': transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=PRETRAIN_MEAN_STD[0], std=PRETRAIN_MEAN_STD[1])])}
cryodataset.get_transforms(transforms=base_transforms)

# Step 4: Create a DataLoader for training
dataloader = torch.utils.data.DataLoader(cryodataset, batch_size=32, shuffle=True)
```

## API Reference

### Preprocessing

#### `raw_data_preprocess`

```python
from cryodata.data_preprocess.mrc_preprocess import raw_data_preprocess

new_cs_data = raw_data_preprocess(
    raw_dataset_dir,
    dataset_save_dir,
    resize=224,
    is_to_int8=True,
    num_processes=8,
    chunksize=1,
)
```

The main entry point for the preprocessing pipeline. Reads cryoSPARC `.cs` metadata and associated MRC particle stacks from `raw_dataset_dir`, applies the selected transforms, and writes an LMDB dataset to `dataset_save_dir`. Internally it calls `raw_csdata_process_from_cryosparc_dir` to locate and merge the correct `.cs` files. Returns the merged cryoSPARC `Dataset` object.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `raw_dataset_dir` | `str` | — | Path to a cryoSPARC job output directory (e.g., a particle extraction job) |
| `dataset_save_dir` | `str` | — | Directory where processed data and metadata will be saved |
| `resize` | `int` | `224` | Target image size in pixels (square); uses FFT-based downsampling when reducing, bicubic otherwise |
| `is_to_int8` | `bool` | `True` | Normalize each particle to [0, 255] and cast to `uint8` for compact storage |
| `num_processes` | `int` | `8` | Number of worker processes for LMDB conversion |
| `chunksize` | `int` | `0` | Multiprocessing chunk size for LMDB conversion; `0` uses a chunk size of `1` |
| `particle_chunk_size` | `int` or `None` | `None` | Maximum particles loaded from each MRC stack per worker task; `None` chooses an adaptive memory-bounded chunk size |

---

#### `raw_csdata_process_from_cryosparc_dir`

```python
from cryodata.data_preprocess.mrc_preprocess import raw_csdata_process_from_cryosparc_dir

cs_data, mrc_dir = raw_csdata_process_from_cryosparc_dir(
    raw_data_path,
    processed_cs_path=None,
)
```

Scans a cryoSPARC job directory and locates the relevant `.cs` particle file and the corresponding MRC stack directory. Handles various cryoSPARC job types (extraction, import, restack, downsampling). When both a particles `.cs` file and a passthrough file are found, they are merged via an inner join. Returns the `Dataset` object and the path (or list of paths) to the MRC stacks.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `raw_data_path` | `str` | — | Path to a cryoSPARC job directory or a `.cs` file inside one |
| `processed_cs_path` | `str` | `None` | Optional path to an already processed `.cs` file to load instead of rebuilding `new_particles.cs` |

---

#### `mrcs_resize`

```python
from cryodata.data_preprocess.mrc_preprocess import mrcs_resize

resized = mrcs_resize(mrcs, width, height=None, is_freqs=True)
```

Resizes a 2D image or a batch of images. Accepts a NumPy array or a PIL `Image`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mrcs` | `np.ndarray` or `PIL.Image` | — | Single image `(H, W)` or image stack `(N, H, W)` |
| `width` | `int` | — | Target width in pixels |
| `height` | `int` | `None` | Target height; defaults to `width` for square output |
| `is_freqs` | `bool` | `True` | When `True` and both input and output are square downsampling operations, resize in the Fourier domain (FFT crop); otherwise use bicubic spatial interpolation |

---

#### `mrcs_to_int8`

```python
from cryodata.data_preprocess.mrc_preprocess import mrcs_to_int8

uint8_stack = mrcs_to_int8(mrcs)
```

Normalizes each image in a batch to [0, 255] and converts to `uint8`. Accepts both NumPy arrays and PyTorch tensors of shape `(N, H, W)`. Applies `to_int8` to every image in the batch independently.

---

#### `to_int8`

```python
from cryodata.data_preprocess.mrc_preprocess import to_int8

img_uint8 = to_int8(mrcdata)
```

Normalizes a single 2D image to [0, 255] and converts to `uint8`. For NumPy input, returns a PIL `Image` (grayscale). For PyTorch tensor input, returns a `uint8` tensor.

---

#### `window_mask`

```python
from cryodata.data_preprocess.mrc_preprocess import window_mask

mask = window_mask(resolution, in_rad, out_rad=0.99)
```

Generates a 2D radial linear-taper windowing mask of shape `(resolution, resolution)`. The mask is 1.0 inside `in_rad` and tapers linearly to 0.0 at `out_rad`. Useful for suppressing edge artifacts before computing FFTs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resolution` | `int` | — | Image size in pixels; must be even |
| `in_rad` | `float` | — | Inner radius as a fraction of the image half-width (e.g. `0.85` means 85% of the half-width) |
| `out_rad` | `float` | `0.99` | Outer radius where the mask reaches 0 |

---

#### `sample_and_evaluate`

```python
from cryodata.data_preprocess.mrc_preprocess import sample_and_evaluate

mean_len = sample_and_evaluate(
    path_list, save_path,
    num_stacks=50, num_particles=20000,
    window=False, window_r=0.85, needs_FT=False,
    resize=None, is_to_int8=True, return_stats=False,
)
```

Estimates dataset statistics by randomly sampling MRC stacks. Saves `means_stds_raw.data`, `means_stds_FT.data`, and `img_dim.data` to `save_path`. Returns the average number of particles per stack (`mean_imgs_len`), which is used to estimate the LMDB map size.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path_list` | `list[str]` | — | List of MRC file paths to sample from |
| `save_path` | `str` | — | Directory to save the computed statistics |
| `num_stacks` | `int` | `50` | Number of MRC stacks to randomly sample |
| `num_particles` | `int` | `20000` | Total number of particles to sample across all stacks |
| `window` | `bool` | `False` | Apply a radial window mask before computing statistics |
| `window_r` | `float` | `0.85` | Inner radius for the window mask |
| `needs_FT` | `bool` | `False` | Also compute and save Hartley-transform statistics |
| `resize` | `int` | `None` | Optional processed image size used when estimating serialized LMDB particle size |
| `is_to_int8` | `bool` | `True` | Whether the processed-size estimate should model `uint8` storage |
| `return_stats` | `bool` | `False` | Return a statistics dictionary instead of only `mean_imgs_len` |

---

#### `CryoMRCSSaver`

```python
from cryodata.data_preprocess.mrcs_export import CryoMRCSSaver

with CryoMRCSSaver(
    save_path='path/to/exported_particles',
    particles_per_mrcs_file=1000,
    reference_cs_path='path/to/reference/new_particles.cs',
    dataset_transform=cryodataset.transform,
    orig_min=-5.0,
    orig_max=5.0,
) as saver:
    for batch in dataloader:
        saver.write_batch(batch['aug1'], batch['item'])
```

Online exporter for saving dataloader-produced cryo-EM particles back to `.mrcs` stacks. The saver writes particles incrementally and closes each `.mrcs` file as soon as `particles_per_mrcs_file` is reached. When `reference_cs_path` is provided and at least one particle is written, `close()` saves `generated_particles.cs` and `generated_particles.star` in the export root.

When `reference_cs_path` is provided, metadata rows are copied from the reference `.cs` using `batch['item']`, so CTF and alignment metadata stay one-to-one with exported particles even when the dataloader is shuffled. The saver updates `blob/path`, `blob/idx`, `blob/shape`, `blob/psize_A`, and alignment pixel-size/shift fields when present.

Input images should be single-channel tensors with shape `(B, 1, H, W)` or `(B, H, W)`. If the dataset transform contains `transforms.Normalize`, the saver reverses it before converting `[0, 1]` values back to approximate MRC-space values using `orig_min` and `orig_max`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_path` | `str` or `Path` | — | Root directory for `.mrcs`, `.cs`, and `.star` outputs |
| `particles_per_mrcs_file` | `int` | — | Maximum number of particles per output `.mrcs` stack |
| `reference_cs_path` | `str` or `Path` | `None` | Reference cryoSPARC `.cs` file used to copy per-particle metadata |
| `orig_min` | `float` | `-5.0` | Minimum value used to map normalized image values back to MRC-space |
| `orig_max` | `float` | `5.0` | Maximum value used to map normalized image values back to MRC-space |
| `dataset_transform` | transform | `None` | Transform pipeline used by the dataset; any `Normalize` step is inverted |
| `normalize_mean` | `float` or sequence | `None` | Explicit single-channel normalization mean, used with `normalize_std` |
| `normalize_std` | `float` or sequence | `None` | Explicit single-channel normalization std, used with `normalize_mean` |
| `output_prefix` | `str` | `'generated_particles'` | Prefix for output files |

---

### Dataset

#### `CryoMetaData`

```python
from cryodata.cryoemDataset import CryoMetaData

meta_data = CryoMetaData(processed_data_path='path/to/processed/data')
```

Loads and stores all metadata for a preprocessed cryo-EM dataset using LMDB storage. Only `processed_data_path` is required; all other parameters are optional.

If the processed dataset contains `labels_score_source.data`, CryoIEF loads it as the primary per-particle provenance label with `0=calculated score`, `1=_good/_bad default score`, and `2=missing score in a non-_good/_bad dataset`. For backward compatibility, older processed datasets that only contain `labels_used_default_score.data` are still supported by synthesizing `labels_score_source` as `0` for calculated labels and `1` for default/imputed labels. The legacy in-memory `labels_used_default_score` view is still exposed and is derived from `labels_score_source` with `0 -> 0` and `{1,2} -> 1`.

<!-- If the processed dataset contains `labels_data_source.data`, CryoIEF loads it as the per-particle source-modality label. Supported string values are `ptcls`, `mics`, `et_tilts`, and `et_ptcls`. Older datasets without this file are treated as `ptcls`. -->

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `processed_data_path` | `str` | — | Path to the directory produced by `raw_data_preprocess` |
| `emfile_path` | `str` | `None` | Optional path to a `.star` or `.cs` particle file for selection/filtering |
| `selected_emfile_path` | `str` | `None` | Optional path to a second particle file specifying selected particles |

---

#### `CryoEMDataset`

```python
from cryodata.cryoemDataset import CryoEMDataset

dataset = CryoEMDataset(metadata=meta_data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
```

A `torch.utils.data.Dataset` that loads preprocessed cryo-EM particles from an LMDB database. Images larger than 384 pixels are treated as micrographs; smaller images are treated as particles. A single particle transform can be passed as `CryoEMDataset(metadata, transform=...)`; use `get_transforms({'ptcls': ..., 'mics': ...})` when routing separate transforms by data source.

---



### Format Conversion

#### `cs2star`

```python
from cryodata.cs_star_translate.cs2star import cs2star

# Single CS file
cs2star('particles.cs', 'output.star')

# CS file with a passthrough file
cs2star(['particles.cs', 'passthrough_particles.cs'], 'output.star')
```

Converts a cryoSPARC `.cs` file to a RELION-compatible STAR file. When multiple input paths are provided, the first is the primary `.cs` file and the rest are passthrough files whose columns are merged in. The output STAR file includes RELION 3.1 optics group metadata.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `str` or `list[str]` | — | Path(s) to `.cs` file(s); if a list, first entry is the primary file and the rest are passthroughs |
| `output` | `str` | — | Path for the output `.star` file |
| `minphic` | `float` | `None` | Minimum posterior probability threshold for class assignment |
| `boxsize` | `int` | `None` | Override particle box size in the output |
| `noswapxy` | `bool` | `False` | Disable the default X/Y coordinate swap when converting normalized particle coordinates |
| `invertx` | `bool` | `False` | Invert X coordinates |
| `inverty` | `bool` | `False` | Invert Y coordinates |

---

### FFT Utilities

The `fft` module provides centered Fourier and Hartley transforms for 2D cryo-EM images. All functions accept NumPy arrays; `fft2_center` and `ht2_center` additionally accept PyTorch tensors via `tensor=True`.

```python
from cryodata.data_preprocess import fft
```

| Function | Description |
|----------|-------------|
| `fft2_center(img, tensor=False)` | 2D centered FFT. Returns a complex array of the same shape as `img`. |
| `fftn_center(img)` | N-D centered FFT. |
| `ht2_center(img, tensor=False)` | 2D Hartley transform: `Re(FFT) - Im(FFT)`. Real-valued output, same shape as `img`. |
| `htn_center(img)` | N-D Hartley transform. |
| `iht2_center(img)` | Inverse 2D Hartley transform. |
| `ihtn_center(vol)` | Inverse N-D Hartley transform. |
| `symmetrize_ht(ht)` | Adds a wrap-around row and column to a Hartley-transformed image or batch `(N, H, W)`, making it `(N, H+1, W+1)`. Required before saving FFT data for downstream use. |
| `symmetrize_ht_torch(ht)` | Same as `symmetrize_ht` but for PyTorch tensors. |

**Example — compute a real-valued Fourier representation:**

```python
import numpy as np
from cryodata.data_preprocess import fft
from cryodata.data_preprocess.mrc_preprocess import window_mask

image = np.random.randn(224, 224).astype(np.float32)

# Apply a window mask to reduce edge ringing
mask = window_mask(224, in_rad=0.85)
image_windowed = image * mask

# 2D Hartley transform
ht = fft.ht2_center(image_windowed)

# Add the wrap-around border required by downstream models
ht_sym = fft.symmetrize_ht(ht)  # shape: (225, 225)
```

## Dependencies

| Category | Packages |
|----------|---------|
| Deep learning | `torch`, `torchvision`, `accelerate` |
| Scientific computing | `numpy`, `scipy`, `numba`, `pyFFTW` |
| Data handling | `pandas`, `lmdb`, `mrcfile` |
| Cryo-EM | `cryosparc_tools` |
| ML utilities | `scikit-learn`, `annoy` |
| Visualization | `matplotlib`, `seaborn`, `Pillow` |

## Related Projects

- [cryo-IEF](https://github.com/westlake-repl/Cryo-IEF) — the deep learning model this package was built to support
- [CryoRanker](https://github.com/westlake-repl/Cryo-IEF) — a deep learning model for cryo-EM particle ranking
- [CryoDECO](https://github.com/yanyang1998/CryoDECO) — an _ab initio_ heterogeneous reconstruction algorithm that leverages Cryo-IEF priors
- [CryoWizard](https://github.com/SMART-StructBio-AI/CryoWizard) — integrates CryoRanker into a fully automated single-particle cryo-EM processing pipeline
- [cryoSPARC](https://cryosparc.com/) — upstream software for particle extraction and reconstruction

## Citation

Please cite the following paper if this work is useful for your research:
```
@article{yan_comprehensive_2025,
	title = {A comprehensive foundation model for cryo-{EM} image processing},
	issn = {1548-7105},
	url = {https://doi.org/10.1038/s41592-025-02916-8},
	doi = {10.1038/s41592-025-02916-8},
	abstract = {Cryogenic electron microscopy (cryo-EM) has become a premier technique for determining high-resolution structures of biological macromolecules. However, its broad application is constrained by the demand for specialized expertise. Here, to address this limitation, we introduce the Cryo-EM Image Evaluation Foundation (Cryo-IEF) model, a versatile tool pre-trained on {\textasciitilde}65 million cryo-EM particle images through unsupervised learning. Cryo-IEF performs diverse cryo-EM processing tasks, including particle classification by structure, pose-based clustering and image quality assessment. Building on this foundation, we developed CryoWizard, a fully automated single-particle cryo-EM processing pipeline enabled by fine-tuned Cryo-IEF for efficient particle quality ranking. CryoWizard resolves high-resolution structures across samples of varied properties and effectively mitigates the prevalent challenge of preferred orientation in cryo-EM.},
	journal = {Nature Methods},
	author = {Yan, Yang and Fan, Shiqi and Yuan, Fajie and Shen, Huaizong},
	month = nov,
	year = {2025},
}
```
