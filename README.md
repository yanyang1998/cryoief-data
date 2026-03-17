# cryodata

[![PyPI version](https://badge.fury.io/py/cryodata.svg)](https://badge.fury.io/py/cryodata)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
<a href="https://doi.org/10.1038/s41592-025-02916-8"><img src="https://img.shields.io/badge/Paper-Nature%20Methods-blue" style="max-width: 100%;"></a>
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Cryo-EM data processing tools for deep learning. This package provides a full pipeline for converting raw cryo-EM particle data from [cryoSPARC](https://cryosparc.com/) into PyTorch-ready datasets, as used by [cryo-IEF](https://github.com/westlake-repl/Cryo-IEF), [CryoDECO](https://github.com/yanyang1998/CryoDECO) and [CryoWizard](https://github.com/SMART-StructBio-AI/CryoWizard).

## Features


- **Preprocessing pipeline** — resize, normalize, and window-mask cryo-EM particles from cryoSPARC jobs
- **LMDB dataset creation** — fast multi-process conversion of MRC stacks into LMDB databases for efficient training I/O
- **PyTorch dataset & sampler** — `CryoEMDataset` and `CryoMetaData` classes with support for balanced resampling
- **Fourier-space representations** — optional FFT/Hilbert-transform outputs alongside real-space images
- **Format conversion** — convert cryoSPARC `.cs` files to RELION `.star` format

## Installation

```bash
pip install cryodata
```

For development:

```bash
git clone https://github.com/yanyang1998/cryoief-data
cd cryoief-data
pip install -e .
```

## Quick Start

```python
from cryodata.data_preprocess.mrc_preprocess import raw_data_preprocess
from cryodata.cryoemDataset import CryoEMDataset, CryoMetaData
import torch

raw_data_path = 'path/to/cryosparc/particles/job'
processed_data_path = 'path/to/processed/data'

# Step 1: Preprocess raw cryoSPARC particle data
new_cs_data = raw_data_preprocess(
    raw_data_path,
    processed_data_path,
    resize=224,          # resize particles to 224×224
    save_raw_data=False, # skip saving unprocessed images
    save_FT_data=False,  # skip saving Fourier-space images
    is_to_int8=True,     # convert to uint8 for storage efficiency
)

# Step 2: Load the dataset
meta_data = CryoMetaData(processed_data_path=processed_data_path)
cryodataset = CryoEMDataset(metadata=meta_data)

# Step 3: Create a DataLoader for training
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
    save_raw_data=True,
    save_FT_data=True,
    use_lmdb=True,
    num_processes=8,
)
```

The main entry point for the preprocessing pipeline. Reads cryoSPARC `.cs` metadata and associated MRC particle stacks from `raw_dataset_dir`, applies the selected transforms, and writes the output to `dataset_save_dir`. Internally it calls `raw_csdata_process_from_cryosparc_dir` to locate and merge the correct `.cs` files, then builds an LMDB database (when `use_lmdb=True`) or individual pickle files (when `use_lmdb=False`). Returns the merged cryoSPARC `Dataset` object.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `raw_dataset_dir` | `str` | — | Path to a cryoSPARC job output directory (e.g., a particle extraction job) |
| `dataset_save_dir` | `str` | — | Directory where processed data and metadata will be saved |
| `resize` | `int` | `224` | Target image size in pixels (square); uses FFT-based downsampling when reducing, bicubic otherwise |
| `is_to_int8` | `bool` | `True` | Normalize each particle to [0, 255] and cast to `uint8` for compact storage |
| `save_raw_data` | `bool` | `True` | Save unprocessed raw particles alongside the processed ones (only applies when `use_lmdb=False`) |
| `save_FT_data` | `bool` | `True` | Compute and save Hartley-transform (real-valued Fourier) representations (only applies when `use_lmdb=False`) |
| `use_lmdb` | `bool` | `True` | Write output to an LMDB database for fast I/O during training (recommended) |
| `num_processes` | `int` | `8` | Number of worker processes for parallel MRC file processing |

---

#### `raw_csdata_process_from_cryosparc_dir`

```python
from cryodata.data_preprocess.mrc_preprocess import raw_csdata_process_from_cryosparc_dir

cs_data, mrc_dir = raw_csdata_process_from_cryosparc_dir(raw_data_path)
```

Scans a cryoSPARC job directory and locates the relevant `.cs` particle file and the corresponding MRC stack directory. Handles various cryoSPARC job types (extraction, import, restack, downsampling). When both a particles `.cs` file and a passthrough file are found, they are merged via an inner join. Returns the `Dataset` object and the path (or list of paths) to the MRC stacks.

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
| `width` | `int` | — | Target width (and height, if `height` is not given) in pixels |
| `height` | `int` | `None` | Target height; defaults to `width` for square output |
| `is_freqs` | `bool` | `True` | When `True` and target is smaller than source, downsample in the Fourier domain (FFT crop); otherwise use bicubic spatial interpolation |

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

Generates a 2D radial cosine-edge windowing mask of shape `(resolution, resolution)`. The mask is 1.0 inside `in_rad` and tapers smoothly to 0.0 at `out_rad`. Useful for suppressing edge artifacts before computing FFTs.

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

---

### Dataset

#### `CryoMetaData`

```python
from cryodata.cryoemDataset import CryoMetaData

meta_data = CryoMetaData(processed_data_path='path/to/processed/data')
```

Loads and stores all metadata for a preprocessed cryo-EM dataset. Automatically detects whether the dataset uses LMDB storage or individual pickle files. Only `processed_data_path` is required; all other parameters are optional.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `processed_data_path` | `str` | — | Path to the directory produced by `raw_data_preprocess` |
| `emfile_path` | `str` | `None` | Optional path to a `.star` or `.cs` particle file for selection/filtering |
| `selected_emfile_path` | `str` | `None` | Optional path to a second particle file specifying selected particles |
| `ctf_correction_averages` | `bool` | `False` | Load CTF-corrected class-average paths if available |
| `ctf_correction_inference` | `bool` | `False` | Load CTF-corrected particle paths for inference if available |

---

#### `CryoEMDataset`

```python
from cryodata.cryoemDataset import CryoEMDataset

dataset = CryoEMDataset(metadata=meta_data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
```

A `torch.utils.data.Dataset` that loads preprocessed cryo-EM particles from an LMDB database or pickle files. Images larger than 384 pixels are treated as micrographs; smaller images are treated as particles. Supports optional on-the-fly transforms passed at construction time.

---

### Resampling

#### `MyResampleSampler`

```python
from cryodata import MyResampleSampler

sampler = MyResampleSampler(
    data=dataset,
    id_index_dict_pos=pos_dict,   # {class_id: [indices]} for high-quality particles
    id_index_dict_mid=mid_dict,   # {class_id: [indices]} for medium-quality particles
    id_index_dict_neg=neg_dict,   # {class_id: [indices]} for low-quality particles
    resample_num_pos=500,         # max particles per class from the positive set
    resample_num_mid=200,         # max particles per class from the medium set
    resample_num_neg=100,         # max particles per class from the negative set
)
```

A `torch.utils.data.Sampler` designed for fine-tuning scenarios where particles have been labelled as positive, mid, or negative quality. At each epoch it resamples each class up to the specified cap, then concatenates the three groups into a single index list. Shuffle behaviour is controlled by `shuffle_type` (`'all'`, `'class'`, or `'batch'`).

---

#### `MyResampleSampler_pretrain`

```python
from cryodata import MyResampleSampler_pretrain

sampler = MyResampleSampler_pretrain(
    id_index_dict=id_index_dict,     # {class_id: [indices]}
    batch_size_all=256,              # total batch size across all processes
    max_number_per_sample=1000,      # max particles sampled per class per epoch
    shuffle_type='class',            # 'all', 'class', or 'batch' (int)
    shuffle_mix_up_ratio=0.2,        # fraction of each class used for cross-class mixing
    bad_particles_ratio=0.1,         # fraction of slots given to low-quality particles
)
```

A `torch.utils.data.Sampler` for pre-training with large multi-class datasets. Resamples each class up to `max_number_per_sample` and optionally mixes a fraction of particles across classes to improve generalisation. Supports multi-process training via `num_processes`.

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
| `noswapxy` | `bool` | `False` | Disable the default X/Y coordinate swap |
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
