# Cryo-EM data processing tools for deep learning (e.g., cryo-IEF)

This repository contains tools for processing cryo-EM data, particularly for training deep learning models like
cryo-IEF. Raw cryo-EM data, such as particles extracted from cryoSPARC jobs, 
is normally in MRC format and requires preprocessing before it can be used for training. 
The tools are designed to handle various tasks such as data augmentation, normalization, and preparation of
datasets for training. For more details on implementation and usage, refer to
the [cryo-IEF repository](https://github.com/westlake-repl/Cryo-IEF).

## Installation
To install the required packages, run the following command:

```bash 
pip install cryodata
```

## Usage

To use the tools in this repository, you can import the necessary modules in your Python scripts. For example:

```python
from cryodata.data_preprocess.mrc_preprocess import raw_data_preprocess
from cryodata.cryoemDataset import CryoEMDataset, CryoMetaData
import torch

raw_data_path = 'path/to/cryosparc/particles/job'  # path to the raw cryo-EM data from a cryosparc job (e.g., particles extraction)
processed_data_path = 'path/to/processed/data'  # path to save the processed cryoem data

# preprocess the cryoem data (e.g., particles data from a cryosparc job)
new_cs_data = raw_data_preprocess(raw_data_path, processed_data_path,
                        resize=224,
                        save_raw_data=False,
                        save_FT_data=False,
                        is_to_int8=True)
# create a dataset from the processed data
meta_data = CryoMetaData(processed_data_path=processed_data_path)
cryodataset = CryoEMDataset(metadata=meta_data)
dataloader =torch.utils.data.DataLoader(cryodataset, batch_size=32, shuffle=True)

```
In this example, we preprocess the raw cryo-EM data and create a dataset that can be used for training deep learning models.