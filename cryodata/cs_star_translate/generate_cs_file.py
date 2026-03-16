from cryosparc.dataset import Dataset
import numpy as np


def _set_field(cs_data, key, value, data_len):
    cs_data[key] = [value] * data_len if np.isscalar(value) else value


def generate_cs_file(blob_path, blob_id, image_size, psize, df1, df2, accel_kv=300, cs_mm=2.7, amp_contrast=0.07,
                     phase_shift_rad=0.0, bfactor=0.0):
    data_len = len(blob_path)
    cs_data = Dataset(data_len)
    cs_data.add_fields(
        ['ctf/type', 'ctf/exp_group_id', 'ctf/accel_kv', 'ctf/cs_mm', 'ctf/amp_contrast',
         'ctf/df1_A','ctf/df2_A','ctf/df_angle_rad', 'ctf/phase_shift_rad', 'ctf/scale',
         'ctf/scale_const', 'ctf/shift_A', 'ctf/tilt_A','ctf/trefoil_A', 'ctf/tetra_A',
         'ctf/anisomag', 'ctf/bfactor'],
        ['str', 'int', 'float', 'float', 'float',
         'float', 'float', 'float', 'float', 'float',
         'float', ('float', (2,)),('float', (2,)), ('float', (2,)),('float', (4,)),
         ('float', (4,)), 'float'])
    cs_data.add_fields(
        ['blob/path', 'blob/psize_A', 'blob/idx', 'blob/shape', 'blob/sign','blob/import_sig'],
        ['str', 'float', 'int', ('int', (2,)), 'float',int])

    cs_data['blob/path'] = blob_path
    cs_data['ctf/df1_A'] = df1
    cs_data['ctf/df2_A'] = df2
    cs_data['blob/idx'] = blob_id
    cs_data['blob/sign'] = [1.0] * data_len
    cs_data['ctf/scale'] = [1.0] * data_len
    _set_field(cs_data, 'ctf/accel_kv', accel_kv, data_len)
    _set_field(cs_data, 'blob/shape', image_size, data_len)
    _set_field(cs_data, 'blob/psize_A', psize, data_len)
    _set_field(cs_data, 'ctf/cs_mm', cs_mm, data_len)
    _set_field(cs_data, 'ctf/amp_contrast', amp_contrast, data_len)
    _set_field(cs_data, 'ctf/phase_shift_rad', phase_shift_rad, data_len)
    _set_field(cs_data, 'ctf/bfactor', bfactor, data_len)
    return cs_data
