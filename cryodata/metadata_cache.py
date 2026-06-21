import argparse
import json
import os
import pickle
import shutil
import time
from pathlib import Path

import numpy as np


CACHE_VERSION = 1
CACHE_ROOT_NAME = '.cryo_meta_cache'
CACHE_VERSION_NAME = 'v1'
CACHE_MANIFEST_NAME = 'metadata_cache.json'
AUTO_BUILD_MIN_LENGTH = 10_000_000
AUTO_BUILD_MIN_PROTEIN_ID_FILE_SIZE = 64 * 1024 * 1024

DATA_SOURCE_PTCLS = 'ptcls'
DATA_SOURCE_MICS = 'mics'
DATA_SOURCE_ET_TILTS = 'et_tilts'
DATA_SOURCE_ET_PTCLS = 'et_ptcls'
DATA_SOURCE_LABELS = (
    DATA_SOURCE_PTCLS,
    DATA_SOURCE_MICS,
    DATA_SOURCE_ET_TILTS,
    DATA_SOURCE_ET_PTCLS,
)
DATA_SOURCE_TO_CODE = {label: idx for idx, label in enumerate(DATA_SOURCE_LABELS)}


class EncodedLabelArray:
    """Small wrapper around encoded label arrays that returns public string labels."""

    def __init__(self, codes, labels):
        self.codes = codes
        self.labels = tuple(labels)

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, item):
        values = self.codes[item]
        if np.isscalar(values):
            return self.labels[int(values)]
        return np.asarray([self.labels[int(value)] for value in values], dtype=object)

    def __iter__(self):
        for code in self.codes:
            yield self.labels[int(code)]

    def __eq__(self, other):
        return self.tolist() == list(other)

    def tolist(self):
        return [self.labels[int(code)] for code in self.codes]

    def take(self, indices):
        return self.codes[indices]

    def take_labels(self, indices):
        return np.asarray([self.labels[int(value)] for value in self.codes[indices]], dtype=object)

    @property
    def nbytes(self):
        return getattr(self.codes, 'nbytes', 0)


class LazyPoseIdMap:
    """Lazy per-protein pose map backed by compact npy files."""

    def __init__(self, root, manifest):
        self.root = Path(root)
        self.manifest = {int(key): value for key, value in manifest.items()}
        self._cache = {}

    def get(self, protein_id, default=None):
        protein_id = int(protein_id)
        if protein_id not in self.manifest:
            return default
        if protein_id not in self._cache:
            entry = self.manifest[protein_id]
            keys = np.load(self.root / entry['keys'], mmap_mode='r')
            values = np.load(self.root / entry['values'], mmap_mode='r')
            self._cache[protein_id] = {int(key): int(value) for key, value in zip(keys, values)}
        return self._cache[protein_id]

    def __len__(self):
        return len(self.manifest)


def default_cache_dir(processed_data_path):
    return str(Path(processed_data_path) / CACHE_ROOT_NAME / CACHE_VERSION_NAME)


def _pickle_path(processed_data_path, filename):
    return Path(processed_data_path) / filename


def _load_pickle(path, default=None):
    if not Path(path).exists():
        return default
    with open(path, 'rb') as f:
        return pickle.load(f)


def _source_file_record(path):
    path = Path(path)
    if not path.exists():
        return None
    stat = path.stat()
    return {
        'size': int(stat.st_size),
        'mtime_ns': int(stat.st_mtime_ns),
    }


def _same_source_file(path, record):
    if record is None:
        return not Path(path).exists()
    current = _source_file_record(path)
    return current == record


def _atomic_save_npy(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with open(tmp_path, 'wb') as f:
        np.save(f, array)
    os.replace(tmp_path, path)


def _validate_length(name, values, expected_length, data_path):
    if values is None:
        return
    if len(values) != expected_length:
        raise ValueError(
            f'{name} length mismatch: expected {expected_length}, found {len(values)} in {data_path}.'
        )


def _normalize_score_source(score_source_values, legacy_default_values, expected_length):
    if score_source_values is not None:
        values = np.asarray(score_source_values, dtype=np.int16)
        if len(values) == 0:
            return np.zeros(expected_length, dtype=np.int16)
        if len(values) != expected_length:
            return np.zeros(expected_length, dtype=np.int16)
        return values

    if legacy_default_values is not None:
        values = np.asarray(legacy_default_values, dtype=np.int16)
        if len(values) == 0:
            return np.zeros(expected_length, dtype=np.int16)
        if len(values) != expected_length:
            return np.zeros(expected_length, dtype=np.int16)
        return np.where(values == 0, 0, 1).astype(np.int16, copy=False)

    return np.zeros(expected_length, dtype=np.int16)


def _derive_used_default_score(score_source_values):
    values = np.asarray(score_source_values, dtype=np.int16)
    return np.where((values == 0) | (values == 3), 0, 1).astype(np.int8, copy=False)


def _encode_data_source(values, expected_length, data_path):
    if values is None:
        return np.full(expected_length, DATA_SOURCE_TO_CODE[DATA_SOURCE_PTCLS], dtype=np.uint8)
    if len(values) != expected_length:
        raise ValueError(
            f'labels_data_source.data length mismatch: expected {expected_length}, found {len(values)} in {data_path}.'
        )
    codes = np.empty(expected_length, dtype=np.uint8)
    for idx, value in enumerate(values):
        label = str(value)
        if label not in DATA_SOURCE_TO_CODE:
            raise ValueError(
                f"labels_data_source.data contains invalid data source label: {label!r}."
            )
        codes[idx] = DATA_SOURCE_TO_CODE[label]
    return codes


def _build_protein_index(protein_id_list):
    protein_ids = np.asarray(protein_id_list, dtype=np.int64)
    first = {}
    counts = {}
    contiguous = {}
    previous_id = None
    closed = set()
    for idx, protein_id in enumerate(protein_ids):
        protein_id = int(protein_id)
        if protein_id not in first:
            first[protein_id] = idx
            counts[protein_id] = 0
            contiguous[protein_id] = True
        elif previous_id != protein_id and protein_id in closed:
            contiguous[protein_id] = False
        counts[protein_id] += 1
        if previous_id is not None and previous_id != protein_id:
            closed.add(previous_id)
        previous_id = protein_id

    ordered_ids = np.asarray(sorted(first.keys()), dtype=np.int64)
    starts = np.asarray([first[int(protein_id)] for protein_id in ordered_ids], dtype=np.int64)
    count_values = np.asarray([counts[int(protein_id)] for protein_id in ordered_ids], dtype=np.int64)
    contiguous_values = np.asarray([contiguous[int(protein_id)] for protein_id in ordered_ids], dtype=np.bool_)
    return ordered_ids, starts, count_values, contiguous_values


def _build_lmdb_reference_cache(processed_data_path, cache_dir):
    manifest_path = Path(processed_data_path) / 'lmdb_reference_manifest.json'
    if not manifest_path.exists():
        return None

    with manifest_path.open('r') as f:
        manifest = json.load(f)

    proteins = manifest.get('proteins') if isinstance(manifest, dict) else None
    if not isinstance(proteins, list):
        raise ValueError('lmdb_reference_manifest.json must contain a proteins list.')

    segments_by_protein = {}
    source_indices_dir = Path(cache_dir) / 'source_local_indices'
    for protein_entry in proteins:
        protein_id = int(protein_entry['protein_id'])
        cached_segments = []
        for segment_idx, segment in enumerate(protein_entry.get('segments', [])):
            cached_segment = {
                'merged_local_start': int(segment['merged_local_start']),
                'count': int(segment['count']),
                'processed_dir': segment['db_paths']['lmdb_processed'],
                'raw_dir': segment.get('db_paths', {}).get('lmdb_raw'),
                'ft_dir': segment.get('db_paths', {}).get('lmdb_FT'),
            }
            source_local_indices = segment.get('source_local_indices')
            if source_local_indices is not None:
                source_indices = np.asarray(source_local_indices, dtype=np.int64)
                rel_path = f'{protein_id}_{segment_idx}.npy'
                _atomic_save_npy(source_indices_dir / rel_path, source_indices)
                cached_segment['source_local_indices_file'] = rel_path
            cached_segments.append(cached_segment)
        segments_by_protein[str(protein_id)] = cached_segments
    return {
        'segments_by_protein': segments_by_protein,
        'source_indices_dir': 'source_local_indices',
    }


def _build_pose_map_cache(processed_data_path, cache_dir):
    pose_path = Path(processed_data_path) / 'pose_id_map.data'
    if not pose_path.exists():
        return None
    pose_map = _load_pickle(pose_path, default=None)
    if not pose_map:
        return None

    pose_dir = Path(cache_dir) / 'pose_id_map'
    manifest = {}
    for protein_id, protein_pose_map in pose_map.items():
        if not protein_pose_map:
            continue
        keys = np.asarray(list(protein_pose_map.keys()), dtype=np.int64)
        values = np.asarray(list(protein_pose_map.values()), dtype=np.int64)
        key_rel = f'{int(protein_id)}_keys.npy'
        value_rel = f'{int(protein_id)}_values.npy'
        _atomic_save_npy(pose_dir / key_rel, keys)
        _atomic_save_npy(pose_dir / value_rel, values)
        manifest[str(int(protein_id))] = {
            'keys': key_rel,
            'values': value_rel,
        }
    return {
        'root': 'pose_id_map',
        'proteins': manifest,
    }


def is_cache_fresh(processed_data_path, cache_dir=None):
    processed_data_path = Path(processed_data_path)
    cache_dir = Path(cache_dir or default_cache_dir(processed_data_path))
    manifest_path = cache_dir / CACHE_MANIFEST_NAME
    if not manifest_path.exists():
        return False
    try:
        with manifest_path.open('r') as f:
            manifest = json.load(f)
    except Exception:
        return False
    if manifest.get('version') != CACHE_VERSION:
        return False
    for filename, record in manifest.get('source_files', {}).items():
        if not _same_source_file(processed_data_path / filename, record):
            return False
    for name, rel_path in manifest.get('arrays', {}).items():
        if not (cache_dir / rel_path).exists():
            return False
    return True


def _wait_for_cache_or_lock(processed_data_path, cache_dir, lock_dir, timeout_seconds):
    start = time.time()
    while time.time() - start < timeout_seconds:
        if is_cache_fresh(processed_data_path, cache_dir):
            return True
        if not lock_dir.exists():
            return False
        time.sleep(2.0)
    return is_cache_fresh(processed_data_path, cache_dir)


def build_metadata_cache(processed_data_path, metadata_cache_dir=None, force=False):
    processed_data_path = Path(processed_data_path)
    cache_dir = Path(metadata_cache_dir or default_cache_dir(processed_data_path))
    if cache_dir.exists() and is_cache_fresh(processed_data_path, cache_dir) and not force:
        return str(cache_dir)

    cache_parent = cache_dir.parent
    cache_parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = cache_parent / f'{cache_dir.name}.tmp.{os.getpid()}'
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    source_filenames = [
        'protein_id_list.data',
        'labels_classification.data',
        'labels_for_clustering.data',
        'labels_score_source.data',
        'labels_used_default_score.data',
        'labels_data_source.data',
        'source_mrcs_group_id.data',
        'lmdb_reference_manifest.json',
        'pose_id_map.data',
    ]

    protein_id_list = _load_pickle(_pickle_path(processed_data_path, 'protein_id_list.data'))
    if protein_id_list is None:
        raise ValueError(f'{processed_data_path} is missing protein_id_list.data.')
    length = len(protein_id_list)
    protein_ids = np.asarray(protein_id_list, dtype=np.int64)
    _atomic_save_npy(tmp_dir / 'protein_id_list.npy', protein_ids)

    labels_classification = _load_pickle(_pickle_path(processed_data_path, 'labels_classification.data'), default=None)
    if labels_classification is None or len(labels_classification) == 0:
        classification = np.ones(length, dtype=np.float32)
    else:
        _validate_length('labels_classification.data', labels_classification, length, processed_data_path)
        classification = np.asarray(labels_classification, dtype=np.float32)
    _atomic_save_npy(tmp_dir / 'labels_classification.npy', classification)

    labels_for_clustering = _load_pickle(_pickle_path(processed_data_path, 'labels_for_clustering.data'), default=None)
    arrays = {
        'protein_id_list': 'protein_id_list.npy',
        'labels_classification': 'labels_classification.npy',
    }
    if labels_for_clustering is not None and len(labels_for_clustering) > 0:
        _validate_length('labels_for_clustering.data', labels_for_clustering, length, processed_data_path)
        _atomic_save_npy(tmp_dir / 'labels_for_clustering.npy', np.asarray(labels_for_clustering, dtype=np.int64))
        arrays['labels_for_clustering'] = 'labels_for_clustering.npy'

    labels_score_source_raw = _load_pickle(_pickle_path(processed_data_path, 'labels_score_source.data'), default=None)
    legacy_default_score = None
    if labels_score_source_raw is None:
        legacy_default_score = _load_pickle(
            _pickle_path(processed_data_path, 'labels_used_default_score.data'),
            default=None,
        )
    score_source = _normalize_score_source(labels_score_source_raw, legacy_default_score, length)
    used_default_score = _derive_used_default_score(score_source)
    _atomic_save_npy(tmp_dir / 'labels_score_source.npy', score_source)
    _atomic_save_npy(tmp_dir / 'labels_used_default_score.npy', used_default_score)
    arrays['labels_score_source'] = 'labels_score_source.npy'
    arrays['labels_used_default_score'] = 'labels_used_default_score.npy'

    labels_data_source = _load_pickle(_pickle_path(processed_data_path, 'labels_data_source.data'), default=None)
    data_source_codes = _encode_data_source(labels_data_source, length, processed_data_path)
    _atomic_save_npy(tmp_dir / 'labels_data_source_codes.npy', data_source_codes)
    arrays['labels_data_source_codes'] = 'labels_data_source_codes.npy'

    source_mrcs_group_id = _load_pickle(_pickle_path(processed_data_path, 'source_mrcs_group_id.data'), default=None)
    if source_mrcs_group_id is not None:
        _validate_length('source_mrcs_group_id.data', source_mrcs_group_id, length, processed_data_path)
        _atomic_save_npy(tmp_dir / 'source_mrcs_group_id.npy', np.asarray(source_mrcs_group_id, dtype=np.int64))
        arrays['source_mrcs_group_id'] = 'source_mrcs_group_id.npy'

    protein_index_ids, protein_index_starts, protein_index_counts, protein_index_contiguous = _build_protein_index(
        protein_ids
    )
    _atomic_save_npy(tmp_dir / 'protein_index_ids.npy', protein_index_ids)
    _atomic_save_npy(tmp_dir / 'protein_index_starts.npy', protein_index_starts)
    _atomic_save_npy(tmp_dir / 'protein_index_counts.npy', protein_index_counts)
    _atomic_save_npy(tmp_dir / 'protein_index_contiguous.npy', protein_index_contiguous)
    arrays.update({
        'protein_index_ids': 'protein_index_ids.npy',
        'protein_index_starts': 'protein_index_starts.npy',
        'protein_index_counts': 'protein_index_counts.npy',
        'protein_index_contiguous': 'protein_index_contiguous.npy',
    })

    lmdb_reference_cache = _build_lmdb_reference_cache(processed_data_path, tmp_dir)
    pose_map_cache = _build_pose_map_cache(processed_data_path, tmp_dir)

    manifest = {
        'version': CACHE_VERSION,
        'length': int(length),
        'arrays': arrays,
        'data_source_labels': list(DATA_SOURCE_LABELS),
        'source_files': {
            filename: _source_file_record(processed_data_path / filename)
            for filename in source_filenames
        },
        'lmdb_reference_cache': lmdb_reference_cache,
        'pose_id_map_cache': pose_map_cache,
    }
    with (tmp_dir / CACHE_MANIFEST_NAME).open('w') as f:
        json.dump(manifest, f, indent=2)

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    os.replace(tmp_dir, cache_dir)
    return str(cache_dir)


def ensure_metadata_cache(processed_data_path, metadata_cache_dir=None, mode='auto',
                          auto_build_min_length=AUTO_BUILD_MIN_LENGTH, wait_timeout_seconds=3600):
    mode = (mode or 'auto').lower()
    if mode not in ('auto', 'require', 'off'):
        raise ValueError("metadata_cache_mode must be one of: auto, require, off.")
    if mode == 'off':
        return None

    processed_data_path = Path(processed_data_path)
    cache_dir = Path(metadata_cache_dir or default_cache_dir(processed_data_path))
    if is_cache_fresh(processed_data_path, cache_dir):
        return str(cache_dir)

    if mode == 'require':
        raise RuntimeError(
            f'Metadata cache is required but missing or stale at {cache_dir}. '
            f'Build it with: python -m cryodata.build_metadata_cache --processed_data_path {processed_data_path}'
        )

    protein_id_path = processed_data_path / 'protein_id_list.data'
    protein_id_size = protein_id_path.stat().st_size if protein_id_path.exists() else 0
    if protein_id_size < AUTO_BUILD_MIN_PROTEIN_ID_FILE_SIZE and not cache_dir.exists():
        return None

    lock_dir = cache_dir.parent / f'{cache_dir.name}.lock'
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(lock_dir)
    except FileExistsError:
        if _wait_for_cache_or_lock(processed_data_path, cache_dir, lock_dir, wait_timeout_seconds):
            return str(cache_dir)
        raise RuntimeError(f'Timed out waiting for metadata cache lock: {lock_dir}')

    try:
        return build_metadata_cache(processed_data_path, cache_dir)
    finally:
        try:
            os.rmdir(lock_dir)
        except OSError:
            pass


def load_metadata_cache(processed_data_path, metadata_cache_dir=None):
    cache_dir = Path(metadata_cache_dir or default_cache_dir(processed_data_path))
    with (cache_dir / CACHE_MANIFEST_NAME).open('r') as f:
        manifest = json.load(f)
    arrays = {
        name: np.load(cache_dir / rel_path, mmap_mode='r')
        for name, rel_path in manifest.get('arrays', {}).items()
    }
    labels_data_source = EncodedLabelArray(
        arrays['labels_data_source_codes'],
        manifest.get('data_source_labels', DATA_SOURCE_LABELS),
    )
    pose_cache = manifest.get('pose_id_map_cache')
    pose_id_map = None
    if pose_cache is not None:
        pose_id_map = LazyPoseIdMap(cache_dir / pose_cache['root'], pose_cache['proteins'])
    return {
        'cache_dir': str(cache_dir),
        'manifest': manifest,
        'arrays': arrays,
        'labels_data_source': labels_data_source,
        'pose_id_map': pose_id_map,
    }


def approximate_cache_nbytes(cache_payload):
    if not cache_payload:
        return 0
    total = 0
    for value in cache_payload.get('arrays', {}).values():
        total += int(getattr(value, 'nbytes', 0))
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build compact mmap metadata cache for CryoEM datasets.')
    parser.add_argument('--processed_data_path', required=True)
    parser.add_argument('--metadata_cache_dir', default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    cache_dir = build_metadata_cache(
        args.processed_data_path,
        metadata_cache_dir=args.metadata_cache_dir,
        force=args.force,
    )
    print(cache_dir)


if __name__ == '__main__':
    main()
