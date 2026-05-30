"""Update mixed_symmetry_info.json so that every HouseCat6D object_id follows the
OFFICIAL HouseCat6D / VI-Net symmetry rule used in
HouseCat6D/VI-Net/utils/evaluation_utils.py::compute_RT_degree_cm_symmetry.

Official rule (from synset_names = ['BG','box','bottle','can','cup','remote',
                                    'teapot','cutlery','glass','shoe','tube']):

  bottle / can / bowl / glass  -> y-axis continuous symmetry
  phone / eggbox / glue        -> 180-deg y-axis discrete symmetry  (none in HC6D)
  mug + handle_visibility==0   -> y-axis continuous symmetry         (none in HC6D)
  everything else (box, cup, cutlery, remote, shoe, teapot, tube)  -> no symmetry

We:
  - load all 194 HouseCat6D objects via HouseCat6DCameraPose
  - look up each object_id's category
  - overwrite the HouseCat6DCameraPose:<id> entry with the official-rule body
  - leave non-HouseCat6DCameraPose entries (OO9D, YCBV, REAL275) untouched
  - back up the original to mixed_symmetry_info.json.bak before writing

Run:
  python diagnostics/update_hc_symmetry_official.py [--dry-run]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PROJECT = Path('/mnt/train-data-4-hdd/yian/freepose/omni-object_clone')
SYMM_PATH = PROJECT / 'mixed_symmetry_info.json'
BAK_PATH = PROJECT / 'mixed_symmetry_info.json.bak'
DATASET_LABEL = 'HouseCat6DCameraPose'

# Per official compute_RT_degree_cm_symmetry: only these have y-axis continuous symmetry.
# Note: HouseCat6D categories are ['box','bottle','can','cup','remote','teapot',
#                                  'cutlery','glass','shoe','tube'].
# 'bowl' is in the official rule but NOT in HouseCat6D taxonomy, so unreachable here.
Y_AXIS_CONTINUOUS = {'bottle', 'can', 'glass'}

Y_AXIS_DISCRETE = {'phone', 'eggbox', 'glue'}   # 180-deg y-axis -- none in HC6D
MUG_LIKE = {'mug'}                              # not in HC6D


def y_continuous_body() -> dict:
    return {
        'symmetries_continuous': [
            {'axis': [0, 1, 0], 'offset': [0, 0, 0]}
        ]
    }


def y_discrete_180_body() -> dict:
    # 180-deg rotation about y, in 4x4 homogeneous form.
    return {
        'symmetries_discrete': [
            [
                [-1.0, 0.0, 0.0, 0.0],
                [ 0.0, 1.0, 0.0, 0.0],
                [ 0.0, 0.0,-1.0, 0.0],
                [ 0.0, 0.0, 0.0, 1.0],
            ]
        ]
    }


def empty_body() -> dict:
    return {}


def rule_for_category(category: str) -> dict:
    cat = str(category).lower()
    if cat in Y_AXIS_CONTINUOUS:
        return y_continuous_body()
    if cat in Y_AXIS_DISCRETE:
        return y_discrete_180_body()
    # mug w/ handle_visibility==0 would be y-continuous, but HC6D has no mug;
    # cup is NOT in the official symmetric set despite intuition.
    return empty_body()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true',
                        help="Don't write the file, just print diff summary.")
    args = parser.parse_args()

    # Load existing JSON
    with SYMM_PATH.open() as fh:
        sym = json.load(fh)

    # Enumerate HouseCat6D objects via the dataset class to get object_id -> category
    sys.path.insert(0, str(PROJECT))
    from omnivggt.datasets.housecat6d.housecat6d_camera_pose import HouseCat6DCameraPose
    ds = HouseCat6DCameraPose(
        dataset_location='/mnt/train-data-4-hdd/yian/freepose/housecat6d',
        dset='train',
        object_image_root='/mnt/train-data-4-hdd/yian/freepose/housecat6d/housecat6d_aligned_object_refs',
        align_json=str(PROJECT / 'dataset_align.json'),
        verify_files=False,
        z_far=20,
        resolution=(518, 476),
    )

    id_to_cat = {}
    for name, oid in ds.object_name_to_id.items():
        rec = ds.object_records_by_name[name]
        id_to_cat[int(oid)] = (str(name), str(rec.get('category', '')))

    # Build planned updates
    additions, modifications, untouched = 0, 0, 0
    cat_counts = {}
    for oid, (name, cat) in id_to_cat.items():
        key = f'{DATASET_LABEL}:{oid}'
        new_body = rule_for_category(cat)
        old_body = sym.get(key)
        cat_counts.setdefault(cat, []).append(name)
        if old_body is None:
            additions += 1
        elif old_body != new_body:
            modifications += 1
        else:
            untouched += 1
        sym[key] = new_body

    # Summary
    print('Official-rule application plan:')
    for cat in sorted(cat_counts.keys()):
        body = rule_for_category(cat)
        kind = 'y-continuous' if body.get('symmetries_continuous') else ('y-discrete-180' if body.get('symmetries_discrete') else 'NO symmetry')
        print(f'  {cat:10s}  N={len(cat_counts[cat]):>3d}  -> {kind}')
    print(f'\nKeys: additions={additions}, modifications={modifications}, untouched={untouched}')

    if args.dry_run:
        print('\n[dry-run] not writing the file. Sample diff for a few keys:')
        for oid in [1, 2, 17]:
            key = f'{DATASET_LABEL}:{oid}'
            name, cat = id_to_cat.get(oid, ('?', '?'))
            print(f'  {key}  ({name}, category={cat})')
            print(f'    new -> {rule_for_category(cat)}')
        return

    # Back up
    if not BAK_PATH.exists():
        BAK_PATH.write_text(SYMM_PATH.read_text())
        print(f'Backed up original to {BAK_PATH}')

    # Write
    SYMM_PATH.write_text(json.dumps(sym, indent=2))
    print(f'Wrote updated {SYMM_PATH}')


if __name__ == '__main__':
    main()
