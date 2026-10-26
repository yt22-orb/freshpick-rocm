"""Prevent accidental split contamination or relabeling of the frozen identity pilot."""
import hashlib
import json
from pathlib import Path
from freshpick.identifier import PRODUCE

ROOT=Path(__file__).parents[1]/'reports/identity-pilot'


def test_frozen_known_groups_and_images_do_not_cross_splits():
    manifest=ROOT/'manifest.json';rows=json.loads(manifest.read_text())
    freeze=json.loads((ROOT/'freeze.json').read_text())
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==freeze['sha256']
    assert len(rows)==80 and len({r['sha256'] for r in rows})==80
    group_splits={}
    for row in rows:group_splits.setdefault(row['group_id'],set()).add(row['split'])
    assert all(len(splits)==1 for splits in group_splits.values())
    for split in ['train','val','test']:
        assert {r['produce'] for r in rows if r['split']==split}==set(PRODUCE)
    assert all(row['freshness_ground_truth'] is None for row in rows)


def test_unsupported_calibration_and_test_photographers_are_disjoint():
    rows=json.loads((ROOT/'ood-manifest.json').read_text())
    val={r['group_id'] for r in rows if r['split']=='val'}
    test={r['group_id'] for r in rows if r['split']=='test'}
    assert val and test and val.isdisjoint(test)
    assert len({r['sha256'] for r in rows})==len(rows)
