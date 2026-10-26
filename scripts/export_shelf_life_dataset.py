"""Export the packaged reference to CSV; check the pinned research snapshot hash."""
import csv
import hashlib
import io
import json
from pathlib import Path

from freshpick.shelf_life import shelf_life_reference


def export_csv():
    reference = shelf_life_reference()
    output = io.StringIO(newline='')
    fields = ['id', 'produce', 'condition', 'location', 'required_ripeness',
              'min_days', 'max_days', 'age_basis', 'source_temperature',
              'evidence_type', 'source_locator', 'source_ids', 'source_urls',
              'notes', 'dataset_version', 'reviewed_on']
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for row in reference['records']:
        record = dict(row)
        record.update(source_ids=';'.join(row['source_ids']),
                      source_urls=';'.join(reference['sources'][key]['url'] for key in row['source_ids']),
                      notes=' '.join(row['notes']), dataset_version=reference['dataset_version'],
                      reviewed_on=reference['reviewed_on'])
        writer.writerow(record)
    return output.getvalue()


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1] / 'datasets' / 'shelf_life'
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / 'foodkeeper_mirror_manifest.json').read_text())
    snapshot = (root / 'foodkeeper_mirror_snapshot.json').read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != manifest['sha256']:
        raise SystemExit('FoodKeeper mirror snapshot hash does not match its manifest')
    (root / 'produce_storage.csv').write_text(export_csv())
    print(f"Exported {len(shelf_life_reference()['records'])} reviewed reference rows; mirror SHA-256 verified")
