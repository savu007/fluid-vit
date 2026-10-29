"""Lock-free finalization of immutable, completed ordinary-model evaluations."""
import csv
import io
import json
import os
from pathlib import Path
import tempfile


def atomic_text(path, text):
    path = Path(path)
    # Unique temporary names allow concurrent writers; rename publishes a whole file.
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def summarize(root):
    root = Path(root)
    paths = {name: root / name / 'complete.json' for name in ('matey', 'yalamanchi')}
    if not all(path.exists() for path in paths.values()):
        return False
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
    if reports['matey']['test']['test_starts'] != reports['yalamanchi']['test']['test_starts']:
        raise ValueError('Different evaluation origins')
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(['model', 'lead', 'normalized_mse', 'relative_l2', 'predicted_K', 'actual_K'])
    for name, report in reports.items():
        for h in report['test']['horizons']:
            writer.writerow([name, h['lead_steps'], h['normalized_mse'], h['relative_l2'],
                             h['predicted_kinetic_energy'], h['target_kinetic_energy']])
    # Both writers consume identical immutable complete.json files and produce
    # identical content. Each output is atomic; the pair is not a transaction.
    atomic_text(root / 'comparison.json', json.dumps(reports, indent=2, allow_nan=False))
    atomic_text(root / 'comparison.csv', stream.getvalue())
    return True
