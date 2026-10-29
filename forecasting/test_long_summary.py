import json
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from .long_summary import summarize


class SummaryTests(unittest.TestCase):
    def test_missing_then_concurrent_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(summarize(root))
            report = {'test': {'test_starts': [1, 2], 'horizons': [dict(
                lead_steps=1, normalized_mse=.1, relative_l2=.2,
                predicted_kinetic_energy=2., target_kinetic_energy=3.)]}}
            for name in ('matey', 'yalamanchi'):
                (root/name).mkdir()
                (root/name/'complete.json').write_text(json.dumps(report))
            with ThreadPoolExecutor(max_workers=8) as pool:
                self.assertTrue(all(pool.map(summarize, [root]*32)))
            self.assertEqual(set(json.loads((root/'comparison.json').read_text())), {'matey', 'yalamanchi'})
            self.assertEqual(len((root/'comparison.csv').read_text().splitlines()), 3)
            self.assertFalse(list(root.glob('.*')))
            report['test']['test_starts'] = [3]
            (root/'matey'/'complete.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'origins'): summarize(root)

if __name__ == '__main__': unittest.main()
