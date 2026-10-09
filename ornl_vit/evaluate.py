"""Evaluate five future steps on checked cross-file held-out histories."""
import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader
from baseline_vit.engine import select_device
from .dataset import HistoryDataset
from .engine import evaluate, restore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output', required=True, help='New output directory')
    args = parser.parse_args()
    device = select_device(args.device)
    model, saved, index = restore(args.checkpoint, device)
    cfg = saved['config']
    dataset = HistoryDataset(index, args.split, cross_file=True,
                             dt_rtol=cfg['data']['dt_rtol'], dt_atol=cfg['data']['dt_atol'])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    report = evaluate(model, DataLoader(dataset, batch_size=1, num_workers=0),
                      saved['normalization'], device, save_first=output / 'first-forecast.npz')
    result = dict(checkpoint=str(Path(args.checkpoint).resolve()), architecture=saved['architecture'],
        checkpoint_epoch=saved['epoch'], split=args.split, history=4,
        cache_fingerprint=index.fingerprint, starts=dataset.continuous.starts(), rollout=report)
    (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k,v in result.items() if k != 'starts'}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
