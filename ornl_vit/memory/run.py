"""Checkpointed pushforward experiments and a directly supervised one-step diagnostic."""
import argparse
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import yaml

from baseline_vit.dataset import CacheIndex, training_statistics
from forecasting.checkpoint import check_cache
from forecasting.energy import energy_spectrum, plot_energy
from ..config import load_config
from ..dataset import HistoryDataset
from ..engine import evaluate, normalized_mse
from ..model import UPSTREAM_COMMIT
from ..rollout import forecast_five
from .model import CheckpointedMatey, pushforward_five

ARCHITECTURE = 'matey_checkpointed_pushforward5_v1'
ONE_STEP_ARCHITECTURE = 'matey_checkpointed_onestep_v1'
ROLLOUT_ARCHITECTURE = 'matey_checkpointed_rollout5_v1'


def is_rollout(cfg):
    return cfg['training']['objective'] == 'rollout_normalized_mse'


def checkpoint_architecture(cfg):
    return ROLLOUT_ARCHITECTURE if is_rollout(cfg) else ONE_STEP_ARCHITECTURE if is_one_step(cfg) else ARCHITECTURE


def is_one_step(cfg):
    return cfg['training']['objective'] == 'one_step_normalized_mse'


def selection_metric(report, cfg):
    return (report['horizons'][0]['normalized_mse'] if is_one_step(cfg)
            else report['mean_normalized_mse'])


def selection_label(cfg):
    return ('one-step validation normalized MSE' if is_one_step(cfg)
            else 'mean normalized MSE over five validation leads')


def load_experiment(path):
    path = Path(path).resolve()
    settings = yaml.safe_load(path.read_text())
    expected = {'base_config', 'activation_checkpointing', 'objective',
                'smoke_updates', 'wall_budget_seconds'}
    if set(settings) - {'epochs'} != expected or settings['activation_checkpointing'] is not True:
        raise ValueError('Expected the checkpointed pushforward experiment settings')
    if settings['objective'] not in ('pushforward_final_normalized_mse', 'one_step_normalized_mse', 'rollout_normalized_mse'):
        raise ValueError('Unknown training objective')
    if 'epochs' in settings and (type(settings['epochs']) is not int or settings['epochs'] < 1):
        raise ValueError('epochs must be a positive integer')
    for name in ('smoke_updates', 'wall_budget_seconds'):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f'Invalid {name}')
    cfg = load_config(path.parent / settings['base_config'])
    cfg['training']['objective'] = settings['objective']
    if 'epochs' in settings:
        cfg['training']['epochs'] = settings['epochs']
    if is_one_step(cfg):
        cfg['training']['horizon'] = 1
    return cfg, settings


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def memory(device):
    if device.type != 'cuda':
        return {}
    return dict(allocated=torch.cuda.memory_allocated(device),
                reserved=torch.cuda.memory_reserved(device),
                peak_allocated=torch.cuda.max_memory_allocated(device))


class Execution:
    def __init__(self, variant, device):
        self.variant, self.device = variant, device
        self.group = None
        self.rank, self.world = 0, 1
        if variant == 'ringx':
            if not dist.is_initialized():
                dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo',
                                        timeout=timedelta(minutes=15))
            self.group = dist.group.WORLD
            self.rank, self.world = dist.get_rank(), dist.get_world_size()
            if self.world < 2:
                raise ValueError('RingX requires at least two ranks')

    def model(self, cfg, dt, checkpointing=True):
        kwargs = dict(**cfg['model'], dt_reference=dt,
                      dt_rtol=cfg['data']['dt_rtol'], dt_atol=cfg['data']['dt_atol'],
                      activation_checkpointing=checkpointing)
        if self.variant == 'ringx':
            from .ringx_model import RingXMatey
            return RingXMatey(group=self.group, **kwargs).to(self.device)
        return CheckpointedMatey(**kwargs).to(self.device)

    def mean(self, value):
        value = torch.as_tensor(value, device=self.device, dtype=torch.float64).clone()
        if self.world > 1:
            dist.all_reduce(value, group=self.group); value /= self.world
        return value.item()

    def gradients(self, model):
        if self.world > 1:
            from .ringx_model import average_gradients
            average_gradients(model, self.group)

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    def collect(self, value):
        if self.world == 1:
            return [value]
        gathered = [None] * self.world
        dist.all_gather_object(gathered, value, group=self.group)
        return gathered


def local_batch(model, batch, execution):
    history, targets, times = [x.to(execution.device) for x in batch]
    if execution.world > 1:
        history, targets = model.partition(history), model.partition(targets)
    return history, targets, times


def update(model, batch, optimizer, stats, cfg, execution, trace=False):
    history, targets, times = local_batch(model, batch, execution)
    optimizer.zero_grad(set_to_none=True)
    stages = []
    def record(lead):
        stages.append(dict(lead=lead, **memory(execution.device)))
        print(json.dumps(dict(acceptance_memory=stages[-1])), flush=True)
    if is_rollout(cfg):
        prediction = forecast_five(model, history, times, record if trace else None)
        target = targets
    elif is_one_step(cfg):
        prediction = model(history, times[:, :4], times[:, 4])
        target = targets[:, :1]
        if trace: record(1)
    else:
        prediction = pushforward_five(model, history, times, record if trace else None)
        target = targets[:, -1:]
    loss = normalized_mse(prediction if is_rollout(cfg) else prediction[:, None], target, stats)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite training loss')
    loss.backward()
    execution.gradients(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'],
                                   error_if_nonfinite=True)
    optimizer.step()
    if trace:
        stages.append(dict(stage='after_backward_and_update', **memory(execution.device)))
    return execution.mean(loss.detach()), stages


class FullFieldView(torch.nn.Module):
    """Evaluation-only adapter; assemble fields before global metrics/spectra."""
    def __init__(self, model, execution):
        super().__init__(); self.model, self.execution = model, execution

    def forward(self, history, times, target):
        if torch.is_grad_enabled():
            raise RuntimeError('FullFieldView is evaluation-only')
        if self.execution.world == 1:
            return self.model(history, times, target)
        return self.model.assemble(self.model(self.model.partition(history), times, target))


def save_checkpoint(path, model, optimizer, cfg, stats, index, execution, epoch, validation, **extra):
    # Rank RNG and optimizer state permit inspection/recovery, although no automatic resume is implied.
    rng = execution.collect(dict(torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(execution.device) if execution.device.type == 'cuda' else None))
    if execution.rank == 0:
        saved = dict(format_version=1, architecture=checkpoint_architecture(cfg),
            variant=execution.variant, world_size=execution.world,
            model=model.state_dict(), optimizer=optimizer.state_dict(), rng_by_rank=rng,
            config=cfg, normalization=stats, dt_reference=index.dt_reference,
            cache_fingerprint=index.fingerprint, upstream_commit=UPSTREAM_COMMIT,
            epoch=epoch, validation=validation, **extra)
        path = Path(path); temporary = path.with_suffix('.pt.tmp')
        torch.save(saved, temporary); temporary.replace(path)
    execution.barrier()


def restore(path, execution):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('architecture') not in (ARCHITECTURE, ONE_STEP_ARCHITECTURE, ROLLOUT_ARCHITECTURE):
        raise ValueError('Not a memory-experiment checkpoint')
    if (saved['variant'], saved['world_size']) != (execution.variant, execution.world):
        raise ValueError('Upstream partitioned checkpoints require the original execution variant/world size')
    index = CacheIndex(saved['config']); check_cache(saved, index)
    model = execution.model(saved['config'], saved['dt_reference'])
    model.load_state_dict(saved['model']); model.eval()
    return model, saved, index


def optimizer_for(model, cfg):
    opt = cfg['training']
    return torch.optim.AdamW(model.parameters(), lr=opt['learning_rate'], weight_decay=opt['weight_decay'])


def smoke(cfg, settings, index, stats, output, execution):
    seed_all(cfg['seed'])
    model = execution.model(cfg, index.dt_reference).train()
    # Identical initialization, different stochastic-depth draws per spatial rank.
    seed_all(cfg['seed'] + execution.rank)
    optimizer = optimizer_for(model, cfg)
    batch = next(iter(DataLoader(HistoryDataset(index, 'train'), batch_size=cfg['training']['batch_size'])))
    if execution.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(execution.device)
    started = time.monotonic(); records = []
    for step in range(settings['smoke_updates']):
        loss, stages = update(model, batch, optimizer, stats, cfg, execution, trace=True)
        records.append(dict(update=step + 1, loss=loss, stages=stages))
    model.eval()
    local_history, _, times = local_batch(model, batch, execution)
    with torch.no_grad():
        expected = model(local_history, times[:, :4], times[:, 4]).cpu()
    checkpoint = output / 'acceptance.pt'
    save_checkpoint(checkpoint, model, optimizer, cfg, stats, index, execution, 0, None)
    del model, optimizer, local_history
    gc.collect()
    if execution.device.type == 'cuda': torch.cuda.empty_cache()
    restored, saved, _ = restore(checkpoint, execution)
    history, _, times = local_batch(restored, batch, execution)
    with torch.no_grad():
        actual = restored(history, times[:, :4], times[:, 4]).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    if execution.device.type == 'cuda': torch.cuda.synchronize(execution.device)
    validation_start = time.monotonic()
    validation = evaluate(FullFieldView(restored, execution),
        DataLoader(HistoryDataset(index, 'val', max_windows=1), batch_size=1), stats, execution.device)
    if execution.device.type == 'cuda': torch.cuda.synchronize(execution.device)
    validation_seconds = torch.tensor(time.monotonic() - validation_start, device=execution.device)
    if execution.world > 1: dist.all_reduce(validation_seconds, op=dist.ReduceOp.MAX)
    ranks = execution.collect(dict(rank=execution.rank, updates=records,
                                   memory=memory(execution.device)))
    if execution.rank == 0:
        write_json(output / 'acceptance.json', dict(passed=True, full_grid=cfg['model']['grid_size'],
            seconds=time.monotonic() - started, ranks=ranks, validation=validation,
            validation_seconds_per_window=validation_seconds.item(),
            checkpoint_reload=True, production_uses_fresh_initialization=True))
    del restored, saved
    gc.collect()
    if execution.device.type == 'cuda': torch.cuda.empty_cache()
    return validation_seconds.item()


@torch.no_grad()
def test_report(model, cfg, stats, index, output, execution):
    data = HistoryDataset(index, 'test', cross_file=True,
                          dt_rtol=cfg['data']['dt_rtol'], dt_atol=cfg['data']['dt_atol'])
    loader = DataLoader(data, batch_size=1)
    view = FullFieldView(model, execution).eval()
    actual_energy, predicted_energy, times_all = [], [], []
    spectrum_a = spectrum_p = None
    totals = np.zeros((5, 6), dtype=np.float64); count = 0
    std = np.asarray(stats['std']).reshape(1, 3, 1, 1, 1)
    for case, (history, targets, times) in enumerate(loader):
        prediction = forecast_five(view, history.to(execution.device), times.to(execution.device))[0].cpu().numpy()
        if execution.rank != 0:
            continue
        actual = targets[0].numpy().astype(np.float64)
        pred64 = prediction.astype(np.float64)
        initial = history[0, -1].numpy().astype(np.float64)
        error = pred64 - actual
        axes = (1, 2, 3, 4)
        totals += np.stack([np.sum(error**2, axis=axes), np.sum((error/std)**2, axis=axes),
            np.sum(np.abs(error), axis=axes), np.sum((initial[None]-actual)**2, axis=axes),
            np.sum(pred64**2, axis=axes), np.sum(actual**2, axis=axes)], axis=1)
        count += initial.size
        if case == 0:
            np.savez_compressed(output / 'first-forecast.npz', history=history[0].numpy(),
                                actual=targets[0].numpy(), predicted=prediction, times=times[0].numpy())
        truth = np.concatenate((history[0, -1:].numpy(), targets[0].numpy()))
        predicted = np.concatenate((history[0, -1:].numpy(), prediction))
        ae, pe, asp, psp = [], [], [], []
        for a, p in zip(truth, predicted):
            e, s = energy_spectrum(a); ae.append(e); asp.append(s)
            e, s = energy_spectrum(p); pe.append(e); psp.append(s)
        actual_energy.append(ae); predicted_energy.append(pe); times_all.append(times[0, 3:].numpy())
        if spectrum_a is None:
            spectrum_a, spectrum_p = np.zeros_like(asp), np.zeros_like(psp)
        spectrum_a += asp; spectrum_p += psp
    if execution.rank == 0:
        metrics = []
        for lead, (error, norm, absolute, persistence, predicted, actual) in enumerate(totals.tolist(), 1):
            metrics.append(dict(lead_steps=lead, physical_mse=error/count, normalized_mse=norm/count,
                physical_mae=absolute/count, relative_l2=float(np.sqrt(error/actual)) if actual else None,
                persistence_mse=persistence/count, mse_skill_vs_persistence=1-error/persistence if persistence else None,
                predicted_kinetic_energy=1.5*predicted/count, target_kinetic_energy=1.5*actual/count))
        report = dict(horizon=5, samples=len(data), horizons=metrics,
                      mean_normalized_mse=float(totals[:, 1].mean()/count))
        arrays = dict(lead_steps=np.arange(6), actual_energy=np.asarray(actual_energy),
            predicted_energy=np.asarray(predicted_energy), times=np.asarray(times_all),
            actual_spectrum=spectrum_a / len(data), predicted_spectrum=spectrum_p / len(data),
            modes=np.arange(spectrum_a.shape[-1]), grid_side=np.asarray(cfg['model']['grid_size']))
        np.savez_compressed(output / 'energy-data.npz', **arrays)
        objective = 'rollout trained' if is_rollout(cfg) else 'one-step trained' if is_one_step(cfg) else 'pushforward'
        plot_energy(dict(model=f'MATEY {execution.variant} {objective}'), arrays, output)
        write_json(output / 'test.json', dict(variant=execution.variant, world_size=execution.world,
            training_objective=cfg['training']['objective'],
            rollout=report, starts=data.continuous.starts(),
            energy_definition='K=0.5*mean(u²+v²+w²), including mean flow',
            spectrum_definition='Integer Fourier-mode shells; see forecasting/energy.py'))
    execution.barrier()


def train(cfg, settings, index, stats, output, execution, deadline=None):
    seed_all(cfg['seed']); model = execution.model(cfg, index.dt_reference)
    seed_all(cfg['seed'] + execution.rank)
    optimizer = optimizer_for(model, cfg)
    opt = cfg['training']
    training = HistoryDataset(index, 'train', max_windows=opt['max_train_windows'])
    validation = HistoryDataset(index, 'val')
    loader = DataLoader(training, batch_size=opt['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(cfg['seed']), num_workers=opt['num_workers'])
    val_loader = DataLoader(validation, batch_size=opt['batch_size'], num_workers=opt['num_workers'])
    began = time.monotonic(); best = float('inf'); updates = 0; stopped = False
    next_save = 600.
    for epoch in range(1, opt['epochs'] + 1):
        model.train(); loss_sum = 0.; count = 0; epoch_start = time.monotonic()
        if execution.device.type == 'cuda': torch.cuda.reset_peak_memory_stats(execution.device)
        for batch in loader:
            loss, _ = update(model, batch, optimizer, stats, cfg, execution)
            loss_sum += loss; count += 1; updates += 1
            # Synchronized stop preserves collective ordering; leave time for evaluation.
            elapsed = torch.tensor(time.monotonic() - began, device=execution.device)
            if execution.world > 1: dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            if elapsed.item() >= next_save:
                save_checkpoint(output / 'last.pt', model, optimizer, cfg, stats, index, execution,
                    epoch, None, optimizer_updates=updates, partial_epoch=True)
                next_save = elapsed.item() + 600.
            stop = elapsed.item() >= settings['wall_budget_seconds'] or (deadline is not None and time.monotonic() >= deadline)
            stop_tensor = torch.tensor(int(stop), device=execution.device)
            if execution.world > 1: dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            if stop_tensor.item():
                stopped = True; break
        validation_report = evaluate(FullFieldView(model, execution), val_loader, stats, execution.device)
        selection = selection_metric(validation_report, cfg)
        rank_memory = execution.collect(memory(execution.device))
        record = dict(epoch=epoch, partial_epoch=stopped and count < len(loader), optimizer_updates=updates,
            validation=validation_report,
            objective=opt['objective'], supervised_lead=None if is_rollout(cfg) else 1 if is_one_step(cfg) else 5,
            supervised_leads=list(range(1,6)) if is_rollout(cfg) else [1 if is_one_step(cfg) else 5],
            seconds=time.monotonic() - epoch_start, memory_by_rank=rank_memory)
        record['training_loss_rollout' if is_rollout(cfg) else 'training_loss_one_step' if is_one_step(cfg) else 'training_loss_final_lead'] = loss_sum / count
        if execution.rank == 0:
            with (output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            print(json.dumps(record, allow_nan=False), flush=True)
        for name in (['best.pt'] if selection < best else []) + ['last.pt']:
            save_checkpoint(output / name, model, optimizer, cfg, stats, index, execution,
                epoch, validation_report, optimizer_updates=updates, partial_epoch=record['partial_epoch'],
                selection=selection_label(cfg))
        best = min(best, selection)
        if stopped: break
    del model, optimizer
    gc.collect()
    if execution.device.type == 'cuda': torch.cuda.empty_cache()
    model, saved, index = restore(output / 'best.pt', execution)
    test_report(model, cfg, stats, index, output, execution)
    if execution.rank == 0:
        write_json(output / 'complete.json', dict(epochs=epoch, optimizer_updates=updates,
            stopped_for_time=stopped, selected_epoch=saved['epoch'], best_validation=best))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('experiment.yaml')))
    parser.add_argument('--variant', choices=['baseline', 'ringx'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--report', type=Path, help='Home directory for plots, metrics and weight references')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    began = time.monotonic()
    device = torch.device(args.device, int(os.environ.get('LOCAL_RANK', 0))) if args.device == 'cuda' else torch.device('cpu')
    if device.type == 'cuda': torch.cuda.set_device(device)
    execution = Execution(args.variant, device)
    try:
        cfg, settings = load_experiment(args.config)
        cfg['training'].update(output_dir=str(args.output.resolve()), device=args.device)
        if execution.rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / 'config.json', cfg)
            import importlib.metadata
            write_json(args.output / 'execution.json', dict(variant=args.variant, world_size=execution.world,
                settings=settings, torch=str(torch.__version__), upstream_commit=UPSTREAM_COMMIT,
                flash_attn=importlib.metadata.version('flash-attn') if args.variant == 'ringx' else None,
                hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                loss='normalized MSE averaged over all five leads, full recursive gradients' if is_rollout(cfg) else 'normalized MSE on lead one only' if is_one_step(cfg) else 'normalized MSE on lead five only',
                selection=selection_label(cfg),
                sampling='original four-history/five-target windows; unused targets retained for evaluation',
                partition_semantics='upstream local normalization/boundaries' if execution.world > 1 else 'full domain'))
        execution.barrier()
        index = CacheIndex(cfg)
        stats = training_statistics(index) if execution.rank == 0 else None
        if execution.world > 1:
            objects = [stats]; dist.broadcast_object_list(objects, src=0); stats = objects[0]
        if execution.rank == 0:
            write_json(args.output / 'normalization.json', stats)
            write_json(args.output / 'data.json', dict(cache_fingerprint=index.fingerprint,
                train_files=len(cfg['data']['train']), train_windows=len(HistoryDataset(index, 'train')),
                val_windows=len(HistoryDataset(index, 'val')),
                test_windows=len(HistoryDataset(index, 'test', cross_file=True))))
        if args.variant == 'ringx':
            from .test_distributed import kernel_check
            kernel_check(device)
        evaluation_seconds = smoke(cfg, settings, index, stats, args.output, execution)
        if not args.smoke_only:
            # Reserve measured evaluation time with margin, at least one hour.
            allocation = int(os.environ.get('MATEY_JOB_SECONDS', '32400'))
            evaluation_windows = len(HistoryDataset(index, 'val')) + len(HistoryDataset(index, 'test', cross_file=True))
            reserve = max(900. if is_one_step(cfg) else 3600., 1.5 * evaluation_seconds * evaluation_windows + 120.)
            deadline = began + allocation - reserve
            if execution.rank == 0:
                write_json(args.output / 'time-budget.json', dict(allocation_seconds=allocation,
                    evaluation_reserve_seconds=reserve, remaining_training_seconds=deadline-time.monotonic(),
                    estimate_only=True))
            if deadline <= time.monotonic():
                raise RuntimeError('Acceptance timing leaves no training budget; inspect time-budget.json')
            train(cfg, settings, index, stats, args.output, execution, deadline=deadline)
    except Exception as exc:
        if args.output.is_dir():
            write_json(args.output / f'failure-rank{execution.rank}.json',
                       dict(error=repr(exc), memory=memory(device)))
        raise
    finally:
        if execution.rank == 0 and args.report is not None and args.output.is_dir():
            from .publish import publish
            publish(args.output, args.report)
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__':
    main()
