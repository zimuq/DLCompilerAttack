"""
Data-level contribution analysis for DcL-BD — Phase 0 + Phase 1.

Independent line of investigation from the channel-level smoke test. This
script is a static, single-checkpoint diagnostic. It does NOT retrain.

Phase 0 (sanity gate)
---------------------
Reload best.tar, rebuild the compiled ensemble via the existing
compile_ensemble_model pipeline, run evaluate_model on a small slice
of the test loader, hard-gate on:
  ASR active     :  acc_bd_C_bd >= 0.8
  Stealth intact :  acc_bd_D_bd <= 0.2
  Clean OK       :  acc_cl_D_cl >= 0.6 and acc_cl_C_cl >= 0.6
On failure, exit non-zero unless --force.

Phase 1 (per-sample gradient norms)
-----------------------------------
For every (x_i, y_i) in the train_loader (batch_size=1), compute
  g_i^bd    = || d/dtheta_tuned  ( CE(tuned_model(act(C(t_i)[1])), y_target) * bd_rate ) ||_2
  g_i^clean = || d/dtheta_tuned  ( CE(tuned_model(act(D(x_i))),    y_i)            ) ||_2

where C is the compiled ensemble (forward only, no_grad) and D is the
pre-compile feature extractor. Persist raw CSVs, summary JSON, three plots,
and a strictly descriptive (non-interpretive) text summary.

Reuses existing repo utils: load_model, load_dataloader, load_attack_model_cls,
load_DLCL, init_bd_trigger, TargetDevice, CLSetting, evaluate_model,
compile_ensemble_model. Does NOT modify any code in src/attack/.

Usage:
    python experiments/data_contribution/phase01.py \
        --best_path work_dir/.../best.tar \
        --task_id 0 --cl_id 0 --hardware_id 0 \
        --device cuda --num_sanity_batches 10 \
        --out_dir data_contribution_results/task0_inductor

Optional:
    --max_samples N     subsample the train_loader (default: all 50000)
    --force             ignore Phase 0 sanity gate failure
    --phase {0,1,both}  run only one phase (default: both)
    --bd_rate FLOAT     scale on loss_bd_C_bd (default 1.0; matches main.py)
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

# Make repo modules importable when this script is invoked from anywhere.
_THIS = os.path.abspath(__file__)
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from utils import load_dataloader  # noqa: E402
from src import TargetDevice  # noqa: E402
from src.attack import load_DLCL  # noqa: E402
from src.attack.utils import (  # noqa: E402
    CLSetting,
    compile_ensemble_model,
    evaluate_model,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--best_path', required=True,
                   help='Path to Stage-2 best.tar.')
    p.add_argument('--task_id', type=int, default=0)
    p.add_argument('--cl_id', type=int, default=0,
                   help='0=torch.compile, 1=TVM, 2=ONNXRuntime')
    p.add_argument('--hardware_id', type=int, default=0,
                   help='0=GPU, -1=CPU')
    p.add_argument('--device', default='cuda',
                   help='torch device for grad / non-compiled ops')
    p.add_argument('--num_sanity_batches', type=int, default=10)
    p.add_argument('--out_dir', default='data_contribution_results/task0_inductor')
    p.add_argument('--bd_rate', type=float, default=1.0,
                   help='Scale on loss_bd_C_bd. main.py sets 1.0 for task0.')
    p.add_argument('--max_samples', type=int, default=None,
                   help='Optional cap on per-sample loop length.')
    p.add_argument('--force', action='store_true',
                   help='Continue past a Phase 0 sanity gate failure.')
    p.add_argument('--phase', choices=['0', '1', 'both'], default='both')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def move_bd_trigger_to(bd_trigger, device):
    """ImgBackDoorTrigger is not an nn.Module; move its tensors explicitly.

    Per the brief, after torch.load we must move trigger / min_pixel /
    max_pixel / device. ori_trigger is also moved for safety (used by
    normalize_trigger). The trigger Parameter's data is updated in-place
    so existing references stay valid.
    """
    bd_trigger.device = device
    bd_trigger.trigger.data = bd_trigger.trigger.data.to(device)
    bd_trigger.ori_trigger = bd_trigger.ori_trigger.to(device)
    bd_trigger.min_pixel = bd_trigger.min_pixel.to(device)
    bd_trigger.max_pixel = bd_trigger.max_pixel.to(device)
    return bd_trigger


class _SubsetLoader:
    """Take only the first n batches of a DataLoader (no copy of dataset)."""
    def __init__(self, src, n):
        self.src = src
        self.n = n
        self.batch_size = src.batch_size
        self.dataset = src.dataset

    def __iter__(self):
        it = iter(self.src)
        for _ in range(self.n):
            try:
                yield next(it)
            except StopIteration:
                return

    def __len__(self):
        return min(self.n, len(self.src))


def build_cl_setting(save_model, batch_size, work_dir, hardware_target,
                     cl_func, device):
    return CLSetting.from_config({
        'batch_size': batch_size,
        'input_sizes': save_model.input_sizes,
        'input_types': save_model.input_types,
        'work_dir': work_dir,
        'hardware_target': hardware_target,
        'cl_func': cl_func,
        'fp': save_model.fp,
        'device': device,
    })


def grad_norm_l2(module):
    """L2 norm of concatenated .grad over all parameters of `module`."""
    sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            sq += p.grad.detach().pow(2).sum().item()
    return sq ** 0.5


# ---------------------------------------------------------------------------
# Phase 0
# ---------------------------------------------------------------------------

def phase0_sanity(args, save_model, bd_trigger, test_loader, device,
                  cl_func, hardware_target):
    print('\n=== Phase 0: Sanity gate ===')

    cl_setting = build_cl_setting(
        save_model, batch_size=test_loader.batch_size,
        work_dir=args.out_dir, hardware_target=hardware_target,
        cl_func=cl_func, device=device,
    )
    sub_loader = _SubsetLoader(test_loader, args.num_sanity_batches)

    save_model = save_model.to(device).eval()
    print(f'  evaluating over {len(sub_loader)} batches '
          f'(batch_size={test_loader.batch_size}) ...')
    acc = evaluate_model(save_model, cl_setting, sub_loader, bd_trigger)
    acc_cl_D_cl, acc_cl_C_cl, acc_bd_D_cl, acc_bd_D_bd, acc_bd_C_bd = (
        float(a) for a in acc
    )

    print(f'  acc_cl_D_cl  = {acc_cl_D_cl:.4f}   (clean acc, PyTorch path)')
    print(f'  acc_cl_C_cl  = {acc_cl_C_cl:.4f}   (clean acc, compiled path)')
    print(f'  acc_bd_D_cl  = {acc_bd_D_cl:.4f}   (triggered, PyTorch -> true label;'
          ' should be HIGH)')
    print(f'  acc_bd_D_bd  = {acc_bd_D_bd:.4f}   (triggered, PyTorch -> target label;'
          ' should be LOW)')
    print(f'  acc_bd_C_bd  = {acc_bd_C_bd:.4f}   (triggered, compiled -> target label;'
          ' the ASR)')

    checks = {
        'asr_active':      acc_bd_C_bd >= 0.8,
        'stealth_intact':  acc_bd_D_bd <= 0.2,
        'clean_preserved': acc_cl_D_cl >= 0.6 and acc_cl_C_cl >= 0.6,
    }
    sanity_passed = all(checks.values())

    payload = {
        'best_path': args.best_path,
        'task_id': args.task_id,
        'cl_id': args.cl_id,
        'hardware_id': args.hardware_id,
        'num_sanity_batches': args.num_sanity_batches,
        'metrics': {
            'acc_cl_D_cl': acc_cl_D_cl,
            'acc_cl_C_cl': acc_cl_C_cl,
            'acc_bd_D_cl': acc_bd_D_cl,
            'acc_bd_D_bd': acc_bd_D_bd,
            'acc_bd_C_bd': acc_bd_C_bd,
        },
        'sanity_passed': sanity_passed,
        'checks': checks,
    }
    out_path = os.path.join(args.out_dir, 'phase0_sanity.json')
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'  saved {out_path}')

    if not sanity_passed:
        for ck, ok in checks.items():
            if not ok:
                print(f'  FAILED check: {ck}')
        if not args.force:
            print('Sanity gate FAILED. Exiting non-zero (use --force to continue).')
            sys.exit(1)
        print('  --force given: continuing despite failure.')

    return payload


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------

def _summary_stats(values, labels):
    p99 = float(np.percentile(values, 99))
    return {
        'num_samples': int(len(values)),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'percentiles': {
            'p1':  float(np.percentile(values, 1)),
            'p5':  float(np.percentile(values, 5)),
            'p25': float(np.percentile(values, 25)),
            'p50': float(np.percentile(values, 50)),
            'p75': float(np.percentile(values, 75)),
            'p95': float(np.percentile(values, 95)),
            'p99': p99,
        },
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'fraction_below_1e-6': float(np.mean(values < 1e-6)),
        'fraction_above_p99_times_10': float(np.mean(values > p99 * 10)),
        'per_class_mean': {
            str(int(c)): float(values[labels == c].mean())
            for c in sorted(np.unique(labels).tolist())
        },
        'per_class_std': {
            str(int(c)): float(values[labels == c].std())
            for c in sorted(np.unique(labels).tolist())
        },
    }


def _save_plots(out_dir, args, grad_bd, grad_clean, labels):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib not available; skipping plots')
        return

    n = len(grad_bd)

    # 1) histogram (log y)
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(grad_bd, bins=80, log=True,
                color='steelblue', edgecolor='black', linewidth=0.5)
        ax.set_xlabel('grad_norm (loss_bd_C_bd, ||grad||_2)')
        ax.set_ylabel('count (log scale)')
        ax.set_title(f'task_id={args.task_id}, cl_id={args.cl_id}, n={n}')
        plt.tight_layout()
        path = os.path.join(out_dir, 'histogram.png')
        plt.savefig(path, dpi=130)
        plt.close(fig)
        print(f'  saved {path}')
    except Exception as e:
        print(f'  histogram plot failed: {type(e).__name__}: {e}')

    # 2) per-class overlaid histograms (log y)
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        for c in sorted(np.unique(labels).tolist()):
            mask = labels == c
            if mask.sum() > 0:
                ax.hist(grad_bd[mask], bins=60, log=True, alpha=0.45,
                        label=f'class {int(c)}')
        ax.set_xlabel('grad_norm (loss_bd_C_bd)')
        ax.set_ylabel('count (log scale)')
        ax.set_title(f'task_id={args.task_id}, cl_id={args.cl_id} '
                     '(per source class)')
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        path = os.path.join(out_dir, 'per_class_histogram.png')
        plt.savefig(path, dpi=130)
        plt.close(fig)
        print(f'  saved {path}')
    except Exception as e:
        print(f'  per-class histogram plot failed: {type(e).__name__}: {e}')

    # 3) scatter clean vs bd
    try:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(grad_clean, grad_bd, s=2, alpha=0.3, c='tab:blue')
        lo = float(min(grad_clean.min(), grad_bd.min()))
        hi = float(max(grad_clean.max(), grad_bd.max()))
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, lw=1)
        try:
            ax.set_xscale('symlog', linthresh=1e-6)
            ax.set_yscale('symlog', linthresh=1e-6)
        except Exception:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlabel('||grad L_cl_D_cl||_2 (clean)')
        ax.set_ylabel('||grad L_bd_C_bd||_2 (triggered, compiled)')
        ax.set_title(f'Clean vs backdoor per-sample gradient norm (n={n})')
        plt.tight_layout()
        path = os.path.join(out_dir, 'scatter_clean_vs_bd.png')
        plt.savefig(path, dpi=130)
        plt.close(fig)
        print(f'  saved {path}')
    except Exception as e:
        print(f'  scatter plot failed: {type(e).__name__}: {e}')


def _write_summary_text(out_dir, args, grad_bd, grad_clean, labels,
                        bd_rate, target_class, summary):
    try:
        from scipy.stats import pearsonr, spearmanr
        pr, _ = pearsonr(grad_clean, grad_bd)
        sr, _ = spearmanr(grad_clean, grad_bd)
        pearson_r = float(pr)
        spearman_r = float(sr)
    except Exception as e:
        print(f'  correlation calc failed: {e}; reporting NaN')
        pearson_r = float('nan')
        spearman_r = float('nan')

    target_mean = (
        float(grad_bd[labels == target_class].mean())
        if (labels == target_class).any() else float('nan')
    )
    nontarget_mean = (
        float(grad_bd[labels != target_class].mean())
        if (labels != target_class).any() else float('nan')
    )

    rows = []
    for c in sorted(np.unique(labels).tolist()):
        m = grad_bd[labels == c]
        rows.append(f'    class {int(c)}: mean={m.mean():.6g}, '
                    f'std={m.std():.6g}, n={len(m)}')
    per_class_table = '\n'.join(rows)

    text = (
        'Phase 1: Static per-sample gradient norm distribution\n'
        '======================================================\n\n'
        f'Checkpoint: {args.best_path}\n'
        f'Task: {args.task_id}, Compiler: {args.cl_id}, '
        f'Hardware: {args.hardware_id}\n'
        f'bd_rate used: {bd_rate}\n'
        f'Number of samples processed: {len(grad_bd)}\n\n'
        'Distribution statistics\n'
        '-----------------------\n'
        f'Mean:     {summary["mean"]:.6g}\n'
        f'Median:   {summary["median"]:.6g}\n'
        f'Std:      {summary["std"]:.6g}\n'
        f'P1 - P99: {summary["percentiles"]["p1"]:.6g} ... '
        f'{summary["percentiles"]["p99"]:.6g}\n'
        f'Min:      {summary["min"]:.6g}\n'
        f'Max:      {summary["max"]:.6g}\n\n'
        'Saturation diagnostic\n'
        '---------------------\n'
        f'Fraction of samples with grad_norm < 1e-6: '
        f'{summary["fraction_below_1e-6"]:.4f}\n'
        f'Fraction of samples with grad_norm > 10 * p99: '
        f'{summary["fraction_above_p99_times_10"]:.4f}\n\n'
        'Per-class breakdown\n'
        '-------------------\n'
        f'Target class (label {target_class}) mean:    {target_mean:.6g}\n'
        f'Non-target classes mean:                  {nontarget_mean:.6g}\n'
        'Per-class:\n'
        f'{per_class_table}\n\n'
        'Correlation with clean gradient norms\n'
        '-------------------------------------\n'
        f'Pearson r:   {pearson_r:.6g}\n'
        f'Spearman r:  {spearman_r:.6g}\n\n'
        'Notes\n'
        '-----\n'
        'This is a static, single-checkpoint diagnostic. The values '
        'reported here\nare NOT per-sample contributions to the trained '
        'backdoor. They are the\nremaining loss signal magnitude at the '
        'final checkpoint.\n\n'
        'Interpretation of the distribution shape is deferred to '
        'follow-up\nanalysis. This script produces only descriptive '
        'statistics.\n'
    )
    path = os.path.join(out_dir, 'phase1_summary.txt')
    with open(path, 'w') as f:
        f.write(text)
    print(f'  saved {path}')


def phase1_grad_norms(args, save_model, bd_trigger, train_loader_b1, device,
                      cl_func, hardware_target):
    print('\n=== Phase 1: per-sample gradient norms ===')

    # Verified MyModel attribute names: m_1 (D), act, m_2 (tuned_model).
    D = save_model.m_1
    act = save_model.act
    tuned_model = save_model.m_2

    # Freeze D and bd_trigger; act has no learnable params (uses
    # register_buffer per ChannelWiseThresholdActivation).
    D.eval().to(device)
    for p in D.parameters():
        p.requires_grad = False
    act = act.to(device)
    bd_trigger.trigger.requires_grad = False

    tuned_model.to(device).train()
    for p in tuned_model.parameters():
        p.requires_grad = True

    cl_setting = build_cl_setting(
        save_model, batch_size=1,
        work_dir=args.out_dir, hardware_target=hardware_target,
        cl_func=cl_func, device=device,
    )
    print('  building compiled ensemble at batch_size=1 (first call traces)...')
    compiled_model = compile_ensemble_model(D, act, tuned_model, cl_setting)

    cross_entropy = nn.CrossEntropyLoss(reduction='mean')
    target_label = int(bd_trigger.target_label)
    bd_rate = float(args.bd_rate)
    print(f'  bd_rate = {bd_rate}, target_label = {target_label}')

    n_total_dataset = len(train_loader_b1.dataset)
    n_total = (min(n_total_dataset, args.max_samples)
               if args.max_samples is not None else n_total_dataset)
    print(f'  iterating {n_total} samples '
          f'(of {n_total_dataset}); --max_samples = {args.max_samples}')

    grad_norms_bd = []
    grad_norms_clean = []
    labels_list = []

    pbar = tqdm(total=n_total, desc='per-sample grad', unit='sample')
    sample_idx = 0
    speed_warn_emitted = False
    t_first_sample = None

    for batch in train_loader_b1:
        if sample_idx >= n_total:
            break
        x = batch['input'].to(device)
        y = batch['label'].to(device)
        if x.shape[0] != 1:
            raise RuntimeError(f'expected batch_size=1, got {x.shape[0]}')

        # ----- backdoor gradient -----
        t_x = bd_trigger.add_trigger(x)
        t_y = torch.tensor([target_label], device=device, dtype=y.dtype)

        with torch.no_grad():
            out_compiled = compiled_model.forward([t_x])
        m1_out_compiled = out_compiled[1].to(device)
        # act has no learnable params; gradients flow only through tuned_model.
        C_bd_embed = act(m1_out_compiled)

        tuned_model.zero_grad(set_to_none=True)
        C_bd_logit = tuned_model(C_bd_embed)
        loss_bd_C_bd = cross_entropy(C_bd_logit, t_y) * bd_rate
        loss_bd_C_bd.backward()
        grad_norms_bd.append(grad_norm_l2(tuned_model))

        # ----- clean gradient -----
        with torch.no_grad():
            m1_clean = D(x)
        cl_embed = act(m1_clean)

        tuned_model.zero_grad(set_to_none=True)
        cl_logit = tuned_model(cl_embed)
        loss_cl_D_cl = cross_entropy(cl_logit, y)
        loss_cl_D_cl.backward()
        grad_norms_clean.append(grad_norm_l2(tuned_model))

        labels_list.append(int(y.item()))
        sample_idx += 1
        pbar.update(1)

        if t_first_sample is None:
            t_first_sample = time.time()
        elif sample_idx == 6 and not speed_warn_emitted:
            elapsed = time.time() - t_first_sample
            per_sample_ms = (elapsed / 5) * 1000.0
            if per_sample_ms > 200:
                print(f'\n  NOTE: ~{per_sample_ms:.0f} ms/sample. '
                      f'Full {n_total} ~ {per_sample_ms * n_total / 60000:.1f} min. '
                      f'Consider --max_samples 10000 if too slow.')
            speed_warn_emitted = True

    pbar.close()
    tuned_model.zero_grad(set_to_none=True)

    grad_bd = np.asarray(grad_norms_bd, dtype=np.float64)
    grad_clean = np.asarray(grad_norms_clean, dtype=np.float64)
    labels = np.asarray(labels_list, dtype=np.int64)
    sample_idxs = np.arange(len(grad_bd), dtype=np.int64)

    df_bd = pd.DataFrame({
        'sample_index': sample_idxs,
        'class_label': labels,
        'grad_norm': grad_bd,
    })
    df_bd_path = os.path.join(args.out_dir, 'gradient_norms.csv')
    df_bd.to_csv(df_bd_path, index=False)
    print(f'  saved {df_bd_path}')

    df_clean = pd.DataFrame({
        'sample_index': sample_idxs,
        'class_label': labels,
        'grad_norm': grad_clean,
    })
    df_clean_path = os.path.join(args.out_dir, 'gradient_norms_clean.csv')
    df_clean.to_csv(df_clean_path, index=False)
    print(f'  saved {df_clean_path}')

    summary = _summary_stats(grad_bd, labels)
    summary_path = os.path.join(args.out_dir, 'gradient_norms_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  saved {summary_path}')

    _save_plots(args.out_dir, args, grad_bd, grad_clean, labels)
    _write_summary_text(
        args.out_dir, args, grad_bd, grad_clean, labels,
        bd_rate, target_label, summary,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'[phase01] device={device}, best_path={args.best_path}')
    print(f'[phase01] task_id={args.task_id}, cl_id={args.cl_id}, '
          f'hardware_id={args.hardware_id}, out_dir={args.out_dir}')

    # Load checkpoint.
    print('[phase01] loading best.tar ...')
    bd_trigger, save_model, _saved_acc = torch.load(
        args.best_path, weights_only=False, map_location=device,
    )
    save_model = save_model.to(device)
    bd_trigger = move_bd_trigger_to(bd_trigger, device)

    cl_func = load_DLCL(args.cl_id)
    hardware_target = TargetDevice(args.hardware_id)

    if args.phase in ('0', 'both'):
        # main.py uses batch_size=100 for task0; matches what evaluate_model
        # was tuned against during training.
        _, _, test_loader = load_dataloader(
            args.task_id, is_shuffle=False,
            train_batch=100, test_batch=100,
        )
        phase0_sanity(args, save_model, bd_trigger, test_loader, device,
                      cl_func, hardware_target)

    if args.phase == '0':
        return

    print('[phase01] loading train_loader at batch_size=1 ...')
    train_loader_b1, _, _ = load_dataloader(
        args.task_id, is_shuffle=False,
        train_batch=1, test_batch=100,
    )
    phase1_grad_norms(args, save_model, bd_trigger, train_loader_b1, device,
                      cl_func, hardware_target)

    print('[phase01] done.')


if __name__ == '__main__':
    main()
