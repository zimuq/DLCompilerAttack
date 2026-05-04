"""
Phase 2: Counterfactual verification of data-level attribution failure.

For each masking variant, run a fresh Stage 2 retrain from the Stage-1
checkpoint with the specified samples masked out of either:
  - loss_bd_C_bd only (surgical) — isolates the backdoor-injection
    counterfactual without confounding clean accuracy
  - all three loss terms (naive)  — for comparison; clean acc may degrade

Then evaluate ASR and the other 4 metrics on the test set with the
existing evaluate_model utility.

Reuses the existing Stage 2 forward pipeline (D / act / compiled / tuned_model)
verbatim — only the loss aggregation is modified. Does not touch
src/attack/finetune.py.

Sample identity is tracked via Option A: a sample_index column added to
the HF dataset AFTER the seed=33 shuffle, so sample_index = post-shuffle
iteration position (matches Phase 1's gradient_norms.csv ordering).

Usage:
    python experiments/data_contribution/phase02_counterfactual.py \
        --task_id 0 --cl_id 0 --hardware_id 0 \
        --device cuda \
        --grad_norms_csv data_contribution_results/task0_inductor/gradient_norms.csv \
        --step1_path 'work_dir/.../<task_name>.step1' \
        --out_dir data_contribution_results/task0_inductor/phase02

Optional:
    --variants v1,v2,...    or "all"; omit for default 5-variant subset
    --seed 42               for random-mask variants
    --dry_run               build masks, run one forward per variant, exit
    --force                 re-run variants whose final_metrics.json exists
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make repo modules importable.
_THIS = os.path.abspath(__file__)
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from utils import (  # noqa: E402
    DATASET,
    _MEAN_,
    _STD_,
    load_dataloader,
)
from src import TargetDevice  # noqa: E402
from src.attack import load_DLCL  # noqa: E402
from src.attack.utils import (  # noqa: E402
    CLSetting,
    compile_ensemble_model,
    evaluate_model,
)
from src.model import MyModel  # noqa: E402


ALL_VARIANTS = [
    'baseline',
    'mask_top1pct_grad',
    'mask_top5pct_grad',
    'mask_top10pct_grad',
    'mask_top20pct_grad',
    'mask_top50pct_grad',
    'mask_random_10pct',
    'mask_random_50pct',
    'mask_class0',
    'mask_class1',
    'naive_remove_top10pct',
    'naive_remove_random_10pct',
]
DEFAULT_VARIANTS = [
    'baseline',
    'mask_top10pct_grad',
    'mask_random_10pct',
    'mask_class0',
    'mask_top50pct_grad',
]
NUM_TRAIN_CIFAR10 = 50000

# Mirrors main.py task0 attack_config exactly.
ATTACK_CONFIG = {
    'finetune_epoch': 50,
    'finetune_lr': 1e-4,
    'bd_rate': 1.0,
    'save_freq': 10,
    'batch_size': 100,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task_id', type=int, default=0)
    p.add_argument('--cl_id', type=int, default=0,
                   help='0=torch.compile, 1=TVM, 2=ONNXRuntime')
    p.add_argument('--hardware_id', type=int, default=0,
                   help='0=GPU, -1=CPU')
    p.add_argument('--device', default='cuda')
    p.add_argument('--grad_norms_csv', required=True,
                   help='Phase 1 output: gradient_norms.csv')
    p.add_argument('--step1_path', required=True,
                   help='Stage-1 checkpoint: <task_name>.step1')
    p.add_argument('--out_dir', required=True,
                   help='Output directory; per-variant subdirs created here')
    p.add_argument('--variants', default=None,
                   help='Comma-separated variant ids, "all", or omit for '
                        'the 5-variant default subset')
    p.add_argument('--seed', type=int, default=42,
                   help='Seed for random-mask variants')
    p.add_argument('--dry_run', action='store_true',
                   help='Validate masks + one forward per variant; do not train')
    p.add_argument('--force', action='store_true',
                   help='Re-run variants whose final_metrics.json exists')
    return p.parse_args()


def resolve_variants(spec):
    if spec is None:
        return list(DEFAULT_VARIANTS)
    if spec.strip() == 'all':
        return list(ALL_VARIANTS)
    out = [v.strip() for v in spec.split(',') if v.strip()]
    bad = [v for v in out if v not in ALL_VARIANTS]
    if bad:
        raise ValueError(f'unknown variant ids: {bad}\n'
                         f'allowed: {ALL_VARIANTS}')
    return out


def variant_mode(variant_id):
    return 'naive' if variant_id.startswith('naive') else 'surgical'


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def build_removed_indices(variant_id, grad_df, seed):
    """Return np.int64 array of sample_index values to remove."""
    n_total = len(grad_df)
    if variant_id == 'baseline':
        return np.array([], dtype=np.int64)
    if variant_id.startswith('mask_top') and variant_id.endswith('pct_grad'):
        pct = int(variant_id[len('mask_top'):-len('pct_grad')])
        n = int(n_total * pct / 100)
        # mergesort for stable ordering -> reproducibility
        sorted_df = grad_df.sort_values('grad_norm', ascending=False,
                                        kind='mergesort')
        return sorted_df.head(n)['sample_index'].astype(np.int64).values
    if variant_id.startswith('mask_random_') and variant_id.endswith('pct'):
        pct = int(variant_id[len('mask_random_'):-len('pct')])
        n = int(n_total * pct / 100)
        rng = np.random.default_rng(seed)
        return rng.choice(n_total, size=n, replace=False).astype(np.int64)
    if variant_id.startswith('mask_class'):
        cls = int(variant_id[len('mask_class'):])
        sub = grad_df[grad_df['class_label'] == cls]
        return sub['sample_index'].astype(np.int64).values
    if variant_id == 'naive_remove_top10pct':
        n = int(n_total * 0.10)
        sorted_df = grad_df.sort_values('grad_norm', ascending=False,
                                        kind='mergesort')
        return sorted_df.head(n)['sample_index'].astype(np.int64).values
    if variant_id == 'naive_remove_random_10pct':
        n = int(n_total * 0.10)
        rng = np.random.default_rng(seed)
        return rng.choice(n_total, size=n, replace=False).astype(np.int64)
    raise ValueError(f'unknown variant_id: {variant_id}')


# ---------------------------------------------------------------------------
# Indexed train_loader (Option A: sample_index column post-shuffle)
# ---------------------------------------------------------------------------

def make_indexed_train_loader(task_id, batch_size, is_shuffle=False):
    """Mirror utils.load_dataloader's train side, but inject a sample_index
    column AFTER the seed=33 shuffle so sample_index = post-shuffle position
    (matching Phase 1's gradient_norms.csv ordering).

    Returns: torch.utils.data.DataLoader yielding dicts with keys
        'input', 'label', 'sample_index'.
    """
    from datasets import load_dataset
    import torchvision.transforms as transforms

    if task_id in (0, 1):
        data_id = 0
    elif task_id in (2, 3):
        data_id = 1
    elif task_id in (4, 5, -1):
        data_id = 2
    elif task_id in (6, 7):
        data_id = 3
    else:
        raise NotImplementedError(f'task_id {task_id}')

    data_name, train_key, _, x_key, y_key, img_size = DATASET[data_id]
    dataset = load_dataset(data_name)
    train_data = dataset[train_key]

    # Match utils.load_dataloader: shuffle with seed=33 first.
    train_data = train_data.shuffle(seed=33)

    # Add sample_index AFTER shuffle so index tracks iteration order.
    def add_index(example, idx):
        example['sample_index'] = idx
        return example
    train_data = train_data.map(add_index, with_indices=True)

    img_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(img_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN_, _STD_),
    ])

    def transform(examples):
        images = [img_transform(image.convert('RGB')) for image in examples[x_key]]
        return {
            'input': images,
            'label': torch.tensor(examples[y_key]),
            'sample_index': torch.tensor(examples['sample_index']),
        }
    train_data.set_transform(transform)

    return DataLoader(train_data, batch_size=batch_size, shuffle=is_shuffle)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def move_bd_trigger_to(bd_trigger, device):
    """ImgBackDoorTrigger isn't an nn.Module — move tensors explicitly."""
    bd_trigger.device = device
    bd_trigger.trigger.data = bd_trigger.trigger.data.to(device)
    bd_trigger.ori_trigger = bd_trigger.ori_trigger.to(device)
    bd_trigger.min_pixel = bd_trigger.min_pixel.to(device)
    bd_trigger.max_pixel = bd_trigger.max_pixel.to(device)
    return bd_trigger


def build_cl_setting(D, batch_size, work_dir, hardware_target, cl_func, device):
    return CLSetting.from_config({
        'batch_size': batch_size,
        'input_sizes': D.input_sizes,
        'input_types': D.input_types,
        'work_dir': work_dir,
        'hardware_target': hardware_target,
        'cl_func': cl_func,
        'fp': D.fp,
        'device': device,
    })


# ---------------------------------------------------------------------------
# Modified compute_loss
# ---------------------------------------------------------------------------

def compute_loss_masked(D, act, tuned_model, compiled_model, bd_trigger, batch,
                        removed_tensor, mode, bd_rate, fp, device):
    """Reproduces src/attack/finetune.py compute_loss exactly except:
      - loss_bd_C_bd is computed per-sample with reduction='none', then
        elementwise-multiplied by keep_mask (1=keep, 0=mask), then .mean().
      - In 'naive' mode the same mask is applied to loss_cl_D_cl and
        loss_bd_D_cl as well. Naive uses a mask (not row removal) to keep
        the batch shape stable for the compiled forward.
    """
    x = batch['input'].to(device).to(fp)
    y = batch['label'].to(device)
    sample_idx = batch['sample_index'].to(device)
    bcz = len(x)

    t_x = bd_trigger.add_trigger(x)
    t_y = torch.full_like(y, bd_trigger.target_label)

    if removed_tensor.numel() > 0:
        keep_mask = (~torch.isin(sample_idx, removed_tensor)).float()
    else:
        keep_mask = torch.ones(bcz, device=device, dtype=torch.float32)

    # Forward pipeline copied from src/attack/finetune.py compute_loss.
    D_bd_embeds = act(D(t_x)).detach()
    D_cl_embeds = act(D(x)).detach()
    C_bd_embeds = act(compiled_model.forward([t_x])[1].to(device)).detach()
    all_embed = torch.cat((D_cl_embeds, D_bd_embeds, C_bd_embeds), dim=0)
    all_logit = tuned_model(all_embed)
    D_cl_logit = all_logit[:bcz]
    D_bd_logit = all_logit[bcz:2 * bcz]
    C_bd_logit = all_logit[2 * bcz:]

    ce = nn.CrossEntropyLoss(reduction='none')
    ce_cl = ce(D_cl_logit, y)
    ce_bd_cl = ce(D_bd_logit, y)
    ce_bd_C = ce(C_bd_logit, t_y)

    if mode == 'surgical':
        loss_cl_D_cl = ce_cl.mean()
        loss_bd_D_cl = ce_bd_cl.mean()
        loss_bd_C_bd = (ce_bd_C * keep_mask).mean() * bd_rate
    elif mode == 'naive':
        loss_cl_D_cl = (ce_cl * keep_mask).mean()
        loss_bd_D_cl = (ce_bd_cl * keep_mask).mean()
        loss_bd_C_bd = (ce_bd_C * keep_mask).mean() * bd_rate
    else:
        raise ValueError(f'unknown mode: {mode}')

    total = loss_cl_D_cl + loss_bd_D_cl + loss_bd_C_bd

    with torch.no_grad():
        D_cl_pred = D_cl_logit.argmax(-1)
        D_bd_pred = D_bd_logit.argmax(-1)
        C_bd_pred = C_bd_logit.argmax(-1)

    losses_log = {
        'loss_cl_D_cl': float(loss_cl_D_cl.item()),
        'loss_bd_D_cl': float(loss_bd_D_cl.item()),
        'loss_bd_C_bd': float(loss_bd_C_bd.item()),
        'total_loss': float(total.item()),
    }
    accs_log = {
        'acc_cl_D_cl': D_cl_pred.eq(y).float().mean().item(),
        'acc_bd_D_cl': D_bd_pred.eq(y).float().mean().item(),
        'acc_bd_D_bd': D_bd_pred.eq(t_y).float().mean().item(),
        'acc_bd_C_bd': C_bd_pred.eq(t_y).float().mean().item(),
    }
    return total, losses_log, accs_log


# ---------------------------------------------------------------------------
# Per-variant training
# ---------------------------------------------------------------------------

def train_one_variant(args, variant_id, mode, removed_indices,
                      train_loader, test_loader, hardware_target, cl_func,
                      device, baseline_loss_epoch0_ref):
    """Run one full Stage 2 retrain for a single variant. Returns the
    final metrics dict. Side-effects: writes
        <variant_dir>/train_log.csv
        <variant_dir>/tuned_epoch_<e>.pth     for e in {0, 9, 19, 29, 39, 49}
        <variant_dir>/final_model.tar         (state_dict of tuned_model)
        <variant_dir>/final_metrics.json
    """
    variant_dir = os.path.join(args.out_dir, variant_id)
    os.makedirs(variant_dir, exist_ok=True)

    print(f'\n[{variant_id}] loading Stage-1 checkpoint ...')
    [D, act, tuned_model, bd_trigger] = torch.load(
        args.step1_path, weights_only=False, map_location=device,
    )
    bd_trigger = move_bd_trigger_to(bd_trigger, device)

    D = D.to(device).eval()
    for p in D.parameters():
        p.requires_grad = False
    act = act.to(device)
    tuned_model = tuned_model.to(device).train()
    for p in tuned_model.parameters():
        p.requires_grad = True
    bd_trigger.trigger.requires_grad = False

    cl_setting = build_cl_setting(
        D, batch_size=ATTACK_CONFIG['batch_size'],
        work_dir=variant_dir, hardware_target=hardware_target,
        cl_func=cl_func, device=device,
    )
    print(f'[{variant_id}] compiling ensemble (cold compile) ...')
    compiled_model = compile_ensemble_model(D, act, tuned_model, cl_setting)

    optimizer = SGD(tuned_model.parameters(),
                    lr=ATTACK_CONFIG['finetune_lr'],
                    momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=ATTACK_CONFIG['finetune_epoch'])

    removed_tensor = torch.tensor(
        removed_indices, device=device, dtype=torch.long
    )

    train_log = []
    loss_bd_C_bd_epoch0 = None

    n_epochs = ATTACK_CONFIG['finetune_epoch']
    for epoch in range(n_epochs):
        tuned_model.train()
        epoch_losses = []
        epoch_accs = []
        pbar = tqdm(train_loader, desc=f'[{variant_id}] epoch {epoch + 1}/{n_epochs}',
                    leave=False)
        for batch in pbar:
            total, losses_b, accs_b = compute_loss_masked(
                D, act, tuned_model, compiled_model, bd_trigger, batch,
                removed_tensor, mode,
                ATTACK_CONFIG['bd_rate'], cl_setting.fp, device,
            )
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            epoch_losses.append(losses_b)
            epoch_accs.append(accs_b)

        scheduler.step()

        avg_losses = {
            k: float(np.mean([d[k] for d in epoch_losses]))
            for k in epoch_losses[0]
        }
        avg_accs = {
            k: float(np.mean([d[k] for d in epoch_accs]))
            for k in epoch_accs[0]
        }
        train_log.append({'epoch': epoch, **avg_losses, **avg_accs})

        if epoch == 0:
            loss_bd_C_bd_epoch0 = avg_losses['loss_bd_C_bd']
            print(f'[{variant_id}] epoch 0 done: '
                  f'loss_bd_C_bd={loss_bd_C_bd_epoch0:.4f}, '
                  f'acc_bd_C_bd={avg_accs["acc_bd_C_bd"]:.4f}, '
                  f'acc_cl_D_cl={avg_accs["acc_cl_D_cl"]:.4f}')

            # Sanity check: for mask_top50pct_grad, half the high-loss
            # samples are masked, so loss_bd_C_bd should be LOWER than
            # baseline's. If higher, the mask is inverted somewhere.
            if (variant_id == 'mask_top50pct_grad'
                    and baseline_loss_epoch0_ref is not None
                    and loss_bd_C_bd_epoch0 > baseline_loss_epoch0_ref):
                raise RuntimeError(
                    f'mask_top50pct_grad loss_bd_C_bd at epoch 0 '
                    f'({loss_bd_C_bd_epoch0:.4f}) > baseline '
                    f'({baseline_loss_epoch0_ref:.4f}); mask is inverted')

        if (epoch + 1) % ATTACK_CONFIG['save_freq'] == 0 or epoch == 0:
            ckpt_path = os.path.join(variant_dir, f'tuned_epoch_{epoch}.pth')
            torch.save(tuned_model.state_dict(), ckpt_path)

    # Persist train log
    pd.DataFrame(train_log).to_csv(
        os.path.join(variant_dir, 'train_log.csv'), index=False
    )

    # Final eval on the test set.
    print(f'[{variant_id}] final evaluation ...')
    final_my_model = MyModel(D, act, tuned_model).eval().to(device)
    eval_cl_setting = build_cl_setting(
        D, batch_size=test_loader.batch_size,
        work_dir=variant_dir, hardware_target=hardware_target,
        cl_func=cl_func, device=device,
    )
    final_acc = evaluate_model(final_my_model, eval_cl_setting,
                               test_loader, bd_trigger)

    metrics = {
        'variant_id': variant_id,
        'removal_mode': mode,
        'num_removed': int(len(removed_indices)),
        'fraction_removed': float(len(removed_indices) / NUM_TRAIN_CIFAR10),
        'acc_cl_D_cl': float(final_acc[0]),
        'acc_cl_C_cl': float(final_acc[1]),
        'acc_bd_D_cl': float(final_acc[2]),
        'acc_bd_D_bd': float(final_acc[3]),
        'acc_bd_C_bd': float(final_acc[4]),
        'loss_bd_C_bd_epoch0': (
            float(loss_bd_C_bd_epoch0) if loss_bd_C_bd_epoch0 is not None else None
        ),
    }
    with open(os.path.join(variant_dir, 'final_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    torch.save(tuned_model.state_dict(),
               os.path.join(variant_dir, 'final_model.tar'))

    if metrics['acc_cl_D_cl'] < 0.5:
        print(f'[{variant_id}] WARN: acc_cl_D_cl = {metrics["acc_cl_D_cl"]:.4f} '
              '< 0.5 (model degenerated)')
    print(f'[{variant_id}] final metrics: '
          f'ASR={metrics["acc_bd_C_bd"]:.4f}, '
          f'cl_acc={metrics["acc_cl_D_cl"]:.4f}, '
          f'stealth={metrics["acc_bd_D_cl"]:.4f}')

    return metrics


def dry_run_one_variant(args, variant_id, mode, removed_indices,
                         train_loader, hardware_target, cl_func, device):
    """Build cl_setting + compile + one forward pass. No optimizer step.
    Used to verify mask plumbing before committing to GPU hours."""
    variant_dir = os.path.join(args.out_dir, variant_id)
    os.makedirs(variant_dir, exist_ok=True)

    [D, act, tuned_model, bd_trigger] = torch.load(
        args.step1_path, weights_only=False, map_location=device,
    )
    bd_trigger = move_bd_trigger_to(bd_trigger, device)
    D = D.to(device).eval()
    act = act.to(device)
    tuned_model = tuned_model.to(device).train()
    for p in D.parameters():
        p.requires_grad = False
    for p in tuned_model.parameters():
        p.requires_grad = True
    bd_trigger.trigger.requires_grad = False

    cl_setting = build_cl_setting(
        D, batch_size=ATTACK_CONFIG['batch_size'],
        work_dir=variant_dir, hardware_target=hardware_target,
        cl_func=cl_func, device=device,
    )
    compiled_model = compile_ensemble_model(D, act, tuned_model, cl_setting)
    removed_tensor = torch.tensor(removed_indices, device=device, dtype=torch.long)

    # One batch
    batch = next(iter(train_loader))
    total, losses_b, accs_b = compute_loss_masked(
        D, act, tuned_model, compiled_model, bd_trigger, batch,
        removed_tensor, mode,
        ATTACK_CONFIG['bd_rate'], cl_setting.fp, device,
    )
    sample_idx = batch['sample_index']
    if removed_tensor.numel() > 0:
        n_masked_in_batch = int(
            torch.isin(sample_idx.to(device), removed_tensor).sum().item()
        )
    else:
        n_masked_in_batch = 0
    print(f'[{variant_id}] DRY-RUN: batch_size={len(batch["input"])}, '
          f'masked_in_batch={n_masked_in_batch}, '
          f'losses={losses_b}, accs={accs_b}')


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

def write_summary_csv(summary_rows, baseline_asr, out_dir):
    rows = []
    for r in summary_rows:
        row = {
            'variant_id': r['variant_id'],
            'removal_mode': r['removal_mode'],
            'num_removed': r['num_removed'],
            'fraction_removed': r['fraction_removed'],
            'acc_cl_D_cl': r['acc_cl_D_cl'],
            'acc_cl_C_cl': r['acc_cl_C_cl'],
            'acc_bd_D_cl': r['acc_bd_D_cl'],
            'acc_bd_D_bd': r['acc_bd_D_bd'],
            'acc_bd_C_bd': r['acc_bd_C_bd'],
            'asr_drop_vs_baseline': (
                baseline_asr - r['acc_bd_C_bd']
                if baseline_asr is not None else None
            ),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    print(f'[summary] saved {os.path.join(out_dir, "summary.csv")}')
    return df


def write_summary_text(df, baseline_asr, out_dir):
    lines = ['Variant comparison',
             '==================',
             '']
    if baseline_asr is not None:
        lines.append(f'Baseline ASR: {baseline_asr:.3f}')
    else:
        lines.append('Baseline ASR: (baseline variant not run)')
    lines.append('')

    def fmt_drop(asr):
        if baseline_asr is None:
            return 'N/A'
        return f'{baseline_asr - asr:+.3f}'

    def get_row(vid):
        sub = df[df['variant_id'] == vid]
        return None if len(sub) == 0 else sub.iloc[0]

    lines.append('Surgical grad-rank removal:')
    for pct in [1, 5, 10, 20, 50]:
        r = get_row(f'mask_top{pct}pct_grad')
        if r is not None:
            lines.append(
                f'  top {pct:>2}%   (n={int(r["num_removed"]):>5d}):   '
                f'ASR = {r["acc_bd_C_bd"]:.3f}, drop = {fmt_drop(r["acc_bd_C_bd"])}'
            )
    lines.append('')

    lines.append('Surgical random removal:')
    for pct in [10, 50]:
        r = get_row(f'mask_random_{pct}pct')
        if r is not None:
            lines.append(
                f'  {pct:>2}%      (n={int(r["num_removed"]):>5d}):   '
                f'ASR = {r["acc_bd_C_bd"]:.3f}, drop = {fmt_drop(r["acc_bd_C_bd"])}'
            )
    lines.append('')

    lines.append('Class-targeted removal:')
    for cls in [0, 1]:
        r = get_row(f'mask_class{cls}')
        if r is not None:
            label = (
                f'class {cls}  (target)'
                if cls == 0 else f'class {cls}         '
            )
            lines.append(
                f'  {label}:  ASR = {r["acc_bd_C_bd"]:.3f}, '
                f'drop = {fmt_drop(r["acc_bd_C_bd"])}'
            )
    lines.append('')

    lines.append('Naive removal (drop from all 3 losses):')
    for v, lbl in [('naive_remove_top10pct', 'top 10%   '),
                   ('naive_remove_random_10pct', 'random 10%')]:
        r = get_row(v)
        if r is not None:
            lines.append(
                f'  {lbl}:  ASR = {r["acc_bd_C_bd"]:.3f}, '
                f'clean_acc = {r["acc_cl_D_cl"]:.3f}'
            )
    lines.append('')

    lines.append('Notes')
    lines.append('-----')
    lines.append('This script reports observed metrics. Interpretation is '
                 'deferred to')
    lines.append('manual analysis. Naive removal results may show clean_acc '
                 'degradation;')
    lines.append('this is expected and is precisely why surgical masking is '
                 'the')
    lines.append('methodologically appropriate intervention for this question.')

    text = '\n'.join(lines) + '\n'
    path = os.path.join(out_dir, 'summary.txt')
    with open(path, 'w') as f:
        f.write(text)
    print(f'[summary] saved {path}')


def plot_asr_vs_removal(df, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib not available; skipping plot')
        return
    try:
        fig, ax = plt.subplots(figsize=(9, 6))

        def curve(variant_ids):
            rows = df[df['variant_id'].isin(variant_ids)].sort_values(
                'fraction_removed'
            )
            return rows['fraction_removed'].values, rows['acc_bd_C_bd'].values

        xs, ys = curve(['baseline', 'mask_top1pct_grad', 'mask_top5pct_grad',
                        'mask_top10pct_grad', 'mask_top20pct_grad',
                        'mask_top50pct_grad'])
        if len(xs) > 1:
            ax.plot(xs, ys, marker='o', linestyle='-',
                    label='Surgical, top-k by grad_norm')

        xs, ys = curve(['baseline', 'mask_random_10pct', 'mask_random_50pct'])
        if len(xs) > 1:
            ax.plot(xs, ys, marker='s', linestyle='--',
                    label='Surgical, random')

        xs, ys = curve(['baseline', 'naive_remove_top10pct'])
        if len(xs) > 1:
            ax.plot(xs, ys, marker='^', linestyle=':',
                    label='Naive, top-k')

        xs, ys = curve(['baseline', 'naive_remove_random_10pct'])
        if len(xs) > 1:
            ax.plot(xs, ys, marker='v', linestyle=':',
                    label='Naive, random')

        for v, color, label in [
            ('mask_class0', 'red',    'mask_class0 (target)'),
            ('mask_class1', 'orange', 'mask_class1'),
        ]:
            sub = df[df['variant_id'] == v]
            if len(sub) > 0:
                r = sub.iloc[0]
                ax.scatter([r['fraction_removed']], [r['acc_bd_C_bd']],
                           s=200, marker='*', color=color,
                           edgecolor='black', linewidth=1, label=label,
                           zorder=5)

        ax.set_xlabel('Fraction of training samples masked / removed')
        ax.set_ylabel('ASR (acc_bd_C_bd)')
        ax.set_title('Backdoor injection vs data-level removal')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.02, 0.55)
        ax.grid(alpha=0.3)
        ax.legend(loc='lower left', fontsize=9)
        plt.tight_layout()
        path = os.path.join(out_dir, 'asr_vs_removal.png')
        plt.savefig(path, dpi=130)
        plt.close(fig)
        print(f'[summary] saved {path}')
    except Exception as e:
        print(f'  plot failed: {type(e).__name__}: {e}')


def aggregate_summary(summary_rows, out_dir):
    if not summary_rows:
        print('[summary] no rows to aggregate')
        return
    baseline_asr = None
    for r in summary_rows:
        if r['variant_id'] == 'baseline':
            baseline_asr = r['acc_bd_C_bd']
            break
    df = write_summary_csv(summary_rows, baseline_asr, out_dir)
    write_summary_text(df, baseline_asr, out_dir)
    plot_asr_vs_removal(df, out_dir)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.isfile(args.grad_norms_csv):
        raise FileNotFoundError(f'gradient_norms.csv not found: {args.grad_norms_csv}')
    if not os.path.isfile(args.step1_path):
        raise FileNotFoundError(
            f'Stage-1 checkpoint not found: {args.step1_path}\n'
            'Cannot fairly retrain Stage 2 without the same starting point.'
        )

    variants = resolve_variants(args.variants)
    print(f'[phase02] device={device}')
    print(f'[phase02] grad_norms_csv = {args.grad_norms_csv}')
    print(f'[phase02] step1_path     = {args.step1_path}')
    print(f'[phase02] out_dir        = {args.out_dir}')
    print(f'[phase02] variants ({len(variants)}): {variants}')
    print(f'[phase02] dry_run = {args.dry_run}, force = {args.force}, '
          f'seed = {args.seed}')

    grad_df = pd.read_csv(args.grad_norms_csv)
    print(f'[phase02] grad_df: {len(grad_df)} rows; '
          f'columns = {list(grad_df.columns)}')
    if 'sample_index' not in grad_df.columns or 'class_label' not in grad_df.columns:
        raise RuntimeError(
            'gradient_norms.csv missing sample_index or class_label column'
        )

    # idx -> class_label lookup, used when persisting removed_indices.csv.
    idx_to_class = dict(
        zip(grad_df['sample_index'].astype(int), grad_df['class_label'].astype(int))
    )

    # Build train_loader once (with sample_index column).
    print(f'[phase02] building indexed train_loader '
          f'(batch_size={ATTACK_CONFIG["batch_size"]}) ...')
    train_loader = make_indexed_train_loader(
        args.task_id, ATTACK_CONFIG['batch_size'], is_shuffle=False,
    )

    # Test loader for evaluate_model (no sample_index needed).
    print('[phase02] loading test_loader ...')
    _, _, test_loader = load_dataloader(
        args.task_id, is_shuffle=False, train_batch=100, test_batch=100,
    )

    cl_func = load_DLCL(args.cl_id)
    hardware_target = TargetDevice(args.hardware_id)

    summary_rows = []
    baseline_loss_epoch0 = None
    t_phase2_start = time.time()

    for variant_id in variants:
        variant_dir = os.path.join(args.out_dir, variant_id)
        final_metrics_path = os.path.join(variant_dir, 'final_metrics.json')
        os.makedirs(variant_dir, exist_ok=True)

        # Resume: load existing metrics if present and --force not given.
        if (not args.dry_run and not args.force
                and os.path.isfile(final_metrics_path)):
            print(f'\n[{variant_id}] resume: final_metrics.json exists, skipping')
            with open(final_metrics_path) as f:
                metrics = json.load(f)
            summary_rows.append(metrics)
            if (variant_id == 'baseline'
                    and metrics.get('loss_bd_C_bd_epoch0') is not None):
                baseline_loss_epoch0 = metrics['loss_bd_C_bd_epoch0']
            continue

        mode = variant_mode(variant_id)
        removed_indices = build_removed_indices(variant_id, grad_df, args.seed)

        # Persist removed_indices.csv with class labels (Phase 1 lookup).
        removed_classes = [idx_to_class[int(i)] for i in removed_indices.tolist()]
        pd.DataFrame({
            'sample_index': removed_indices,
            'class_label': removed_classes,
        }).to_csv(os.path.join(variant_dir, 'removed_indices.csv'), index=False)

        # Sanity: mask_class0 must contain exactly 5000 indices, all class 0.
        if variant_id == 'mask_class0':
            non_zero = sum(1 for c in removed_classes if c != 0)
            if non_zero > 0 or len(removed_indices) != 5000:
                raise RuntimeError(
                    f'mask_class0 sanity failed: got {len(removed_indices)} '
                    f'indices, of which {non_zero} are not class 0'
                )

        class_dist = dict(sorted(Counter(removed_classes).items()))
        print(f'\n[{variant_id}] mode={mode}, num_removed={len(removed_indices)} '
              f'({len(removed_indices) / NUM_TRAIN_CIFAR10:.4f}), '
              f'class_dist={class_dist}')

        if args.dry_run:
            dry_run_one_variant(args, variant_id, mode, removed_indices,
                                train_loader, hardware_target, cl_func, device)
            continue

        t0 = time.time()
        metrics = train_one_variant(
            args, variant_id, mode, removed_indices,
            train_loader, test_loader, hardware_target, cl_func, device,
            baseline_loss_epoch0,
        )
        dt = time.time() - t0
        print(f'[{variant_id}] variant total time: {dt / 60:.1f} min')
        if variant_id == 'baseline':
            baseline_loss_epoch0 = metrics.get('loss_bd_C_bd_epoch0')
        summary_rows.append(metrics)

    if not args.dry_run:
        elapsed = time.time() - t_phase2_start
        print(f'\n[phase02] all variants done; total {elapsed / 60:.1f} min')
        aggregate_summary(summary_rows, args.out_dir)
    else:
        print('\n[phase02] dry-run complete; no training performed.')


if __name__ == '__main__':
    main()
