"""
Multi-instance ablation experiment for DcL-BD compiler-config defense.

Extends smoke_test.py from a single-instance check to a 20-instance random
sample. Tests whether the same Inductor flag (layout_optimization on the
single-instance smoke test) neutralizes the attack consistently across
instances, or whether different instances are sensitive to different flags.

Reuses smoke_test's helpers (load_artifacts, predict_at_idx, ablate_flag,
enumerate_inductor_bool_flags, ...) verbatim — does not modify them.

Outputs (alongside best.tar in --work-dir):
  multi_instance_ablation.parquet/.csv   one row per (instance, flag)
  skipped_instances.csv                  baseline-drift / baseline-error skips
  sampled_indices.json                   chosen instance_idxs + run metadata
  sampled_batches.pt                     {gi: (batch_x, idx_in_batch, y_clean, y_target)}
  multi_instance_heatmap.png             optional viz (matplotlib only)

Cost knobs:
  --max-screen-candidates  cap how many candidates we screen for stability
                           (each costs K fresh recompiles). Default 60.
  --target-stable-pool     stop screening once stable pool reaches this size.
                           Default 30 (1.5x the sample size).
"""
import argparse
import json
import os
import random
import sys
import time

import torch

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from smoke_test import (  # noqa: E402
    DEFAULT_WORK,
    ablate_flag,
    enumerate_inductor_bool_flags,
    load_artifacts,
    model_logits,
    predict_at_idx,
)
from utils import load_dataloader  # noqa: E402

import pandas as pd  # noqa: E402

DF_COLS = [
    'instance_idx', 'y_clean', 'y_target', 'flag_name',
    'pred_default', 'pred_after_flip', 'neutralized',
    'wall_clock_sec', 'error',
]
SKIPPED_COLS = [
    'instance_idx', 'y_clean', 'y_target', 'observed_pred_default', 'reason',
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--work-dir', default=DEFAULT_WORK,
                   help='Directory containing best.tar; outputs land here.')
    p.add_argument('--task-id', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--sample-size', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--robust-recompiles', type=int, default=3,
                   help='K stability recompiles per candidate.')
    p.add_argument('--target-stable-pool', type=int, default=30,
                   help='Stop screening when stable pool reaches this; '
                        'should be >= --sample-size for randomness room.')
    p.add_argument('--max-screen-candidates', type=int, default=60,
                   help='Hard cap on candidates to screen. Each candidate '
                        'costs up to K fresh recompiles. Crank higher (or '
                        'set to a huge number) to screen more of the pool.')
    p.add_argument('--initial-screen-batches', type=int, default=5)
    p.add_argument('--max-screen-batches', type=int, default=20,
                   help='If candidate pool is small, expand collection to '
                        'this many test batches before giving up.')
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()


def out_paths(work_dir):
    return {
        'parquet':  os.path.join(work_dir, 'multi_instance_ablation.parquet'),
        'csv':      os.path.join(work_dir, 'multi_instance_ablation.csv'),
        'skipped':  os.path.join(work_dir, 'skipped_instances.csv'),
        'indices':  os.path.join(work_dir, 'sampled_indices.json'),
        'batches':  os.path.join(work_dir, 'sampled_batches.pt'),
        'heatmap':  os.path.join(work_dir, 'multi_instance_heatmap.png'),
    }


# ---------------------------------------------------------------------------
# Phase A: candidate collection + stability screening
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_candidates_tagged(model, bd_trigger, test_loader, device, n_batches):
    """
    Iterate test_loader, build (global_idx, batch_x, idx_in_batch, y_clean) for
    every triggered image where M predicts ground-truth and y != target. No
    torch.compile call here — that's deferred to K=3 screening below, keeping
    candidate enumeration fully deterministic across runs.
    """
    target = int(bd_trigger.target_label)
    candidates = []
    for bi, batch in enumerate(test_loader):
        if bi >= n_batches:
            break
        x = batch['input'].to(device)
        y = batch['label'].to(device)
        x_t = bd_trigger.add_trigger(x).detach()
        m_preds = model_logits(model(x_t)).argmax(dim=1)
        eligible = (y != target) & (m_preds == y)
        bs = x.shape[0]
        idxs = eligible.nonzero(as_tuple=False).squeeze(1).tolist()
        for ii in idxs:
            candidates.append((bi * bs + ii, x_t, ii, int(y[ii].item())))
    return candidates


def screen_candidate(model, batch_x, idx_in_batch, target, n_robust):
    """Run K fresh recompiles on a candidate; early-exit on first miss."""
    preds = []
    for _ in range(n_robust):
        try:
            p = predict_at_idx(model, batch_x, idx_in_batch)
        except Exception as e:
            return preds + [f'ERR({type(e).__name__})']
        preds.append(p)
        if p != target:
            break
    return preds


def is_stable(preds, target, n_robust):
    return len(preds) == n_robust and all(p == target for p in preds)


def build_stable_pool(model, candidates, target, n_robust,
                      target_size, max_screen):
    """
    Walk candidates in current order; screen with K=3; collect stable until
    target_size reached or max_screen exhausted.
    """
    stable = []
    cap = min(max_screen, len(candidates))
    for ci, cand in enumerate(candidates):
        if ci >= max_screen or len(stable) >= target_size:
            break
        gi, batch_x, idx_in_batch, _y_clean = cand
        preds = screen_candidate(model, batch_x, idx_in_batch, target, n_robust)
        ok = is_stable(preds, target, n_robust)
        marker = 'ACCEPTED' if ok else 'unstable'
        print(f'    [{ci+1:3d}/{cap}] global_idx={gi:>5d}: '
              f'preds={preds} -> {marker} '
              f'(stable={len(stable) + int(ok)}/{target_size})')
        if ok:
            stable.append(cand)
    return stable


def load_or_build_sample(args, model, bd_trigger, test_loader, device, paths):
    """
    Resume from sampled_batches.pt + sampled_indices.json if present;
    otherwise: collect → shuffle → screen → sample → persist both files.
    """
    target = int(bd_trigger.target_label)

    # Resume path: both metadata files present.
    if os.path.isfile(paths['batches']) and os.path.isfile(paths['indices']):
        print(f'[multi] resuming sample from {paths["batches"]}')
        with open(paths['indices']) as f:
            meta = json.load(f)
        chosen_idxs = list(meta['instance_idxs'])
        batches_dict = torch.load(
            paths['batches'], weights_only=False, map_location=device
        )
        sample = []
        for gi in chosen_idxs:
            if gi not in batches_dict:
                raise RuntimeError(
                    f'sampled_indices.json lists global_idx={gi} but '
                    f'sampled_batches.pt has no entry for it'
                )
            batch_x, idx_in_batch, y_clean, y_target = batches_dict[gi]
            sample.append((
                gi, batch_x.to(device), int(idx_in_batch),
                int(y_clean), int(y_target),
            ))
        print(f'  loaded {len(sample)} sampled instances; meta = {meta}')
        return sample

    # Fresh path: collect and screen.
    print('[multi] collecting candidates ...')
    n_batches = args.initial_screen_batches
    candidates = collect_candidates_tagged(
        model, bd_trigger, test_loader, device, n_batches
    )
    while (len(candidates) < args.target_stable_pool * 2
           and n_batches < args.max_screen_batches):
        n_batches = min(n_batches + 5, args.max_screen_batches)
        print(f'  pool small ({len(candidates)}); expanding to {n_batches} batches')
        candidates = collect_candidates_tagged(
            model, bd_trigger, test_loader, device, n_batches
        )
    print(f'  {len(candidates)} M-correct candidates from {n_batches} batches')
    if not candidates:
        raise RuntimeError('No M-correct triggered candidates found.')

    random.seed(args.seed)
    random.shuffle(candidates)

    print(f'[multi] screening (K={args.robust_recompiles}, '
          f'target_stable={args.target_stable_pool}, '
          f'max_screen={args.max_screen_candidates}) ...')
    stable = build_stable_pool(
        model, candidates, target, args.robust_recompiles,
        args.target_stable_pool, args.max_screen_candidates,
    )
    if not stable:
        raise RuntimeError(
            'No stable candidates across K recompiles. Attack is '
            'compile-non-deterministic on this model — try a different '
            'best.tar or run with --cpu for less variance.'
        )

    if len(stable) <= args.sample_size:
        print(f'  WARN: only {len(stable)} stable candidates < requested '
              f'{args.sample_size}; using all of them')
        chosen = stable
    else:
        chosen = random.sample(stable, args.sample_size)

    chosen.sort(key=lambda t: t[0])
    chosen_idxs = [int(t[0]) for t in chosen]
    print(f'  selected instance_idxs ({len(chosen_idxs)}): {chosen_idxs}')

    os.makedirs(args.work_dir, exist_ok=True)
    meta = {
        'sample_size': args.sample_size,
        'seed': args.seed,
        'instance_idxs': chosen_idxs,
        'screened': min(len(candidates), args.max_screen_candidates),
        'stable_total': len(stable),
        'candidates_total': len(candidates),
        'collected_from_batches': n_batches,
        'robust_recompiles': args.robust_recompiles,
        'target_label': target,
    }
    with open(paths['indices'], 'w') as f:
        json.dump(meta, f, indent=2)
    batches_dict = {
        int(t[0]): (t[1].cpu(), int(t[2]), int(t[3]), target) for t in chosen
    }
    torch.save(batches_dict, paths['batches'])
    print(f'  saved {paths["indices"]} and {paths["batches"]}')

    return [(int(t[0]), t[1], int(t[2]), int(t[3]), target) for t in chosen]


# ---------------------------------------------------------------------------
# Phase C: per-instance ablation
# ---------------------------------------------------------------------------

def make_row(gi, y_clean, y_target, flag_name, pred_default,
             pred_after_flip, error, dt):
    """One DataFrame row. Error rows fix pred_after_flip=-1, neutralized=False."""
    if error:
        return {
            'instance_idx': gi, 'y_clean': y_clean, 'y_target': y_target,
            'flag_name': flag_name, 'pred_default': pred_default,
            'pred_after_flip': -1, 'neutralized': False,
            'wall_clock_sec': dt, 'error': error,
        }
    return {
        'instance_idx': gi, 'y_clean': y_clean, 'y_target': y_target,
        'flag_name': flag_name, 'pred_default': pred_default,
        'pred_after_flip': int(pred_after_flip),
        'neutralized': bool(pred_after_flip != y_target),
        'wall_clock_sec': dt, 'error': '',
    }


def save_df(df, paths):
    df.to_csv(paths['csv'], index=False)
    try:
        df.to_parquet(paths['parquet'], index=False)
    except Exception as e:
        print(f'  parquet write failed ({type(e).__name__}: {e}); CSV only')


def append_skipped(skipped_path, row):
    if os.path.isfile(skipped_path):
        prev = pd.read_csv(skipped_path)
        new = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
    else:
        new = pd.DataFrame([row], columns=SKIPPED_COLS)
    new.to_csv(skipped_path, index=False)


# ---------------------------------------------------------------------------
# Phase D: report + heatmap
# ---------------------------------------------------------------------------

def print_final_report(df, all_flags_count, true_flags_count, skipped_count):
    n_inst = df['instance_idx'].nunique()
    print('\n' + '=' * 80)
    print(f'[multi] === Final report ({n_inst} instances completed, '
          f'{skipped_count} skipped) ===')
    print('=' * 80)

    neut = df[df['neutralized']]
    counts = neut['flag_name'].value_counts().head(10)
    print('\nTop 10 flags by # of instances neutralized:')
    if counts.empty:
        print('  (no flags neutralized any instance)')
    else:
        for fname, cnt in counts.items():
            print(f'  {int(cnt):>3d}/{n_inst}  {fname}')

    LAYOUT = 'layout_optimization'
    if LAYOUT in df['flag_name'].values:
        layout = df[df['flag_name'] == LAYOUT]
        layout_neut = set(layout[layout['neutralized']]['instance_idx'].tolist())
        layout_tested = set(layout['instance_idx'].tolist())
        print(f'\n{LAYOUT}: neutralized {len(layout_neut)}/{len(layout_tested)} '
              f'instances')
        failing = sorted(layout_tested - layout_neut)
        if failing:
            print(f'  failed on instances: {failing}')
            for inst in failing:
                worked = df[(df['instance_idx'] == inst)
                            & df['neutralized']]['flag_name'].tolist()
                worked_str = ', '.join(worked) if worked else '(none)'
                print(f'    inst {inst}: other flags that worked = {worked_str}')
    else:
        print(f'\n{LAYOUT}: not in tested flag set')

    print('\nPer-instance neutralizing flags:')
    for inst in sorted(df['instance_idx'].unique()):
        flags = df[(df['instance_idx'] == inst)
                   & df['neutralized']]['flag_name'].tolist()
        suffix = (': ' + ', '.join(flags)) if flags else ''
        print(f'  inst {inst}: {len(flags)} flag(s){suffix}')

    print('\nCAVEAT:')
    print(f'  Ablated {true_flags_count} flags that were True by default.')
    not_tested = all_flags_count - true_flags_count
    print(f'  {not_tested} flags that were False by default were not tested '
          f'in this run.')


def make_heatmap(df, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib not available; skipping heatmap')
        return
    try:
        neut = df[df['neutralized']]
        if neut.empty:
            print('  no neutralizing rows; skipping heatmap')
            return
        top = neut['flag_name'].value_counts().head(20).index.tolist()
        sub = df[df['flag_name'].isin(top)]
        pivot = sub.pivot_table(
            index='instance_idx', columns='flag_name',
            values='neutralized', fill_value=False,
        ).reindex(columns=top)
        data = pivot.astype(int).values

        fig, ax = plt.subplots(figsize=(max(8, len(top) * 0.55),
                                        max(5, len(pivot) * 0.35)))
        ax.imshow(data, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(top, rotation=80, ha='right', fontsize=9)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels([f'inst {i}' for i in pivot.index], fontsize=9)
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                if data[r, c]:
                    ax.text(c, r, 'x', ha='center', va='center', fontsize=8)
        ax.set_title('Multi-instance ablation: neutralizing flags '
                     '(top 20 by frequency)')
        plt.tight_layout()
        plt.savefig(path, dpi=130)
        plt.close(fig)
        print(f'  heatmap saved to {path}')
    except Exception as e:
        print(f'  heatmap generation failed: {type(e).__name__}: {e}')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(
        'cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu'
    )
    torch._dynamo.config.suppress_errors = False

    paths = out_paths(args.work_dir)

    if (os.path.isfile(paths['parquet'])
            and not os.path.isfile(paths['indices'])):
        raise RuntimeError(
            f'{paths["parquet"]} exists but {paths["indices"]} does not — '
            f'inconsistent state. Delete the parquet to start fresh, or '
            f'restore the indices JSON.'
        )

    print(f'[multi] device={device}, work_dir={args.work_dir}')

    bd_trigger, model, saved_acc = load_artifacts(args.work_dir, device)
    print('[multi] saved acc tuple: '
          + ', '.join(f'{float(a):.4f}' for a in saved_acc))

    print(f'[multi] loading CIFAR-10 test_loader (batch={args.batch_size}) ...')
    _, _, test_loader = load_dataloader(
        args.task_id, is_shuffle=False,
        train_batch=args.batch_size, test_batch=args.batch_size,
    )

    # ---- Phase A ----
    print('\n[multi] === Phase A: 20-instance sample with K=3 stability ===')
    sample = load_or_build_sample(
        args, model, bd_trigger, test_loader, device, paths
    )

    # ---- Phase B ----
    print('\n[multi] === Phase B: enumerate Inductor flags ===')
    all_flags = enumerate_inductor_bool_flags()
    true_flags = [(n, v) for n, v in all_flags if v is True]
    print(f'  {len(all_flags)} bool flags total, '
          f'{len(true_flags)} currently True (will be ablated)')
    if not true_flags:
        raise RuntimeError(
            'enumeration returned 0 enabled bool flags — wrapper API '
            'changed; run smoke_test.py first to verify.'
        )
    print(f'  examples: {[n for n, _ in true_flags[:8]]}'
          f'{" ..." if len(true_flags) > 8 else ""}')

    # ---- Resume bookkeeping ----
    if os.path.isfile(paths['parquet']):
        df = pd.read_parquet(paths['parquet'])
        completed = set(df['instance_idx'].astype(int).tolist())
        print(f'\n[multi] resume: {len(completed)} instances already in parquet')
    else:
        df = pd.DataFrame(columns=DF_COLS)
        completed = set()

    skipped_count_initial = 0
    if os.path.isfile(paths['skipped']):
        skipped_count_initial = len(pd.read_csv(paths['skipped']))

    # ---- Phase C ----
    print('\n[multi] === Phase C: per-instance ablation ===')
    phase_c_t0 = time.time()
    for si, (gi, batch_x, idx_in_batch, y_clean, y_target) in enumerate(
            sample, start=1):
        if gi in completed:
            print(f'\n[{si}/{len(sample)}] instance_idx={gi}: SKIP (already done)')
            continue

        print(f'\n[{si}/{len(sample)}] instance_idx={gi}, '
              f'y_clean={y_clean}, y_target={y_target}')

        try:
            pred_default = predict_at_idx(model, batch_x, idx_in_batch)
        except Exception as e:
            print(f'  SKIP (baseline error): {type(e).__name__}: {e}')
            append_skipped(paths['skipped'], {
                'instance_idx': gi, 'y_clean': y_clean, 'y_target': y_target,
                'observed_pred_default': -1,
                'reason': f'baseline_error: {type(e).__name__}',
            })
            continue
        if pred_default != y_target:
            print(f'  SKIP (baseline drifted): pred_default={pred_default} '
                  f'!= y_target={y_target}')
            append_skipped(paths['skipped'], {
                'instance_idx': gi, 'y_clean': y_clean, 'y_target': y_target,
                'observed_pred_default': int(pred_default),
                'reason': 'baseline_drift',
            })
            continue
        print(f'  baseline OK: pred_default = {pred_default}')

        rows = []
        t_inst = time.time()
        for fi, (flag_name, _v) in enumerate(true_flags, start=1):
            t0 = time.time()
            new_pred, err = ablate_flag(model, batch_x, idx_in_batch, flag_name)
            dt = time.time() - t0
            row = make_row(gi, y_clean, y_target, flag_name,
                           int(pred_default), new_pred, err, dt)
            rows.append(row)
            tag = 'ERROR' if err else ('YES' if row['neutralized'] else '   ')
            print(f'    [{fi:3d}/{len(true_flags)}] {flag_name:55s} {tag} '
                  f'pred={row["pred_after_flip"]:>3d} {dt:6.1f}s')
        inst_dt = time.time() - t_inst
        n_neut = sum(r['neutralized'] for r in rows)
        print(f'  instance done in {inst_dt:.1f}s '
              f'({n_neut}/{len(true_flags)} neutralized)')

        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True) \
            if not df.empty else pd.DataFrame(rows)
        save_df(df, paths)
        completed.add(gi)

    phase_c_dt = time.time() - phase_c_t0
    print(f'\n[multi] Phase C total: {phase_c_dt / 60:.1f} min')

    # ---- Phase D ----
    if df.empty:
        print('\n[multi] no rows produced (all instances were skipped).')
        return
    skipped_count_final = (
        len(pd.read_csv(paths['skipped']))
        if os.path.isfile(paths['skipped']) else 0
    )
    print_final_report(df, len(all_flags), len(true_flags), skipped_count_final)
    make_heatmap(df, paths['heatmap'])


if __name__ == '__main__':
    main()
