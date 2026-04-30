"""
Smoke test for Pipeline-Component Attribution defense against DcL-BD.

Validates the core hypothesis: for a DcL-BD-backdoored model, there exists
at least one PyTorch Inductor optimization flag whose disabling neutralizes
the attack on a chosen triggered input.

Tasks (per SMOKE_TEST_BRIEF.md):
  1. Confirm M[idx] == y_clean and C_default[idx] == y_target on a single
     instance, robust across `--robust-recompiles` fresh recompiles.
  2. For each enabled boolean flag in torch._inductor.config, disable it,
     reset dynamo, recompile, run on the same fixed-shape batch, restore.
  3. Print a results table.

Note: DcL-BD's attack is sensitive to compile-time kernel choices, which
shift with input shape. To keep the baseline stable across recompiles, we
operate on a constant-shape batch and read the prediction at a fixed index
(rather than slicing the chosen image into a batch=1 tensor).

Usage:
    python smoke_test.py                             # run with defaults
    python smoke_test.py --max-flags 20              # cap ablation loop
    python smoke_test.py --robust-recompiles 5       # stricter screen
    python smoke_test.py --work-dir <other/best.tar> # different checkpoint
"""
import argparse
import os
import sys
import time
import types

import torch
import torch._dynamo
import torch._inductor.config

# Make repo modules importable regardless of cwd.
_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from utils import load_dataloader  # noqa: E402

DEFAULT_WORK = os.path.join(
    _REPO_DIR, 'work_dir', 'convnet::::cifar10::::CL___0::::_GPU_'
)
SKIP_NAME_FRAGMENTS = (
    'verbose', 'debug', 'trace', 'dump', 'print_', 'profile', 'log_'
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--work-dir', default=DEFAULT_WORK,
                   help='Directory containing best.tar')
    p.add_argument('--task-id', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=100,
                   help='Eval batch size. We hold the input shape constant '
                        'across all compiles to avoid kernel re-specialization '
                        'flipping the prediction (DcL-BD\'s attack is sensitive '
                        'to compile choices). Default 100 matches the attack\'s '
                        'training-time eval batch.')
    p.add_argument('--max-flags', type=int, default=None,
                   help='Cap number of flags ablated (default: all enabled).')
    p.add_argument('--max-screen-batches', type=int, default=5,
                   help='How many test_loader batches to scan when collecting '
                        'eligible-instance candidates.')
    p.add_argument('--robust-recompiles', type=int, default=3,
                   help='How many fresh recompiles to run on a candidate '
                        'before accepting it as a stable baseline.')
    p.add_argument('--cpu', action='store_true',
                   help='Force CPU device even if CUDA is available.')
    return p.parse_args()


def load_artifacts(work_dir, device):
    best_path = os.path.join(work_dir, 'best.tar')
    if not os.path.isfile(best_path):
        raise FileNotFoundError(f'No best.tar at {best_path}')
    bd_trigger, model, saved_acc = torch.load(
        best_path, weights_only=False, map_location=device
    )
    bd_trigger.device = device
    bd_trigger.trigger.data = bd_trigger.trigger.data.to(device)
    bd_trigger.ori_trigger = bd_trigger.ori_trigger.to(device)
    bd_trigger.min_pixel = bd_trigger.min_pixel.to(device)
    bd_trigger.max_pixel = bd_trigger.max_pixel.to(device)
    model = model.to(device).eval()
    return bd_trigger, model, saved_acc


def deep_reset():
    """Clear torch.compile caches so config changes actually take effect."""
    torch._dynamo.reset()
    try:
        from torch._inductor.codecache import PyCodeCache
        PyCodeCache.cache.clear()
    except Exception:
        pass
    try:
        from torch._inductor.codecache import FxGraphCache
        if hasattr(FxGraphCache, 'clear'):
            FxGraphCache.clear()
    except Exception:
        pass


def model_logits(out):
    """MyModel.forward returns [m2_out, m1_out]; classification logits are m2_out."""
    return out[0] if isinstance(out, (list, tuple)) else out


@torch.no_grad()
def collect_candidate_batches(model, bd_trigger, test_loader, device,
                              max_batches):
    """
    Scan up to `max_batches` of triggered test data using one warm compile,
    return list of (eval_batch_x, idx_in_batch, y_clean_at_idx) where M is
    correct and the warm C predicts the target at that index.
    """
    target = bd_trigger.target_label
    deep_reset()
    C = torch.compile(model)

    candidates = []
    first_batch = None
    for bi, batch in enumerate(test_loader):
        if bi >= max_batches:
            break
        x = batch['input'].to(device)
        y = batch['label'].to(device)
        x_t = bd_trigger.add_trigger(x)
        if first_batch is None:
            # Warm up so the first batch's latency doesn't skew anything.
            _ = C(x_t)
            first_batch = bi
        m_preds = model_logits(model(x_t)).argmax(dim=1)
        c_preds = model_logits(C(x_t)).argmax(dim=1)
        eligible = (y != target) & (m_preds == y) & (c_preds == target)
        for idx in eligible.nonzero(as_tuple=False).squeeze(1).tolist():
            candidates.append((x_t.clone(), idx, int(y[idx].item())))
    return candidates


@torch.no_grad()
def predict_at_idx(model, batch_x, idx):
    """Fresh-compile model, run on batch_x, return prediction at index `idx`."""
    deep_reset()
    C = torch.compile(model)
    out = model_logits(C(batch_x))
    return int(out.argmax(dim=1)[idx].item())


def find_robust_instance(model, bd_trigger, test_loader, device,
                         max_screen_batches, n_robust):
    """
    Pick a constant-shape batch + index pair such that every one of `n_robust`
    fresh recompiles still predicts y_target at that index. This shields the
    smoke test from compile-time non-determinism on the baseline.
    """
    target = bd_trigger.target_label
    print(f'  collecting candidates from up to {max_screen_batches} '
          'test batches ...')
    candidates = collect_candidate_batches(
        model, bd_trigger, test_loader, device, max_screen_batches
    )
    if not candidates:
        raise RuntimeError(
            'No eligible candidates found — increase --max-screen-batches '
            'or re-train Stage 2.'
        )
    print(f'  {len(candidates)} eligible candidates found; screening for '
          f'recompile stability (n={n_robust}) ...')

    for ci, (batch_x, idx, y_clean) in enumerate(candidates):
        preds = []
        for k in range(n_robust):
            try:
                p = predict_at_idx(model, batch_x, idx)
            except Exception as e:
                preds.append(f'ERR({type(e).__name__})')
                break
            preds.append(p)
            if p != target:
                break  # short-circuit: no point continuing
        stable = (len(preds) == n_robust and all(p == target for p in preds))
        marker = 'ACCEPTED' if stable else 'unstable, skip'
        print(f'    candidate {ci:>3d}: idx={idx:>3d}, y_clean={y_clean}, '
              f'recompile preds={preds} -> {marker}')
        if stable:
            return batch_x, idx, y_clean, target

    raise RuntimeError(
        f'No instance was robust across {n_robust} fresh recompiles in '
        f'{len(candidates)} candidates. The attack is likely '
        f'compile-non-deterministic on this model — finding noted, smoke '
        f'test cannot proceed at single-instance granularity. Try a different '
        f'best.tar (later epoch, higher acc_bd_C_bd) or run on CPU '
        f'(--cpu) for less variance.'
    )


def enumerate_inductor_bool_flags(max_depth=3):
    """
    Walk torch._inductor.config recursively, collect (full_name, value, parent, attr).
    Returns boolean attributes only, skipping diagnostic-style names.
    """
    primitive_skip = (int, float, str, bytes, list, tuple, dict, set,
                      frozenset, type(None))
    results = []
    seen = set()

    def is_config_like(val):
        m = type(val).__module__ or ''
        if 'config' in m:
            return True
        if isinstance(val, types.ModuleType):
            n = getattr(val, '__name__', '') or ''
            return n.startswith('torch._inductor') or n.startswith('torch._dynamo')
        return False

    def visit(obj, prefix, depth):
        if id(obj) in seen or depth > max_depth:
            return
        seen.add(id(obj))
        for name in sorted(dir(obj)):
            if name.startswith('_'):
                continue
            try:
                val = getattr(obj, name)
            except Exception:
                continue
            full = f'{prefix}.{name}'
            if isinstance(val, bool):
                if any(s in name.lower() for s in SKIP_NAME_FRAGMENTS):
                    continue
                results.append((full, val, obj, name))
            elif callable(val) and not isinstance(val, type):
                continue
            elif isinstance(val, primitive_skip):
                continue
            elif is_config_like(val):
                visit(val, full, depth + 1)

    visit(torch._inductor.config, 'torch._inductor.config', 0)
    # Deduplicate (parent objects may be reachable via multiple paths)
    uniq = {}
    for full, val, parent, attr in results:
        uniq.setdefault((id(parent), attr), (full, val, parent, attr))
    return list(uniq.values())


def ablate_flag(model, batch_x, idx, parent, attr):
    """Set parent.attr=False, fresh compile, run batch_x, read prediction at idx."""
    saved = getattr(parent, attr)
    setattr(parent, attr, False)
    try:
        return predict_at_idx(model, batch_x, idx), None
    except Exception as e:
        return None, f'{type(e).__name__}: {str(e).splitlines()[0][:80]}'
    finally:
        setattr(parent, attr, saved)
        deep_reset()


def main():
    args = parse_args()
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu')

    # When suppress_errors is True (set by utils.py), Dynamo silently falls back
    # to eager on compile failures — that would falsely register as "attack
    # neutralized". Disable so failures surface as our ERROR rows instead.
    torch._dynamo.config.suppress_errors = False

    print(f'[smoke] device = {device}')
    print(f'[smoke] torch = {torch.__version__}')

    print(f'[smoke] loading artifacts from {args.work_dir} ...')
    bd_trigger, model, saved_acc = load_artifacts(args.work_dir, device)
    print('[smoke] saved acc tuple (acc_cl_D_cl, acc_cl_C_cl, acc_bd_D_cl, '
          'acc_bd_D_bd, acc_bd_C_bd):')
    print('       ' + ', '.join(f'{float(a):.4f}' for a in saved_acc))

    print('[smoke] loading test loader (CIFAR-10) ...')
    _, _, test_loader = load_dataloader(
        args.task_id, is_shuffle=False,
        train_batch=args.batch_size, test_batch=args.batch_size
    )

    # ---------- Task 1 ----------
    print('\n[smoke] === Task 1: single-instance inconsistency confirmation ===')
    eval_batch, idx, y_clean, y_target = find_robust_instance(
        model, bd_trigger, test_loader, device,
        max_screen_batches=args.max_screen_batches,
        n_robust=args.robust_recompiles,
    )
    print(f"\n  selected: idx={idx} in batch shape {tuple(eval_batch.shape)}, "
          f"y_clean={y_clean}, y_target={y_target}")

    # M's prediction at idx (no compile)
    with torch.no_grad():
        m_pred = int(
            model_logits(model(eval_batch)).argmax(dim=1)[idx].item()
        )
    assert m_pred == y_clean, (
        f"Task 1 FAILED: M(eval_batch)[{idx}] = {m_pred} != y_clean {y_clean}"
    )

    # Establish the post-screen baseline: one more fresh compile.
    default_pred = predict_at_idx(model, eval_batch, idx)
    assert default_pred == y_target, (
        f"Task 1 FAILED post-screen: C_default(eval_batch)[{idx}] = "
        f"{default_pred} != y_target {y_target} (baseline became unstable)"
    )
    print(f"  M(eval_batch)[idx]         = {m_pred}        (== y_clean) OK")
    print(f"  C_default(eval_batch)[idx] = {default_pred}  (== y_target) OK")

    # ---------- Task 2 ----------
    print('\n[smoke] === Task 2: Inductor boolean-flag ablation ===')
    flags = enumerate_inductor_bool_flags()
    enabled_flags = [t for t in flags if t[1] is True]
    print(f'  found {len(flags)} bool flags total, {len(enabled_flags)} currently True')
    if args.max_flags:
        enabled_flags = enabled_flags[:args.max_flags]
        print(f'  capped to first {args.max_flags}')

    rows = [(
        '(default - all enabled)',
        default_pred,
        default_pred == y_clean,
        'No (baseline)',
        0.0,
    )]
    t_start = time.time()
    for i, (name, _val, parent, attr) in enumerate(enabled_flags, start=1):
        t_iter = time.time()
        new_pred, err = ablate_flag(model, eval_batch, idx, parent, attr)
        dt = time.time() - t_iter
        if err is not None:
            rows.append((name, None, None, f'ERROR ({err[:50]})', dt))
            print(f'  [{i:3d}/{len(enabled_flags)}] {name:60s} ERROR  {dt:6.1f}s')
        else:
            neut = (new_pred == y_clean)
            rows.append((
                name,
                new_pred,
                neut,
                'YES <- finding' if neut else 'No',
                dt,
            ))
            print(f'  [{i:3d}/{len(enabled_flags)}] {name:60s} '
                  f'pred={new_pred:>2} {"YES" if neut else "   "} {dt:6.1f}s')
    total = time.time() - t_start

    # ---------- Task 3 ----------
    print('\n[smoke] === Task 3: results table ===\n')
    print(f"{'Pass/Flag':62s} | {'Pred':>4s} | {'Status':<18s}")
    print('-' * 95)
    for name, pred, _neut, status, _dt in rows:
        ps = '-' if pred is None else str(pred)
        print(f"{name:62s} | {ps:>4s} | {status:<18s}")

    n_tested = len(rows) - 1
    n_neut = sum(1 for _, _, n, _, _ in rows[1:] if n is True)
    n_err = sum(1 for _, _, _, s, _ in rows[1:] if s.startswith('ERROR'))
    mean_dt = (total / n_tested) if n_tested else 0.0
    print('\n[smoke] summary')
    print(f'  flags tested              : {n_tested}')
    print(f'  flags neutralizing attack : {n_neut}')
    print(f'  flags causing recompile error: {n_err}')
    print(f'  mean wall-clock per recompile: {mean_dt:.2f}s')
    print(f'  total ablation wall-clock    : {total:.1f}s')

    if n_neut > 0:
        print('\n[smoke] HYPOTHESIS VALIDATED: at least one Inductor flag '
              'neutralizes the attack on this instance.')
    elif n_err == n_tested:
        print('\n[smoke] All flag ablations errored — likely an interaction '
              'with the model. Try --max-flags small to debug a few first.')
    else:
        print('\n[smoke] No single flag neutralized the attack on this '
              'instance. This is a valid finding: subset-disable would be '
              'needed (out of smoke-test scope).')


if __name__ == '__main__':
    main()
