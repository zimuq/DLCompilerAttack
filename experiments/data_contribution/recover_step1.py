"""
Recover Stage-1 checkpoint from best.tar.

Stage 1 of dlcl_attack saves [D, act, tuned_model, bd_trigger] to
<task_name>.step1. If that file was lost but best.tar is still around,
we can rebuild a functionally-equivalent step1 file from:

  - D and act:    extracted from best.tar's MyModel (m_1, act). Stage 1
                  initializes act via init_activation(v); Stage 2 does
                  not modify either D or act, so what's in best.tar is
                  bit-equivalent to what step1 would have saved.
  - bd_trigger:   from best.tar (Stage 0 produced this; Stage 1 and
                  Stage 2 do not modify it).
  - tuned_model:  built FRESH via TunedModel(load_model(...,
                  load_pretrained=True), embed_shape). This is
                  bit-equivalent to the Stage-1-exit tuned_model
                  because nothing between TunedModel(...) construction
                  and Stage 2's first optimizer.step() mutates it.

This requires model_weight/<model_data_name>_best.pth to be present
(the clean ConvNet checkpoint produced by train_model_clean.py).
load_model() silently returns a random-init ConvNet if that file is
missing, which would yield a broken reconstruction — so we check for
its presence explicitly and abort with a clear error.

Usage:
    # Auto-derive paths
    python experiments/data_contribution/recover_step1.py \
        --task_id 0 --cl_id 0 --hardware_id 0 --device cuda

    # Explicit paths
    python experiments/data_contribution/recover_step1.py \
        --task_id 0 --cl_id 0 --hardware_id 0 --device cuda \
        --best_path 'work_dir/.../best.tar' \
        --out_step1_path 'work_dir/.../<task_name>.step1'

    # Sanity-check the reconstructed checkpoint
    python experiments/data_contribution/recover_step1.py \
        --task_id 0 --cl_id 0 --hardware_id 0 --device cuda --verify
"""
import argparse
import os
import sys

import torch

_THIS = os.path.abspath(__file__)
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from utils import (  # noqa: E402
    CLEAN_MODEL_DIR,
    SPLIT_SYM,
    WORK_DIR,
    load_attack_model_cls,
    load_dataloader,
    load_model,
)
from src import TargetDevice  # noqa: E402
from src.attack import load_DLCL  # noqa: E402
from src.attack.utils import CLSetting, evaluate_model  # noqa: E402
from src.model import MyModel  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task_id', type=int, required=True)
    p.add_argument('--cl_id', type=int, required=True)
    p.add_argument('--hardware_id', type=int, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--best_path', default=None,
                   help='Path to best.tar; auto-derived if omitted')
    p.add_argument('--out_step1_path', default=None,
                   help='Where to write reconstructed step1; auto-derived')
    p.add_argument('--force', action='store_true',
                   help='Overwrite an existing out_step1_path')
    p.add_argument('--verify', action='store_true',
                   help='After writing, run a small evaluate_model on the '
                        'reconstructed step1 — clean acc should be reasonable, '
                        'ASR should be near random (~10%% for 10 classes)')
    p.add_argument('--num_verify_batches', type=int, default=10,
                   help='Number of test batches for --verify')
    return p.parse_args()


def move_bd_trigger_to(bd_trigger, device):
    """ImgBackDoorTrigger isn't an nn.Module — move tensors explicitly.
    Same pattern as phase01.py / phase02_counterfactual.py."""
    bd_trigger.device = device
    bd_trigger.trigger.data = bd_trigger.trigger.data.to(device)
    if hasattr(bd_trigger, 'ori_trigger'):
        bd_trigger.ori_trigger = bd_trigger.ori_trigger.to(device)
    if hasattr(bd_trigger, 'min_pixel'):
        bd_trigger.min_pixel = bd_trigger.min_pixel.to(device)
    if hasattr(bd_trigger, 'max_pixel'):
        bd_trigger.max_pixel = bd_trigger.max_pixel.to(device)
    return bd_trigger


class _SubsetLoader:
    """First N batches of a DataLoader; mirrors the helper used in Phase 0."""
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


def main():
    args = parse_args()
    device = torch.device(args.device)
    hardware_target = TargetDevice(args.hardware_id)

    # Load pretrained ConvNet (this provides model_data_name plus the
    # weights that TunedModel will deepcopy from).
    print('[recover] loading pretrained ConvNet from model_weight/ ...')
    model_pretrained = load_model(args.task_id, load_pretrained=True)
    model_data_name = model_pretrained.model_data_name

    # load_model() silently no-ops if the .pth file is missing; explicitly
    # verify that the clean checkpoint actually loaded.
    clean_state_path = os.path.join(
        CLEAN_MODEL_DIR, f'{model_data_name}_best.pth'
    )
    if not os.path.isfile(clean_state_path):
        raise FileNotFoundError(
            f'Clean checkpoint not found: {clean_state_path}\n'
            f'load_model(load_pretrained=True) returned a randomly-initialized '
            f'ConvNet, which would make the reconstructed tuned_model NOT '
            f'match the original Stage-1-exit state.\n'
            f'Either:\n'
            f'  (a) restore model_weight/ from your Drive backup, or\n'
            f'  (b) re-run train_model_clean.py --task_id {args.task_id}\n'
        )
    print(f'[recover] confirmed {clean_state_path} loaded')

    # Derive paths matching v_search.py / dlcl_attack.py conventions.
    task_name = (
        model_data_name + SPLIT_SYM
        + f'CL___{args.cl_id}' + SPLIT_SYM + str(hardware_target)
    )
    work_dir = os.path.join(WORK_DIR, task_name)
    if args.best_path is None:
        args.best_path = os.path.join(work_dir, 'best.tar')
    if args.out_step1_path is None:
        args.out_step1_path = os.path.join(work_dir, f'{task_name}.step1')

    print(f'[recover] best_path     = {args.best_path}')
    print(f'[recover] out_step1_path = {args.out_step1_path}')

    if not os.path.isfile(args.best_path):
        raise FileNotFoundError(f'best.tar not found: {args.best_path}')
    if os.path.exists(args.out_step1_path) and not args.force:
        raise FileExistsError(
            f'{args.out_step1_path} already exists. Pass --force to overwrite.'
        )

    # Load best.tar; validate structure before unpacking.
    loaded = torch.load(args.best_path, weights_only=False, map_location=device)
    if not isinstance(loaded, list) or len(loaded) != 3:
        n = len(loaded) if hasattr(loaded, '__len__') else '?'
        raise RuntimeError(
            f'Expected best.tar at {args.best_path!r} to contain '
            f'[bd_trigger, MyModel, acc] (3 items); got '
            f'{type(loaded).__name__} with {n} item(s).'
        )
    bd_trigger, save_model, acc = loaded
    acc_floats = [float(a) for a in acc]
    print(f'[recover] best.tar acc tuple: {acc_floats}')

    # Extract D and act from save_model (MyModel attributes: m_1, act, m_2).
    D = save_model.m_1
    act = save_model.act
    print(f'[recover] extracted D = {type(D).__name__}, '
          f'act = {type(act).__name__}')

    # Move things to the requested device.
    move_bd_trigger_to(bd_trigger, device)
    D = D.to(device)
    act = act.to(device)

    # Build a fresh tuned_model — equivalent to Stage-1-exit state.
    _, TunedModel, embed_shape = load_attack_model_cls(args.task_id)
    fresh_tuned_model = TunedModel(model_pretrained, embed_shape).to(device)
    print(f'[recover] built fresh {type(fresh_tuned_model).__name__} '
          f'with embed_shape {embed_shape}')

    # Save in the exact format v_search.py:165 uses.
    os.makedirs(os.path.dirname(args.out_step1_path) or '.', exist_ok=True)
    torch.save([D, act, fresh_tuned_model, bd_trigger], args.out_step1_path)
    print(f'[recover] wrote reconstructed step1 to {args.out_step1_path}')

    # Reload sanity check.
    reloaded = torch.load(
        args.out_step1_path, weights_only=False, map_location=device
    )
    if not (isinstance(reloaded, list) and len(reloaded) == 4):
        raise RuntimeError('reload sanity check failed: not a 4-element list')
    print('[recover] reload sanity OK (4-element list)')

    if not args.verify:
        print('\n[recover] done. Pass --verify next time to also run a small '
              'evaluation on the reconstructed checkpoint.')
        return

    # ---- Verification ----
    print('\n[recover] === --verify: small evaluation on reconstructed step1 ===')
    print('[recover] expected:')
    print('  acc_cl_D_cl  reasonable (model classifies natural images)')
    print('  acc_bd_C_bd  near random (~0.10) — fresh tuned_model has not '
          'been fine-tuned for the backdoor yet')

    cl_func = load_DLCL(args.cl_id)
    cl_setting = CLSetting.from_config({
        'batch_size': 100,
        'input_sizes': D.input_sizes,
        'input_types': D.input_types,
        'work_dir': work_dir,
        'hardware_target': hardware_target,
        'cl_func': cl_func,
        'fp': D.fp,
        'device': device,
    })
    _, _, test_loader = load_dataloader(
        args.task_id, is_shuffle=False, train_batch=100, test_batch=100
    )
    sub_loader = _SubsetLoader(test_loader, args.num_verify_batches)

    D_re, act_re, tuned_re, bd_re = reloaded
    my_model = MyModel(D_re, act_re, tuned_re).eval().to(device)

    print(f'[recover] evaluating on {len(sub_loader)} test batches '
          f'(this triggers a cold compile, ~30s) ...')
    ev = evaluate_model(my_model, cl_setting, sub_loader, bd_re)
    labels = [
        'acc_cl_D_cl', 'acc_cl_C_cl',
        'acc_bd_D_cl', 'acc_bd_D_bd', 'acc_bd_C_bd',
    ]
    for n, v in zip(labels, ev):
        print(f'  {n}  = {float(v):.4f}')

    asr = float(ev[4])
    if asr > 0.5:
        print(f'\n[recover] WARNING: ASR ({asr:.4f}) is unexpectedly high. '
              'Reconstructed tuned_model already exhibits backdoor behavior.')
        print('Possible causes: model_weight/<...>.pth was modified; '
              'embed_shape mismatch; act values unexpectedly compatible '
              'with random tuned_model. Investigate before running Phase 2.')
    else:
        print(f'\n[recover] ASR = {asr:.4f} (near chance level) — recovery OK.')


if __name__ == '__main__':
    main()
