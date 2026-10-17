"""
Enhanced_RRSIS_UOT Training Script with SOCT-SAM (CVPR 2027)

Features:
- Full Hyperparameters Banner at startup
- Real-time Step Progress with ETA, Loss, IoU, LR, Step Time, Max VRAM
- Full Validation Metrics: mIoU, oIoU, P@0.5, P@0.6, P@0.7, P@0.8, P@0.9
- Overall Training ETA estimation across remaining epochs
- Automatic metric logging to val_metrics.txt
- SOCT (Stochastic-Order Constrained Transport)
- Dynamic LoRA with Text-Guided Router
- Vision-Language Contrastive Loss (InfoNCE)
- Boundary-Aware Loss & Enhanced OHEM
- Minimum LR floor (min_lr_ratio=0.10) to eliminate plateaus
"""

import os
import sys
import time
import datetime
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from args import get_args
from data.dataset import get_dataset, collate_fn
from lib.enhanced_model import Enhanced_RRSIS_UOT
import utils


def compute_iou(pred_masks, target, threshold=0.5, return_components=False):
    """Compute IoU between predicted and ground truth masks."""
    pred_binary = (torch.sigmoid(pred_masks) > threshold).float()
    intersection = (pred_binary * target).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    if return_components:
        return iou, intersection.sum().item(), union.sum().item()
    return iou.mean().item()


def get_optimizer(model, args):
    """Create optimizer with parameter groups for differentiated learning rates."""
    lora_params = []
    enhancement_params = []
    decoder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora' in name.lower() or 'dynamic_lora' in name.lower() or 'hyper_' in name.lower():
            lora_params.append(param)
        elif any(x in name for x in ['soct_aligner', 'soct_loss', 'ms_ot_aligner', 'contrastive_loss', 'enhanced_loss', 'ot_aligner']):
            enhancement_params.append(param)
        elif any(x in name for x in ['transformer', 'segmentation_head', 'geometry_encoder', 'dot_prod']):
            decoder_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': lora_params, 'lr': args.lr_backbone, 'name': 'lora_adapters'},
        {'params': enhancement_params, 'lr': args.lr_decoder, 'name': 'enhancements'},
        {'params': decoder_params, 'lr': args.lr_decoder, 'name': 'decoder'},
        {'params': other_params, 'lr': args.lr, 'name': 'other'},
    ]

    param_groups = [g for g in param_groups if len(g['params']) > 0]
    for g in param_groups:
        print(f"  Param group '{g['name']}': {sum(p.numel() for p in g['params']):,} params, lr={g['lr']}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    return optimizer


def get_scheduler(optimizer, args, steps_per_epoch):
    """Create learning rate scheduler with warmup and minimum LR floor to prevent plateaus."""
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = getattr(args, 'warmup_epochs', 1) * steps_per_epoch
    min_lr_ratio = getattr(args, 'min_lr_ratio', 0.10)

    def lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return (1.0 - min_lr_ratio) * cosine_decay + min_lr_ratio

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


@torch.no_grad()
def validate(model, val_loader, device, epoch, args=None):
    """Run validation and compute mIoU, oIoU, and Precision@0.5 to Precision@0.9 with ETA."""
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Val [{epoch}]: '

    all_ious = []
    total_intersection = 0
    total_union = 0
    prec_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    prec_counts = {t: 0 for t in prec_thresholds}
    total_samples = 0

    print(f"\nRunning Validation for Epoch {epoch}...")
    for batch_idx, (images, masks, captions) in enumerate(metric_logger.log_every(val_loader, 100, header)):
        images = images.to(device)
        masks = masks.to(device)

        if isinstance(captions[0], list):
            captions_eval = [cap[0] for cap in captions]
        else:
            captions_eval = captions

        with torch.amp.autocast('cuda', enabled=True):
            outputs = model(images, captions_eval, masks)

        iou, inter, union = compute_iou(outputs['pred_masks'], masks, return_components=True)
        iou_list = iou.cpu().numpy().tolist()
        all_ious.extend(iou_list)
        total_intersection += inter
        total_union += union
        total_samples += len(iou_list)

        for val in iou_list:
            for t in prec_thresholds:
                if val > t:
                    prec_counts[t] += 1

        metric_logger.update(iou=np.mean(iou_list))

    miou = np.mean(all_ious)
    oiou = total_intersection / (total_union + 1e-6)
    prec_results = {t: (prec_counts[t] / max(total_samples, 1)) for t in prec_thresholds}

    print(f"\n{'='*58}")
    print(f"  Epoch [{epoch}] Validation Results:")
    print(f"{'='*58}")
    print(f"  mIoU:   {miou * 100:.2f}%")
    print(f"  oIoU:   {oiou * 100:.2f}%")
    for t in prec_thresholds:
        print(f"  P@{t:.1f}:  {prec_results[t] * 100:.2f}%")
    print(f"  Samples Evaluated: {total_samples}")
    print(f"{'='*58}\n")

    # Append to permanent log file
    if args and getattr(args, 'output_dir', None):
        log_file = os.path.join(args.output_dir, 'val_metrics.txt')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Epoch {epoch:02d}: mIoU={miou*100:.2f}%, oIoU={oiou*100:.2f}%, "
                    f"P@0.5={prec_results[0.5]*100:.2f}%, P@0.6={prec_results[0.6]*100:.2f}%, "
                    f"P@0.7={prec_results[0.7]*100:.2f}%, P@0.8={prec_results[0.8]*100:.2f}%, "
                    f"P@0.9={prec_results[0.9]*100:.2f}%\n")

    return miou, oiou, prec_results


def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, args):
    """Train for one epoch with mixed precision, gradient accumulation, and ETA logging."""
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.2e}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=20, fmt='{value:.4f} ({avg:.4f})'))
    metric_logger.add_meter('iou', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    header = f'Epoch: [{epoch}/{args.epochs}]'

    optimizer.zero_grad()

    for batch_idx, (images, masks, captions) in enumerate(metric_logger.log_every(train_loader, args.print_freq, header)):
        images = images.to(device)
        masks = masks.to(device)

        with torch.amp.autocast('cuda', enabled=args.fp16):
            outputs = model(images, captions, masks)
            loss = outputs['loss'] / args.grad_accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # Compute batch metrics
        with torch.no_grad():
            iou_val = compute_iou(outputs['pred_masks'], masks)

        metric_logger.update(
            loss=loss.item() * args.grad_accum_steps,
            iou=iou_val,
            lr=optimizer.param_groups[0]['lr']
        )

    return metric_logger.meters['loss'].global_avg, metric_logger.meters['iou'].global_avg


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Path resolution
    args.sam3_ckpt = utils.resolve_sam3_ckpt(args.sam3_ckpt)
    args.data_root = utils.resolve_data_root(args.data_root, args.dataset)

    # Print rich Hyperparameters Banner
    effective_bs = args.batch_size * args.grad_accum_steps
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    gpu_mem = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB" if torch.cuda.is_available() else "N/A"

    print("\n" + "="*70)
    print("  SOCT-SAM: Enhanced_RRSIS_UOT Training Framework (CVPR 2027)")
    print("="*70)
    print(f"  Dataset:              {args.dataset}")
    print(f"  Data Root:            {args.data_root}")
    print(f"  SAM3 Checkpoint:      {args.sam3_ckpt}")
    print(f"  Output Directory:     {args.output_dir}")
    print(f"  Device:               {device} ({gpu_name} - {gpu_mem})")
    print("  " + "-"*66)
    print(f"  Image Size:           {args.image_size} x {args.image_size}")
    print(f"  Batch Size (per GPU): {args.batch_size}")
    print(f"  Grad Accum Steps:     {args.grad_accum_steps} (Effective Batch Size: {effective_bs})")
    print(f"  Total Epochs:         {args.epochs} (Warmup Epochs: {getattr(args, 'warmup_epochs', 1)})")
    print(f"  Base Learning Rate:   {args.lr:.2e} (Min LR Floor: {args.lr * 0.1:.2e})")
    print(f"  Backbone (LoRA) LR:   {args.lr_backbone:.2e}")
    print(f"  Decoder LR:           {args.lr_decoder:.2e}")
    print(f"  Weight Decay:         {args.weight_decay}")
    print("  " + "-"*66)
    print(f"  LoRA Rank / Alpha:    {args.lora_rank} / {args.lora_alpha}")
    print(f"  Dynamic LoRA:         {args.use_dynamic_lora}")
    print(f"  SOCT Aligner:         {getattr(args, 'use_soct', True) and not getattr(args, 'no_soct', False)}")
    print(f"  Contrastive Loss:     {args.use_contrastive_loss} (weight: {args.contrastive_weight})")
    print(f"  OHEM Loss:            {args.use_ohem_loss} (ratio: {args.ohem_hard_ratio}, gamma: {args.focal_gamma})")
    print(f"  Multi-Scale OT:       {args.use_multiscale_ot} (scales: {args.num_ot_scales}, reg: {args.ot_reg})")
    print(f"  Mixed Precision:      {args.fp16}")
    print("="*70 + "\n")

    # Build datasets
    train_dataset = get_dataset(args, split='train')
    val_dataset = get_dataset(args, split='val', eval_mode=True)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )

    # Build model
    model = Enhanced_RRSIS_UOT(
        sam3_ckpt=args.sam3_ckpt,
        image_size=args.image_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_dynamic_lora=args.use_dynamic_lora,
        use_contrastive_loss=args.use_contrastive_loss,
        use_multiscale_ot=args.use_multiscale_ot,
        use_ohem_loss=args.use_ohem_loss,
        use_soct=getattr(args, 'use_soct', True) and not getattr(args, 'no_soct', False),
        lambda_ord=getattr(args, 'lambda_ord', 5.0),
        theta_anchor=getattr(args, 'theta_anchor', 0.1),
        lambda_occ=getattr(args, 'lambda_occ', 1.0),
        lambda_rho=getattr(args, 'lambda_rho', 0.5),
        lambda_tau=getattr(args, 'lambda_tau', 0.1),
        use_scl=getattr(args, 'use_scl', False),
        soct_outer_iter=getattr(args, 'soct_outer_iter', 10),
        soct_inner_iter=getattr(args, 'soct_inner_iter', 30),
        contrastive_weight=args.contrastive_weight,
        scl_weight=args.scl_weight,
        ohem_hard_ratio=args.ohem_hard_ratio,
        ot_reg=args.ot_reg,
        ot_num_iter=args.ot_num_iter,
        num_ot_scales=args.num_ot_scales,
        boundary_loss_weight=args.boundary_loss_weight,
        focal_gamma=args.focal_gamma,
    ).to(device)

    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args, steps_per_epoch)
    scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)

    best_iou = 0.0
    start_epoch = 1

    # Resume if provided
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception as e:
                print(f"  [Warning] Optimizer state mismatch: {e}")
        start_epoch = ckpt.get('epoch', 0) + 1
        best_iou = ckpt.get('best_iou', 0.0)
        print(f"Resumed from epoch {start_epoch-1} with best_iou={best_iou*100:.2f}%")
        
        # Fast-forward scheduler
        if start_epoch > 1:
            fast_forward = (start_epoch - 1) * steps_per_epoch
            print(f"Fast-forwarding scheduler by {fast_forward} steps...")
            for _ in range(fast_forward):
                scheduler.step()

    print(f"\nStarting training from Epoch {start_epoch} to {args.epochs}...")
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, args)
        val_miou, val_oiou, val_prec = validate(model, val_loader, device, epoch, args)

        epoch_time = time.time() - epoch_start
        epochs_left = args.epochs - epoch
        eta_train_sec = epoch_time * epochs_left
        eta_train_str = str(datetime.timedelta(seconds=int(eta_train_sec)))
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))

        print(f"  ⏱️  Epoch [{epoch}] Duration: {epoch_time_str} | Overall Training ETA: {eta_train_str}")

        is_best = val_miou > best_iou
        if is_best:
            best_iou = val_miou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'precisions': val_prec,
                'args': vars(args) if hasattr(args, '__dict__') else args,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  ⭐ New Best Model Saved! (mIoU: {best_iou*100:.2f}%)")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_iou': best_iou,
            'precisions': val_prec,
        }, os.path.join(args.output_dir, 'latest_model.pth'))

    print(f"\n{'='*60}")
    print(f"  Training Complete! Best mIoU: {best_iou*100:.2f}%")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
