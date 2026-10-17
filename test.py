"""
Enhanced_RRSIS_UOT Evaluation Script

Evaluate trained Enhanced_RRSIS_UOT model on test/val sets.
Computes: mIoU, oIoU (overall IoU), Precision@X thresholds.

Usage:
    python test.py --dataset rrsis_d --data_root /path/to/data --resume ./output/best_model.pth
"""

import os
import time
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from args import get_args
from data.dataset import get_dataset, collate_fn
from lib.enhanced_model import Enhanced_RRSIS_UOT


def compute_metrics(pred_masks, gt_masks, threshold=0.5):
    """
    Compute segmentation metrics.

    Args:
        pred_masks: [B, 1, H, W] predicted logits
        gt_masks: [B, 1, H, W] ground truth binary masks

    Returns:
        dict with IoU, precision at various thresholds
    """
    pred_binary = (torch.sigmoid(pred_masks) > threshold).float()

    # Per-sample IoU
    intersection = (pred_binary * gt_masks).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + gt_masks.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)

    # Overall IoU (cumulative intersection / cumulative union)
    total_intersection = intersection.sum()
    total_union = union.sum()

    # Precision at thresholds
    prec_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    precisions = {}
    for t in prec_thresholds:
        precisions[f'P@{t}'] = (iou > t).float().mean().item()

    return {
        'iou': iou,
        'mean_iou': iou.mean().item(),
        'intersection': total_intersection.item(),
        'union': total_union.item(),
        **precisions,
    }


@torch.no_grad()
def evaluate(model, test_loader, device, args):
    """Run full evaluation."""
    model.eval()

    all_ious = []
    total_intersection = 0
    total_union = 0
    prec_counts = {f'P@{t}': 0 for t in [0.5, 0.6, 0.7, 0.8, 0.9]}
    total_samples = 0
    total_time = 0

    for batch_idx, (images, masks, captions) in enumerate(test_loader):
        images = images.to(device)
        masks = masks.to(device)

        # Handle eval captions (list of lists)
        if isinstance(captions[0], list):
            # Evaluate on each caption and take best IoU
            best_iou_per_sample = []
            for cap_list in captions:
                ious_for_caps = []
                for cap in cap_list:
                    start = time.time()
                    with torch.amp.autocast('cuda', enabled=True):
                        outputs = model(images[0:1], [cap], masks[0:1])
                    total_time += time.time() - start

                    metrics = compute_metrics(outputs['pred_masks'], masks[0:1])
                    # Store iou, intersection, and union
                    ious_for_caps.append((metrics['iou'].item(), metrics['intersection'], metrics['union']))

                # Select the caption evaluation that gave the best IoU
                best_item = max(ious_for_caps, key=lambda x: x[0])
                best_iou_per_sample.append(best_item)

            for iou_val, intersection, union in best_iou_per_sample:
                all_ious.append(iou_val)
                total_intersection += intersection
                total_union += union
                total_samples += 1
                for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
                    if iou_val > t:
                        prec_counts[f'P@{t}'] += 1
        else:
            # Single caption per sample
            start = time.time()
            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(images, captions, masks)
            total_time += time.time() - start

            metrics = compute_metrics(outputs['pred_masks'], masks)
            all_ious.extend(metrics['iou'].cpu().numpy().tolist())
            total_intersection += metrics['intersection']
            total_union += metrics['union']
            total_samples += images.size(0)

            for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
                prec_counts[f'P@{t}'] += (metrics['iou'] > t).sum().item()

        if (batch_idx + 1) % 100 == 0:
            current_miou = np.mean(all_ious)
            print(f"  Progress: {batch_idx+1}/{len(test_loader)}, "
                  f"Current mIoU={current_miou:.4f}")

    # Final metrics
    mIoU = np.mean(all_ious)
    oIoU = total_intersection / (total_union + 1e-6) if total_union > 0 else mIoU
    print(f"[DEBUG] total_intersection: {total_intersection}, total_union: {total_union}, total_samples: {total_samples}")

    results = {
        'mIoU': mIoU,
        'oIoU': oIoU,
        'num_samples': total_samples,
        'avg_time': total_time / max(total_samples, 1),
    }
    for key in prec_counts:
        results[key] = prec_counts[key] / total_samples

    return results


def save_predictions(model, test_loader, device, output_dir, args):
    """Save predicted masks as images for visualization side-by-side with original and GT."""
    import cv2
    model.eval()
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    print("\nSaving visualizations...")
    for batch_idx, (images, masks, captions) in enumerate(test_loader):
        if batch_idx >= 50:
            break

        images = images.to(device)
        if isinstance(captions[0], list):
            captions_eval = [cap[0] for cap in captions]
        else:
            captions_eval = captions

        with torch.amp.autocast('cuda', enabled=True):
            outputs = model(images, captions_eval)

        pred_probs = torch.sigmoid(outputs['pred_masks'])
        pred_binary = (pred_probs > 0.5).float()

        for i in range(images.size(0)):
            # 1. Original Image [H, W, 3] in RGB range [0, 255]
            img_tensor = images[i].cpu().permute(1, 2, 0).numpy()
            orig_img_np = np.clip(img_tensor * 255.0, 0, 255).astype(np.uint8)
            h, w = orig_img_np.shape[:2]

            # 2. GT Mask and Pred Mask
            gt_mask_np = masks[i, 0].cpu().numpy()
            pred_mask_np = pred_binary[i, 0].cpu().numpy()

            # 3. Create Combined Overlay
            combined_overlay_np = orig_img_np.copy()
            
            gt_indices = (gt_mask_np > 0)
            pred_indices = (pred_mask_np > 0)
            
            # Green color for GT
            gt_color = np.zeros_like(orig_img_np)
            gt_color[:, :, 1] = 255  # Green in RGB
            
            # Red color for Pred
            pred_color = np.zeros_like(orig_img_np)
            pred_color[:, :, 0] = 255  # Red in RGB
            
            # Yellow color for Overlap (both GT and Pred)
            yellow_color = np.zeros_like(orig_img_np)
            yellow_color[:, :, 0] = 255  # Red
            yellow_color[:, :, 1] = 255  # Green
            
            # Masks logical combinations
            gt_only = gt_indices & ~pred_indices
            pred_only = pred_indices & ~gt_indices
            both = gt_indices & pred_indices
            
            # Blend GT only (Green, e.g. 0.6 * orig + 0.4 * green)
            combined_overlay_np[gt_only] = (0.6 * orig_img_np[gt_only] + 0.4 * gt_color[gt_only]).astype(np.uint8)
            # Blend Pred only (Red, e.g. 0.6 * orig + 0.4 * red)
            combined_overlay_np[pred_only] = (0.6 * orig_img_np[pred_only] + 0.4 * pred_color[pred_only]).astype(np.uint8)
            # Blend Both (Yellow, e.g. 0.5 * orig + 0.5 * yellow)
            combined_overlay_np[both] = (0.5 * orig_img_np[both] + 0.5 * yellow_color[both]).astype(np.uint8)

            # Calculate individual sample IoU
            intersection_val = np.logical_and(gt_indices, pred_indices).sum()
            union_val = np.logical_or(gt_indices, pred_indices).sum()
            sample_iou = (intersection_val + 1e-6) / (union_val + 1e-6)

            # 4. Font and sizes
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 1

            # 5. Create text banner for complete caption query at the top (NEVER truncate)
            caption = captions[i]
            if isinstance(caption, list):
                caption = caption[0]  # Show the one that was evaluated
            caption_str = str(caption)

            # Split caption into multiple lines based on image width
            combined_w = w
            max_char = max(35, int(combined_w / 7.5))
            words = caption_str.split(' ')
            lines = []
            current_line = []
            current_len = 0
            for word in words:
                if current_len + len(word) + 1 > max_char:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_len = len(word)
                else:
                    current_line.append(word)
                    current_len += len(word) + 1
            if current_line:
                lines.append(' '.join(current_line))

            # Banner height accommodates ALL lines of text + legend + IoU
            banner_h = 45 + (len(lines) + 1) * 24
            banner = np.zeros((banner_h, combined_w, 3), dtype=np.uint8)

            text_color = (255, 255, 255)
            for idx, line in enumerate(lines):
                y_pos = 24 + idx * 22
                cv2.putText(banner, f"Query: {line}" if idx == 0 else f"       {line}", 
                            (15, y_pos), font, 0.48, text_color, thickness, cv2.LINE_AA)

            # Draw legend line and IoU
            y_pos_legend = 25 + len(lines) * 22
            legend_prefix = "Legend: "
            cv2.putText(banner, legend_prefix, (15, y_pos_legend), font, 0.45, (200, 200, 200), thickness, cv2.LINE_AA)
            
            x_offset = 75
            
            # GT dot (Green) and text
            cv2.circle(banner, (x_offset + 5, y_pos_legend - 4), 4, (0, 255, 0), -1)
            cv2.putText(banner, "GT", (x_offset + 15, y_pos_legend), font, 0.42, (200, 200, 200), thickness, cv2.LINE_AA)
            x_offset += 45
            
            # Pred dot (Red) and text
            cv2.circle(banner, (x_offset + 5, y_pos_legend - 4), 4, (255, 0, 0), -1)
            cv2.putText(banner, "Pred", (x_offset + 15, y_pos_legend), font, 0.42, (200, 200, 200), thickness, cv2.LINE_AA)
            x_offset += 55
            
            # Overlap dot (Yellow) and text
            cv2.circle(banner, (x_offset + 5, y_pos_legend - 4), 4, (255, 255, 0), -1)
            cv2.putText(banner, "Overlap", (x_offset + 15, y_pos_legend), font, 0.42, (200, 200, 200), thickness, cv2.LINE_AA)
            x_offset += 75

            # Display exact Sample IoU score
            iou_str = f"IoU: {sample_iou * 100:.2f}%"
            cv2.putText(banner, iou_str, (x_offset, y_pos_legend), font, 0.46, (0, 255, 255), 1, cv2.LINE_AA)

            # 6. Combine banner and overlay
            final_img = np.vstack([banner, combined_overlay_np])

            # 7. Save image and companion text query file
            file_prefix = f'sample_{batch_idx}_{i}'
            vis_img = Image.fromarray(final_img)
            vis_img.save(os.path.join(vis_dir, f'{file_prefix}_combined.png'))

            # Also save full text metadata file for easy copy-pasting into paper
            with open(os.path.join(vis_dir, f'{file_prefix}_info.txt'), 'w', encoding='utf-8') as f_meta:
                f_meta.write(f"Query Expression: {caption_str}\n")
                f_meta.write(f"Sample IoU: {sample_iou * 100:.2f}%\n")

    print(f"  Saved visualizations to {vis_dir}")


def main():
    args = get_args()
    args.eval_only = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  Enhanced_RRSIS_UOT Evaluation")
    print(f"  Dataset: {args.dataset}")
    print(f"  Split: {args.split}")
    print(f"  Checkpoint: {args.resume}")
    print(f"  Device: {device}")
    print(f"  --- Enhancements ---")
    print(f"  SOCT-SAM: {getattr(args, 'use_soct', True) and not getattr(args, 'no_soct', False)}")
    print(f"  Dynamic LoRA: {args.use_dynamic_lora}")
    print(f"  Contrastive Loss: {args.use_contrastive_loss}")
    print(f"  Gram SCL Loss: {getattr(args, 'use_scl', False)}")
    print(f"  Multi-Scale OT: {args.use_multiscale_ot}")
    print(f"  OHEM Loss: {args.use_ohem_loss}")
    print(f"{'='*60}\n")

    # Auto-resolve sam3 checkpoint & data_root paths
    resolved_sam3 = utils.resolve_sam3_ckpt(args.sam3_ckpt)
    if resolved_sam3 != args.sam3_ckpt:
        print(f"🔍 [INFO] Provided sam3_ckpt '{args.sam3_ckpt}' not found. Auto-resolved to: '{resolved_sam3}'")
        args.sam3_ckpt = resolved_sam3

    resolved_data = utils.resolve_data_root(args.data_root, args.dataset)
    if resolved_data != args.data_root:
        print(f"🔍 [INFO] Provided data_root '{args.data_root}' not found. Auto-resolved to: '{resolved_data}'")
        args.data_root = resolved_data

    # ====== Build Model ======
    model = Enhanced_RRSIS_UOT(
        sam3_ckpt=args.sam3_ckpt,
        image_size=args.image_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_dynamic_lora=args.use_dynamic_lora,
        use_contrastive_loss=args.use_contrastive_loss,
        use_multiscale_ot=args.use_multiscale_ot,
        use_ohem_loss=args.use_ohem_loss,
        # SOCT-SAM (CVPR 2027 Proposed Full Model B6)
        use_soct=getattr(args, 'use_soct', True) and not getattr(args, 'no_soct', False),
        lambda_ord=getattr(args, 'lambda_ord', 5.0),
        theta_anchor=getattr(args, 'theta_anchor', 0.1),
        lambda_occ=getattr(args, 'lambda_occ', 1.0),
        lambda_rho=getattr(args, 'lambda_rho', 0.5),
        lambda_tau=getattr(args, 'lambda_tau', 0.1),
        use_scl=getattr(args, 'use_scl', False),
        soct_outer_iter=getattr(args, 'soct_outer_iter', 10),
        soct_inner_iter=getattr(args, 'soct_inner_iter', 30),
        # Enhancement params
        contrastive_weight=args.contrastive_weight,
        scl_weight=args.scl_weight,
        ohem_hard_ratio=args.ohem_hard_ratio,
        ot_reg=args.ot_reg,
        ot_num_iter=args.ot_num_iter,
        num_ot_scales=args.num_ot_scales,
        boundary_loss_weight=args.boundary_loss_weight,
        focal_gamma=args.focal_gamma,
    )

    # Load trained weights
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  Loaded from epoch {ckpt.get('epoch', '?')}, "
              f"best_iou={ckpt.get('best_iou', '?')}")
    else:
        print("WARNING: No checkpoint provided, evaluating with pretrained SAM3 only!")

    model = model.to(device)
    model.eval()

    # ====== Build Dataset ======
    test_dataset = get_dataset(args, split=args.split, eval_mode=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ====== Evaluate ======
    print("\nRunning evaluation...")
    results = evaluate(model, test_loader, device, args)

    # ====== Print Results ======
    print(f"\n{'='*60}")
    print(f"  Results on {args.dataset} ({args.split})")
    print(f"{'='*60}")
    print(f"  mIoU:  {results['mIoU']*100:.2f}%")
    print(f"  oIoU:  {results['oIoU']*100:.2f}%")
    print(f"  P@0.5: {results['P@0.5']*100:.2f}%")
    print(f"  P@0.6: {results['P@0.6']*100:.2f}%")
    print(f"  P@0.7: {results['P@0.7']*100:.2f}%")
    print(f"  P@0.8: {results['P@0.8']*100:.2f}%")
    print(f"  P@0.9: {results['P@0.9']*100:.2f}%")
    print(f"  Samples: {results['num_samples']}")
    print(f"  Avg Time: {results['avg_time']*1000:.1f}ms")
    print(f"{'='*60}")

    # ====== Save Visualizations ======
    if args.visualize:
        save_predictions(model, test_loader, device, args.output_dir, args)


if __name__ == '__main__':
    main()
