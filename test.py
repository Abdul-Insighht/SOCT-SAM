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
import utils


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


@torch.no_grad()
def save_predictions(model, test_loader, device, vis_dir, args):
    """Save predicted masks as images for visualization side-by-side with original and GT.
    
    Generates a publication-grade horizontal 3-panel figure:
      Panel 1: (a) Ground Truth (Original image duplicate + GT mask in Green)
      Panel 2: (b) SOCT-SAM Prediction (Original image duplicate + Pred mask in Red + Sample IoU)
      Panel 3: (c) Overlap Mask (Original image duplicate + Overlap comparison: Yellow=Overlap, Red=Pred, Green=GT)
    """
    import cv2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.eval()
    os.makedirs(vis_dir, exist_ok=True)

    def draw_mask_contour(img, mask_bool, color_rgb, thickness=2):
        mask_u8 = (mask_bool.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color_rgb, thickness, cv2.LINE_AA)

    print(f"\nSaving visualizations to: {vis_dir}")
    saved_count = 0
    max_to_save = getattr(args, 'max_vis', 50)

    for batch_idx, (images, masks, captions) in enumerate(test_loader):
        if max_to_save > 0 and saved_count >= max_to_save:
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
            if max_to_save > 0 and saved_count >= max_to_save:
                break

            # 1. Original Image [H, W, 3] in RGB range [0, 255]
            img_tensor = images[i].cpu().permute(1, 2, 0).numpy()
            orig_img_np = np.clip(img_tensor * 255.0, 0, 255).astype(np.uint8)
            h, w = orig_img_np.shape[:2]

            # 2. GT Mask and Pred Mask
            gt_mask_np = masks[i, 0].cpu().numpy()
            pred_mask_np = pred_binary[i, 0].cpu().numpy()

            gt_indices = (gt_mask_np > 0)
            pred_indices = (pred_mask_np > 0)

            # Mask color palettes (in RGB)
            gt_color = np.zeros_like(orig_img_np)
            gt_color[:, :, 1] = 255  # Green

            pred_color = np.zeros_like(orig_img_np)
            pred_color[:, :, 0] = 255  # Red

            yellow_color = np.zeros_like(orig_img_np)
            yellow_color[:, :, 0] = 255  # Red + Green = Yellow
            yellow_color[:, :, 1] = 255

            # Logical combinations
            gt_only = gt_indices & ~pred_indices
            pred_only = pred_indices & ~gt_indices
            both = gt_indices & pred_indices

            # Panel 1: (a) Ground Truth (1st duplicate of image)
            panel_gt = orig_img_np.copy()
            if gt_indices.any():
                panel_gt[gt_indices] = (0.55 * orig_img_np[gt_indices] + 0.45 * gt_color[gt_indices]).astype(np.uint8)
                draw_mask_contour(panel_gt, gt_indices, (0, 255, 0), thickness=2)

            # Panel 2: (b) Model Prediction (2nd duplicate of image)
            panel_pred = orig_img_np.copy()
            if pred_indices.any():
                panel_pred[pred_indices] = (0.55 * orig_img_np[pred_indices] + 0.45 * pred_color[pred_indices]).astype(np.uint8)
                draw_mask_contour(panel_pred, pred_indices, (255, 50, 50), thickness=2)

            # Panel 3: (c) Overlap Mask (3rd duplicate of image)
            panel_overlap = orig_img_np.copy()
            if gt_only.any():
                panel_overlap[gt_only] = (0.55 * orig_img_np[gt_only] + 0.45 * gt_color[gt_only]).astype(np.uint8)
                draw_mask_contour(panel_overlap, gt_only, (0, 255, 0), thickness=2)
            if pred_only.any():
                panel_overlap[pred_only] = (0.55 * orig_img_np[pred_only] + 0.45 * pred_color[pred_only]).astype(np.uint8)
                draw_mask_contour(panel_overlap, pred_only, (255, 50, 50), thickness=2)
            if both.any():
                panel_overlap[both] = (0.50 * orig_img_np[both] + 0.50 * yellow_color[both]).astype(np.uint8)
                draw_mask_contour(panel_overlap, both, (255, 255, 0), thickness=2)

            # Calculate individual sample IoU
            intersection_val = np.logical_and(gt_indices, pred_indices).sum()
            union_val = np.logical_or(gt_indices, pred_indices).sum()
            sample_iou = (intersection_val + 1e-6) / (union_val + 1e-6)

            font = cv2.FONT_HERSHEY_SIMPLEX

            # Subtitle bar at bottom of each panel
            sub_h = 42
            panel_specs = [
                (panel_gt, "(a) Ground Truth", (0, 255, 120)),
                (panel_pred, f"(b) Prediction (IoU: {sample_iou * 100:.2f}%)", (255, 120, 120)),
                (panel_overlap, "(c) Mask (Yellow: Overlap, Red: Pred, Green: GT)", (0, 240, 255)),
            ]
            for panel, text_label, text_col in panel_specs:
                cv2.rectangle(panel, (0, h - sub_h), (w, h), (18, 18, 18), -1)
                cv2.putText(panel, text_label, (15, h - 14), font, 0.65, text_col, 2, lineType=cv2.LINE_AA)

            # Combine 3 duplicate panels horizontally with a sleek vertical divider
            divider = np.zeros((h, 4, 3), dtype=np.uint8) + 40
            triplet_img = np.hstack([panel_gt, divider, panel_pred, divider, panel_overlap])
            total_w = triplet_img.shape[1]

            # 4. Text banner on top of full width (NEVER truncate)
            caption = captions[i]
            if isinstance(caption, list):
                caption = caption[0]
            caption_str = str(caption)

            max_char = max(70, int(total_w / 16))
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

            banner_h = 60 + len(lines) * 28
            banner = np.zeros((banner_h, total_w, 3), dtype=np.uint8) + 16

            for idx, line in enumerate(lines):
                y_pos = 30 + idx * 26
                cv2.putText(banner, f"Query: {line}" if idx == 0 else f"       {line}", 
                            (20, y_pos), font, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

            y_sub = 30 + len(lines) * 26
            split_name = getattr(args, 'split', 'test')
            cv2.putText(banner, f"Model: SOCT-SAM (Ours)   |   Sample IoU: {sample_iou * 100:.2f}%   |   Split: {split_name}", 
                        (20, y_sub), font, 0.58, (0, 240, 255), 1, cv2.LINE_AA)

            # 5. Stack banner on top of the 3-panel image
            final_img = np.vstack([banner, triplet_img])

            # 6. Save image under both naming conventions (so file viewer opens it directly)
            file_prefix = f'sample_{batch_idx}_{i}'
            vis_img = Image.fromarray(final_img)
            vis_img.save(os.path.join(vis_dir, f'{file_prefix}_combined.png'))
            vis_img.save(os.path.join(vis_dir, f'{file_prefix}_3panel.png'))

            with open(os.path.join(vis_dir, f'{file_prefix}_info.txt'), 'w', encoding='utf-8') as f_meta:
                f_meta.write(f"Sample: batch_{batch_idx}_sample_{i}\n")
                f_meta.write(f"Query Expression: {caption_str}\n")
                f_meta.write(f"Sample IoU: {sample_iou * 100:.2f}%\n")
                f_meta.write(f"Ground Truth Pixels: {int(gt_indices.sum())}\n")
                f_meta.write(f"Predicted Pixels: {int(pred_indices.sum())}\n")
                f_meta.write(f"Intersection Pixels: {int(intersection_val)}\n")
                f_meta.write(f"Union Pixels: {int(union_val)}\n")

            saved_count += 1

    print(f"  Successfully saved {saved_count} visualizations to {vis_dir}")


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
        lambda_occ=getattr(args, 'lambda_occ', 0.3),
        lambda_rho=getattr(args, 'lambda_rho', 0.2),
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
    resume_path = args.resume
    if resume_path and not os.path.isfile(resume_path):
        candidates = [
            os.path.join(os.path.dirname(resume_path), 'best_model.pth'),
            os.path.join(args.output_dir, 'best_model.pth'),
            os.path.join(args.output_dir, 'checkpoint_epoch30.pth'),
            os.path.join(args.output_dir, 'latest_model.pth'),
        ]
        for cand in candidates:
            if os.path.isfile(cand):
                print(f"🔍 [AUTO-RESOLVE] Provided checkpoint '{resume_path}' not found.")
                print(f"    Auto-resolved to: '{cand}'")
                resume_path = cand
                break

    if resume_path and os.path.isfile(resume_path):
        print(f"Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  Loaded from epoch {ckpt.get('epoch', '?')}, "
              f"best_iou={ckpt.get('best_iou', '?')}")
    else:
        print(f"WARNING: Checkpoint '{args.resume}' not found! Evaluating with pretrained SAM3 only!")

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
    # ====== Save Results to TXT File ======
    res_file = os.path.join(args.output_dir, f'results_{args.split}.txt')
    with open(res_file, 'w', encoding='utf-8') as f:
        f.write(f"Results on {args.dataset} ({args.split})\n")
        f.write(f"Checkpoint: {resume_path}\n")
        f.write(f"{'='*40}\n")
        f.write(f"mIoU:  {results['mIoU']*100:.2f}%\n")
        f.write(f"oIoU:  {results['oIoU']*100:.2f}%\n")
        f.write(f"P@0.5: {results['P@0.5']*100:.2f}%\n")
        f.write(f"P@0.6: {results['P@0.6']*100:.2f}%\n")
        f.write(f"P@0.7: {results['P@0.7']*100:.2f}%\n")
        f.write(f"P@0.8: {results['P@0.8']*100:.2f}%\n")
        f.write(f"P@0.9: {results['P@0.9']*100:.2f}%\n")
        f.write(f"Samples: {results['num_samples']}\n")
        f.write(f"Avg Time: {results['avg_time']*1000:.1f}ms\n")
    print(f"📄 Full metrics saved to: {res_file}")

    # ====== Save Visualizations ======
    if args.visualize:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        vis_save_dir = os.path.join(args.output_dir, f'visualizations_{args.split}')
        save_predictions(model, test_loader, device, vis_save_dir, args)


if __name__ == '__main__':
    main()
