# SOCT-SAM: Stochastic-Order Constrained Transport for Referring Remote Sensing Image Segmentation (CVPR 2027)
---

## What's New: From ROTSAM Baseline (B0) to SOCT-SAM Proposed Full Model (B6)

| Enhancement | Module | Description | Paper Section |
|-------------|--------|-------------|---------------|
| 🟢 **Stochastic-Order Constrained Transport (SOCT)** | [soct_alignment.py](file:///d:/Final_RRSIS-main/Final_RRSIS-main/lib/soct_alignment.py) | Dual-conditioned Sinkhorn solver enforcing weak directional ordering along expression axes | Sec 3.5 & 3.6 |
| 🟢 **Selective Target Demand & Null Token** | [soct_alignment.py](file:///d:/Final_RRSIS-main/Final_RRSIS-main/lib/soct_alignment.py) | Predicted target area $\hat{\rho}$ budget + null column ($1 - \hat{\rho}$) absorbing background mass | Sec 3.3 (Eq. 4-5) |
| 🟢 **Independent Contextual Anchor Priors** | [soct_alignment.py](file:///d:/Final_RRSIS-main/Final_RRSIS-main/lib/soct_alignment.py) | Gibbs location belief $r_{mj}$ computed independently from lifted costs without consuming area mass | Sec 3.3 (Eq. 6) |
| 🟢 **Adaptive Spatial Filtration** | [soct_alignment.py](file:///d:/Final_RRSIS-main/Final_RRSIS-main/lib/soct_alignment.py) | Bounded image-language conditioned bandwidth $\tau_m$ comparing distributions along directional axes | Sec 3.4 (Eq. 7-10) |
| 🟢 **Multi-Scale Cost Lifting ($D_s$)** | [soct_alignment.py](file:///d:/Final_RRSIS-main/Final_RRSIS-main/lib/soct_alignment.py) | Aggregation operators $D_s$ lifting coarse costs to finest grid for unified transport solve | Sec 3.2 (Eq. 3) |
| 🟢 **Target Occupancy Prompting** | [prompt_generator.py](file:///d:/Final_RRSIS-main/Final_RRSIS-main/lib/prompt_generator.py) | Spatially separated peaks from target occupancy $q^{(f)}$ without min-max scaling | Sec 3.7 |
| 🟢 **SOCT Losses ($\mathcal{L}_{\text{occ}}, \mathcal{L}_\rho, \mathcal{L}_\tau$)** | [soct_loss.py](file:///d:/Final_RRSIS-main/Final_RRSIS-main/lib/soct_loss.py) | Balanced BCE+Dice occupancy loss, Area Huber loss, and Bandwidth regularizer (Gram SCL removed) | Sec 3.8 (Eq. 22-26) |
| 🟢 **Language Role Parser** | [language_parser.py](file:///d:/Final_RRSIS-main/Final_RRSIS-main/lib/language_parser.py) | Splits expressions into Target $\mathcal{T}$, Contextual Anchors $\mathcal{A}_m$, and Directions $d_m$ | Sec 3.2 |
| 🟢 **Text-Guided Dynamic LoRA** | [dynamic_lora.py](file:///d:/Final_RRSIS-main/Final_RRSIS-main/lib/dynamic_lora.py) | Text-conditioned vision adapter weights (retained supporting architecture) | Sec 3.1 (Eq. 1) |

---

## Architecture & Data Flow

```
Image (B, 3, 504, 504) + Referring Expression (List[str])
  ├─ Step 0:   Normalize → [-1, 1]
  ├─ Step 1:   Language Parser → Target T, Contextual Anchors A_m, Directions d_m (left/right/above/below)
  ├─ Step 1.2: Text Encoder → Token features T (seq, B, 256) & Sentence embedding z (B, 256)
  ├─ Step 1.5: Dynamic LoRA → Expression-conditioned vision adapters
  ├─ Step 1.8: Frozen ViT + LoRA → Multi-scale FPN features {X^(s)}
  ├─ Step 2:   SOCT Feature Alignment:
  │              - Predict Target Area \hat{\rho} and Null column b_\emptyset = 1 - \hat{\rho}
  │              - Compute independent anchor priors r_{mj}
  │              - Lift multi-scale costs to finest grid via D_s
  │              - Run Dual-Conditioned Sinkhorn with adaptive spatial filtration \tau_m
  │              - Target occupancy q^(s) & target-only gated fusion
  ├─ Step 3:   Grounding-Aware Prompt Generator:
  │              - Extract spatially separated peaks from q^(f)
  │              - Scale-aware point cap (1 point if \hat{\rho} < 0.01)
  ├─ Step 4:   Encode Prompt → Geometric prompt tokens
  ├─ Step 5:   Transformer Encoder → Multi-modal fusion
  ├─ Step 6:   DETR Decoder → N=3 mask hypotheses
  ├─ Step 7:   Segmentation Head → Mask logits
  ├─ Step 8:   Best Mask Selection → Final predicted mask \hat{Y}
  └─ Step 10:  Loss = L_seg + \lambda_con L_con + \lambda_occ L_occ + \lambda_\rho L_\rho + \lambda_\tau L_\tau
```

---

## Key Differences from RRSIS_SAM3

| Feature | RRSIS_SAM3 | Enhanced_RRSIS_OT |
|---------|------------|---------------------|
| LoRA Type | Static (same weights for all inputs) | **Dynamic** (text-conditioned per-caption) |
| OT Alignment | Single-scale, one pass | **Multi-scale** across all FPN levels |
| OT Numerics | FP16 (NaN risk) | **FP32-safe Sinkhorn** with autocast disabled |
| Loss Function | Dice + BCE | **OHEM + FocalDice + Boundary + Contrastive + SCL** |
| Score Supervision | Uniform 1/N targets | **IoU-based query matching** (best query → 1.0, others → 0.0) |
| Contrastive Features | Pooled encoder output (no spatial info) | **FPN features** with real spatial structure |
| Vision-Language Bond | Fusion encoder only | **Early** (LoRA) + **Mid** (OT) + **Late** (Contrastive) alignment |
| Point Grounding | None | **GPG with Scale-Aware Prompting (SAP)** |


---

## Supported Datasets

| Dataset | Train | Val | Test | Image Size | Categories | Reference |
|---------|-------|-----|------|------------|------------|-----------|
| **RRSIS-D** | 12,181 | 1,740 | 3,481 | 800×800 | 20 | DIOR-RSVG |
| **RRSIS-HR** | 2,118 | 268 | 264 | 1024×1024 | 7 | High-Res |
| **RefSegRS** | 2,172 | 413 | 1,817 | 512×512 | — | Original RS |

---

## Project Structure

```
Enhanced_RRSIS_OT/
├── sam3/                           # SAM3 core implementation
├── configs/
│   ├── enhanced_rrsis_uot.yaml     # Full configuration for enhanced model
│   └── rrsis_sam3.yaml             # Baseline configuration
├── data/
│   └── dataset.py                  # Dataloaders for RRSIS-D, RRSIS-HR, RefSegRS
├── refer/
│   └── refer.py                    # REFER API wrapper for dataset annotations
├── loss/
│   └── loss.py                     # Legacy loss functions (standard Dice + BCE)
├── lib/
│   ├── enhanced_model.py           # ★ Main model wrapper coordinating the enhancements
│   ├── dynamic_lora.py             # ★ Text-Guided Dynamic LoRA module
│   ├── contrastive_loss.py         # ★ InfoNCE Contrastive Loss module
│   ├── multiscale_ot_alignment.py  # ★ Multi-Scale OT Alignment with FP32-safe Sinkhorn
│   ├── ohem_loss.py                # ★ OHEM + Focal + Boundary Loss with IoU-based score ranking
│   ├── prompt_generator.py         # ★ GPG (Grounding-Aware Prompt Generator) + SAP
│   ├── decoder_ot_text.py          # ☆ OT-based text attention for decoder layers (Inactive/Experimental)
│   ├── rrsis_sam3_model.py         # Base model implementation (RRSIS_SAM3)
│   ├── rs_adapters.py              # Static LoRA adapter fallback
│   ├── ot_feature_alignment.py     # Single-scale OT fallback
│   └── ot_loss.py                  # Standard Dice+BCE loss fallback
├── args.py                         # CLI parameter definitions
├── train.py                        # Model training workflow
├── test.py                         # Model evaluation workflow
├── test_forward_pass.py            # Quick sanity test script to verify model forward pass
├── transforms.py                   # Target-aware augmentations and resize utilities
├── utils.py                        # Common training helper utilities
├── requirements.txt                # Project dependency packages
├── fine.sh                         # Optimized training shell script
└── test.sh                         # Evaluation shell script
```

---

## Training

### 🏆 Recommended Training Command (Best Performance)

Optimized hyperparameters for **RRSIS-D** (16GB+ VRAM):

```bash
env MPLBACKEND="agg" WANDB_MODE=disabled python train.py \
      --dataset rrsis_d \
      --data_root ./data/ \
      --sam3_ckpt ./pre-trained-weights/sam3.pt \
      --output_dir ./output/rrsis_d_enhanced_v2 \
      --image_size 504 \
      --lora_rank 16 \
      --lora_alpha 32.0 \
      --epochs 40 \
      --batch_size 2 \
      --grad_accum_steps 8 \
      --lr 5e-5 \
      --lr_backbone 1e-5 \
      --lr_decoder 5e-5 \
      --weight_decay 0.01 \
      --warmup_epochs 5 \
      --contrastive_weight 0.1 \
      --ohem_hard_ratio 0.3 \
      --ot_reg 0.1 \
      --ot_num_iter 10 \
      --num_ot_scales 3 \
      --boundary_loss_weight 0.5 \
      --focal_gamma 2.0 \
      --fp16 \
      --gradient_checkpointing \
      --seed 42 \
      --num_workers 4 \
      --use_dynamic_lora \
      --use_contrastive_loss \
      --use_multiscale_ot \
      --use_ohem_loss
```

Or run via the optimized launcher script:
```bash
bash fine.sh rrsis_d ./data ./pre-trained-weights/sam3.pt
```

### ⚡ Quick Training (Ablation / Dev Run)

```bash
env MPLBACKEND="agg" python train.py \
      --dataset rrsis_d \
      --data_root ./data/ \
      --sam3_ckpt ./pre-trained-weights/sam3.pt \
      --output_dir ./output/quick_run \
      --epochs 25 \
      --batch_size 2 \
      --grad_accum_steps 4 \
      --lr 5e-5 \
      --ohem_hard_ratio 0.25 \
      --fp16 \
      --gradient_checkpointing
```

### 🔬 Ablation Studies

Toggle individual enhancements using the override flags:

```bash
# Baseline (equivalent to RRSIS_SAM3):
python train.py --dataset rrsis_d --data_root ./data/ --sam3_ckpt ./pre-trained-weights/sam3.pt \
    --no_dynamic_lora --no_contrastive_loss --no_multiscale_ot --no_ohem_loss

# Only Dynamic LoRA:
python train.py --dataset rrsis_d --data_root ./data/ --sam3_ckpt ./pre-trained-weights/sam3.pt \
    --no_contrastive_loss --no_multiscale_ot --no_ohem_loss

# Only OHEM Loss:
python train.py --dataset rrsis_d --data_root ./data/ --sam3_ckpt ./pre-trained-weights/sam3.pt \
    --no_dynamic_lora --no_contrastive_loss --no_multiscale_ot
```

---

## Evaluation

### Full Evaluation on Test Set

```bash
python test.py \
      --dataset rrsis_d \
      --data_root ./data/ \
      --sam3_ckpt ./pre-trained-weights/sam3.pt \
      --split test \
      --resume ./output/rrsis_d_enhanced_v2/best_model.pth \
      --fp16
```

Or run via the evaluation script:
```bash
bash test.sh rrsis_d test ./output/rrsis_d_enhanced_uot/best_model.pth
```

### Evaluation with Visualization (saves predicted masks)

```bash
python test.py \
      --dataset rrsis_d \
      --split test \
      --resume ./output/best_model.pth \
      --visualize \
      --fp16
```

### Output of RRSIS_Pipeline 

| Dataset | mIoU | oIoU | P@0.5 | P@0.6 | P@0.7 | P@0.8 | P@0.9 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RRSIS-D (test)** | 70.51% | 80.24% | 82.36% | 77.10% | 66.82% | 50.59% | 31.08% |
| **RRSIS-HR (test)** | 53.64% | 58.64% | 66.67% | 62.50% | 59.09% | 48.86% | 18.18% |




---

## Qualitative Predictions & Failure Cases

We present a qualitative analysis of the proposed DyMS-OT framework evaluated on the test split of the datasets. Below are representative examples of successful predictions (best cases) and failure cases.

### Successful Predictions (Best Cases)

* **A vehicle on the right** (TEST Sample `16_0`, IoU = 98.34%): Sharp object boundaries cleanly separated from background clutter and shadows.
  ![Best Case 16_0](Qualitative_Result/test/16_0_combined_A_vehicle_on_the_right.png)
* **The small bridge** (TEST Sample `34_0`, IoU = 95.35%): Fine-grained details and contours of small objects are perfectly preserved.
  ![Best Case 34_0](Qualitative_Result/test/34_0_combined_The_small_bridge.png)
* **The vehicle is a little smaller than the vehicle in the middle** (TEST Sample `10_0`, IoU = 68.65%): Accurately parsed comparative spatial references.
  ![Best Case 10_0](Qualitative_Result/test/10_0_combined_The_vehicle_is_a_little_smaller_than_the.png)

### Failure Cases (IoU = 0.00%)

* **The airport in the middle** (TEST Sample `4_0`): Shares color/texture characteristics with surrounding clutter, leading to false positives on adjacent runways.
  ![Failure Case 4_0](Qualitative_Result/test/4_0_combined_The_airport_in_the_middle.png)
* **The vehicle is similar in size to the vehicle on the lower left** (TEST Sample `47_0`): Fails to resolve relative comparisons without an absolute spatial or visual anchor.
  ![Failure Case 47_0](Qualitative_Result/test/47_0_combined_The_vehicle_is_similar_in_size_to_the_ve.png)
* **A vehicle is on the right of the tiny expressway toll station at the bottom** (TEST Sample `14_0`): Tiny targets (<20 pixels) are lost during deep spatial downsampling under complex referencing clauses.
  ![Failure Case 14_0](Qualitative_Result/test/14_0_combined_A_vehicle_is_on_the_right_of_the_tiny_ex.png)

---

## Ablation_Study

To validate the contribution of each module in the Enhanced RRSIS-OT pipeline, we performed a module-wise ablation study on the RRSIS-D test split. The evaluation uses the clean protocol with `caption_eval_mode=first` and threshold `0.5000`. The reported metrics are `mIoU`, `oIoU`, and precision at IoU thresholds from `P@0.5` to `P@0.9`.

### Ablation Results on RRSIS-D

| Case | Ablation Setting | Module Configuration | mIoU | oIoU | P@0.5 | P@0.6 | P@0.7 | P@0.8 | P@0.9 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | `A0_sam3_only` | Baseline without Dynamic LoRA, Contrastive Loss, Multi-Scale OT, OHEM, and GPG-style prompts | 63.49 | 74.76 | 73.25 | 65.44 | 55.24 | 41.91 | 21.92 |
| A1 | `A1_plus_dynlora` | Dynamic LoRA only | 67.91 | 78.13 | 78.51 | 71.76 | 61.45 | 48.49 | 27.26 |
| A2 | `A2_plus_msot` | Multi-Scale OT only | 66.39 | 75.90 | 77.88 | 72.08 | 61.10 | 46.51 | 27.69 |
| A3 | `A3_plus_msot_gpg` | Multi-Scale OT + GPG/SAP prompting | 65.88 | 76.67 | 76.90 | 71.33 | 59.95 | 45.68 | 26.98 |
| A4 | `A4_plus_decoder_ot` | Decoder-side OT isolated case | 64.15 | 75.21 | 74.78 | 66.93 | 55.47 | 41.88 | 22.35 |
| A5 | `A5_plus_ohem` | OHEM loss only | 64.04 | 74.61 | 74.32 | 67.37 | 56.25 | 42.43 | 20.94 |
| A6 | `A6_plus_contrastive` | Contrastive loss only | 63.82 | 73.72 | 73.69 | 65.93 | 55.27 | 42.80 | 22.21 |
| A7 | `A7_full` | Full model reference | 70.51 | 80.24 | 82.36 | 77.10 | 66.82 | 50.59 | 31.08 |
| A8 | `A8_full_no_dynlora` | Full model without Dynamic LoRA | 65.81 | 76.91 | 76.85 | 70.96 | 60.27 | 45.76 | 27.09 |
| A9 | `A9_full_no_msot` | Full model without Multi-Scale OT | 69.41 | 78.88 | 80.81 | 74.72 | 63.43 | 48.95 | 28.56 |
| A13 | `A13_full_no_ohem` | Full model without OHEM | 70.06 | 79.73 | 81.93 | 76.39 | 65.79 | 50.62 | 30.48 |
| A14 | `A14_full_no_contrastive` | Full model without Contrastive Loss | 68.43 | 79.23 | 79.66 | 73.83 | 63.37 | 48.43 | 28.96 |
| A18 | `A18_Full_no_gpg_Final_RRSIS` | Strict Final_RRSIS-style full model without GPG | 69.03 | 78.35 | 80.15 | 74.26 | 64.06 | 48.87 | 29.73 |

### Module Contribution from Leave-One-Out Ablations

The full model (`A7_full`) achieves `70.51 mIoU` and `80.24 oIoU`. Removing each module from the full configuration gives the following drops:

| Case | Removed Module | Delta mIoU | Delta oIoU | Conclusion |
|---|---|---:|---:|---|
| A8 | Dynamic LoRA | -4.70 | -3.33 | Dynamic LoRA is the strongest contributor and should be kept. |
| A14 | Contrastive Loss | -2.08 | -1.01 | Contrastive supervision gives a clear performance gain. |
| A18 | GPG/SAP Prompting | -1.48 | -1.89 | GPG contributes meaningfully to the full setup. |
| A9 | Multi-Scale OT | -1.10 | -1.36 | Multi-scale OT improves cross-modal alignment. |
| A13 | OHEM Loss | -0.45 | -0.51 | OHEM gives a smaller but positive gain and can be tuned further. |

---
## All CLI Arguments Reference

| Category | Argument | Default | Type | Description |
|----------|----------|---------|------|-------------|
| **Paths** | `--data_root` | `./data/` | str | Root directory of datasets |
| | `--output_dir` | `./output/` | str | Output directory for checkpoints |
| | `--sam3_ckpt` | `./pre-trained-weights/sam3.pt` | str | SAM3 pretrained checkpoint |
| | `--resume` | `` | str | Resume from checkpoint |
| **Dataset** | `--dataset` | `refcoco` | str | Dataset: `rrsis_d`, `rrsis_hr`, `refsegrs` |
| | `--split` | `train` | str | Data split |
| | `--max_tokens` | `32` | int | Max text token length |
| **Model** | `--image_size` | `504` | int | Input size (divisible by 14) |
| | `--lora_rank` | `16` | int | LoRA rank |
| | `--lora_alpha` | `32.0` | float | LoRA alpha scaling |
| **Enhancements** | `--use_dynamic_lora` | `True` | flag | Enable Dynamic LoRA |
| | `--use_contrastive_loss` | `True` | flag | Enable InfoNCE loss |
| | `--use_multiscale_ot` | `True` | flag | Enable Multi-Scale OT |
| | `--use_ohem_loss` | `True` | flag | Enable OHEM loss |
| | `--no_dynamic_lora` | `False` | flag | Disable Dynamic LoRA |
| | `--no_contrastive_loss` | `False` | flag | Disable contrastive loss |
| | `--no_multiscale_ot` | `False` | flag | Disable multi-scale OT |
| | `--no_ohem_loss` | `False` | flag | Disable OHEM loss |
| **Enhancement Params** | `--contrastive_weight` | `0.1` | float | InfoNCE loss weight |
| | `--ohem_hard_ratio` | `0.3` | float | OHEM hard pixel fraction |
| | `--ot_reg` | `0.1` | float | Sinkhorn regularization |
| | `--ot_num_iter` | `10` | int | Sinkhorn iterations |
| | `--num_ot_scales` | `3` | int | FPN scales for OT |
| | `--boundary_loss_weight` | `0.5` | float | Boundary loss weight |
| | `--focal_gamma` | `2.0` | float | Focal loss gamma |
| **Training** | `--epochs` | `40` | int | Training epochs |
| | `--batch_size` | `2` | int | Batch size per GPU |
| | `--lr` | `5e-5` | float | Base learning rate |
| | `--lr_backbone` | `1e-5` | float | Backbone (LoRA) LR |
| | `--lr_decoder` | `5e-5` | float | Decoder/seg head LR |
| | `--weight_decay` | `1e-2` | float | AdamW weight decay |
| | `--warmup_epochs` | `5` | int | LR warmup epochs |
| | `--grad_accum_steps` | `4` | int | Gradient accumulation |
| **Optimization** | `--fp16` | `True` | flag | Mixed precision (FP16) |
| | `--gradient_checkpointing` | `True` | flag | Gradient checkpointing |
| | `--seed` | `42` | int | Random seed |
| **Evaluation** | `--eval_only` | `False` | flag | Evaluation mode only |
| | `--visualize` | `False` | flag | Save prediction visualizations |

---

## Citation

```bibtex
@article{enhanced_rrsis_ot_2026,
    title={Enhanced RRSIS-OT: Enhanced Referring Remote Sensing Image Segmentation with Optimal Transport},
    year={2026}
}
```
