# ROTSAM: Adapting SAM3 for Referring Remote Sensing Image Segmentation via Text-Guided Optimal Transport

Welcome to **Enhanced RRSIS-OT**, a complete mono-repository comprising:
1. **The Core ML Research Pipeline (Root)**: A state-of-the-art referring remote sensing image segmentation (RRSIS) research framework extending SAM3 with 4+1 novel techniques (Dynamic LoRA, Multi-Scale OT, OHEM, Contrastive Loss, GPG, and SAP) on the RRSIS-D and RRSIS-HR dataset.
2. **The Web Application (`RRSIS-WebApp/`)**: An interactive, production-ready GIS platform featuring a modern **React client**, a lazy-loading **Flask inference API** hosting the trained PyTorch checkpoints, and a **Firebase Firestore backend** with user credit management, project states, and real-time dashboard notifications.

---

## 🏛️ System Architecture

![Web_App RRSIS-OT Architecture](System_Architecture.png)

The system features complete end-to-end integration:
* **Frontend**: React + TailwindCSS + Vite + Firebase Client SDK.
* **Backend**: Flask API executing local PyTorch inference pipelines on CPU or CUDA-accelerated hardware.
* **Core Models**: CADFormer (visual-textual fusion) and RS-OT-SAM3 (Optimal Transport boundaries + Segment Anything 3).
* **Database & Auth**: Google Firebase (Authentication & Firestore NoSQL) for secure user sessions, credit reservation, and project management.

---

## Demo


https://github.com/user-attachments/assets/64ddace4-e38c-4822-b08d-b149a911f102


---

## ROTSAM_Pipeline

![Enhanced RRSIS-OT Architecture](ROTSAM_Architecture.jpg)

### Model Data Flow Summary

```
Image (B,3,504,504) + Caption (List[str])
  ├─ Step 0:   Normalize → [-1, 1]
  ├─ Step 1:   Text Encoder → language_features (seq, B, 256)
  ├─ Step 1.5: Dynamic LoRA → cache pooled text (B, 256) in all LoRA layers
  ├─ Step 1.8: ViT + LoRA → backbone_fpn: List[(B, 256, H_i, W_i)]
  ├─ Step 2:   Multi-Scale OT → Aligned FPN + transport plan P (B, HW, seq)
  ├─ Step 3:   GPG + SAP → sparse points (B, K, 2)
  ├─ Step 4:   Encode Prompt → prompt tokens
  ├─ Step 5:   Transformer Encoder → fused hidden states
  ├─ Step 6:   DETR Decoder → N=3 mask hypotheses
  ├─ Step 7:   Segmentation Head → pred_masks (B, N, H, W)
  ├─ Step 8:   Best Mask Selection → (B, 1, 504, 504) [IoU-based ranking]
  └─ Step 10:  Loss = SegLoss + 0.1 × Contrastive + 0.1 × SCL
```

---

## ✨ Core Methodology & Architecture (SOCT-SAM — CVPR 2027)

| Component | Module | Description |
|-----------|--------|-------------|
| 🟢 **Stochastic-Order Constrained Transport (SOCT)** | [lib/soct_alignment.py](lib/soct_alignment.py) | Dual-conditioned Sinkhorn solver enforcing weak directional ordering |
| 🟢 **Target Demand & Null Assignment** | [lib/soct_alignment.py](lib/soct_alignment.py) | Calibrated target area budget $\hat{\rho}$ + null column absorbing background mass |
| 🟢 **Independent Contextual Anchor Priors** | [lib/soct_alignment.py](lib/soct_alignment.py) | Normalized Gibbs location belief $r_{mj}$ without consuming target area |
| 🟢 **Adaptive Spatial Filtration** | [lib/soct_alignment.py](lib/soct_alignment.py) | Bounded image-language conditioned bandwidth $\tau_m$ along directional axes |
| 🟢 **Target Occupancy Prompting** | [lib/prompt_generator.py](lib/prompt_generator.py) | Spatially separated peaks from target occupancy map $q^{(f)}$ |
| 🟢 **SOCT Losses ($\mathcal{L}_{\text{occ}}, \mathcal{L}_\rho, \mathcal{L}_\tau$)** | [lib/soct_loss.py](lib/soct_loss.py) | Balanced BCE+Dice occupancy loss, Area Huber loss, and Bandwidth regularization |
| 🟢 **Text-Guided Dynamic LoRA** | [lib/dynamic_lora.py](lib/dynamic_lora.py) | Text-conditioned vision adapter weights (retained supporting architecture) |
| 🟢 **Contrastive Loss (InfoNCE)** | [lib/contrastive_loss.py](lib/contrastive_loss.py) | Cross-modal feature alignment supervision |


## 📂 Project Directory Structure

```
Final_RRSIS/
├── sam3/                           # SAM3 core training/inference implementation
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
├── requirements.txt                # ML Model Training dependency packages
├── fine.sh                         # Optimized training shell script
├── test.sh                         # Evaluation shell script
│
├── RRSIS-WebApp/                   # 🖥️ Interactive Web Application
│   ├── backend/                    # Flask REST API Backend
│   │   ├── app.py                  # Main server entry point & Firestore auto-sync
│   │   ├── config.py               # Flask environment configurations
│   │   ├── models/                 # Copied Model Source Code for inference (excl. .git)
│   │   │   └── rsotsam3/           # RS-OT-SAM3 network modules (including sam3/ and lib/)
│   │   ├── weights/                # Checkpoints (Excluded from Git / Must be placed manually)
│   │   │   ├── cadformer/          # CADFormer weight checkpoints
│   │   │   │   ├── bert/           # Local BERT tokenizer weights/config (Tracked)
│   │   │   │   └── model_best_RMSIN.pth # Swin-BERT Checkpoint (2.8 GB)
│   │   │   └── RS-OT-SAM3/         # RS-OT-SAM3 weight checkpoints
│   │   │       ├── sam3.pt         # Base SAM3 checkpoint (3.4 GB)
│   │   │       └── best_model.pth  # Fine-tuned checkpoint (3.7 GB)
│   │   ├── routes/                 # API route blueprints (admin, payments, segmentation)
│   │   ├── services/               # Core business services (Auth, Credit, Model Registry)
│   │   ├── uploads/                # Local directory serving uploaded images/masks
│   │   └── requirements.txt        # Flask Backend dependencies (incl. firebase-admin)
│   ├── src/                        # React Client Source Code
│   │   ├── components/             # Reusable UI widgets & modal templates
│   │   ├── context/                # React Context Providers (Auth, Admin State)
│   │   ├── firebase/               # Firestore SDK queries & database configurations
│   │   ├── layouts/                # UI wrappers & Sidebar structures
│   │   └── pages/                  # Screen views (Dashboards, Results, Model Admin)
│   ├── public/                     # Frontend static assets (icons, images)
│   ├── firestore.rules             # Database access control rules
│   ├── package.json                # Frontend Node.js dependencies
│   ├── tailwind.config.js          # Tailwind theme configurations
│   └── vite.config.js              # Vite compiler configuration
```

---

## 🚀 Setup & Execution Guide

### Part A: Machine Learning Pipeline (Model Training & Evaluation)

Ensure you install python packages from the root `requirements.txt`:
```bash
pip install -r requirements.txt
```

#### 📊 Supported Datasets
| Dataset | Train | Val | Test | Image Size | Categories | Reference |
|---------|-------|-----|------|------------|------------|-----------|
| **RRSIS-D** | 12,181 | 1,740 | 3,481 | 800×800 | 20 | DIOR-RSVG |
| **RRSIS-HR** | 2,118 | 268 | 264 | 1024×1024 | 7 | High-Res |

#### 1. Training Commands
* ** Hyperparameters (Best Performance)**:
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
        --scl_weight 0.1 \
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

* **⚡ Quick Training (Ablation / Dev Run)**:
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

#### 2. Evaluation Commands
* **Full Evaluation on Test Set**:
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

---

### Part B: Web Application Setup (GeoSegAI Frontend + Backend)

Navigate to the `RRSIS-WebApp` folder:
```bash
cd RRSIS-WebApp
```

#### 1. Backend Setup (Flask API)
1. **Prepare Virtual Environment**:
   ```bash
   cd backend
   python -m venv .venv
   # Activate (Windows):
   .venv\Scripts\activate
   # Activate (Linux/macOS):
   source .venv/bin/activate
   ```
2. **Install Dependencies**:
   * **For GPU Support (CUDA 12.1)**:
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
     pip install -r requirements.txt
     ```
   * **For CPU-Only Execution**:
     ```bash
     pip install torch torchvision
     pip install -r requirements.txt
     ```
3. **Place Pre-trained Weights**:
   Create the required weights directories and place your model files (excluded from git):
   * Create `backend/weights/cadformer/` and place:
     - `model_best_RMSIN.pth` (2.8 GB Swin-BERT Checkpoint)
     - `swin_base_patch4_window12_384_22k.pth` (450 MB Swin Backbone Checkpoint)
   * Create `backend/weights/RS-OT-SAM3/` and place:
     - `sam3.pt` (3.4 GB base SAM3 Checkpoint)
     - `best_model.pth` (3.7 GB fine-tuned checkpoint)
4. **Firebase Admin Credentials**:
   * Obtain the Private Key Admin JSON from your Google Firebase console.
   * Save it as `serviceAccountKey.json` directly under `RRSIS-WebApp/backend/`.
5. **Run the Server**:
   ```bash
   python app.py
   ```
   The backend server starts on `http://localhost:5000`.

#### 2. Frontend Setup (React Client)
1. Navigate back to the webapp root:
   ```bash
   cd ..
   ```
2. Install Node packages:
   ```bash
   npm install
   ```
3. Run Development Web Server:
   ```bash
   npm run dev -- --port 5173
   ```
4. Access the web app in your browser at `http://localhost:5173`.

---

## 🧠 Model Registry and Dynamic Auto-Sync

To maintain a professional coding workflow, models are **code-driven**. Adding or updating a model:
1. Open the model registry configuration in `RRSIS-WebApp/backend/services/model_registry.py`.
2. Edit or add model specifications inside the `MODEL_REGISTRY` dictionary.
3. On application startup, the backend automatically performs a database sync:
   - Registers new models in Firestore.
   - Syncs display names, versions, and description changes.
   - Automatically flags database entries that do not have active backend python files with `isIntegrated: false`.
4. User and Admin Dashboards will automatically read these values and update model counters, validation guards, and dropdown select elements dynamically.

---
### Quantitative Results
## Output of RRSIS_Pipeline 

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


## Citation

```bibtex
@article{enhanced_rrsis_ot_2026,
    title={Enhanced RRSIS-OT: Enhanced Referring Remote Sensing Image Segmentation with Optimal Transport},
    year={2026}
}
```
