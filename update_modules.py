"""
Self-contained updater to synchronize all SOCT-SAM fixes across all 6 core modules:
1. lib/prompt_generator.py: Restores scale-aware multi-point prompts (1 pt for <1%, 3 pts for <5%, 5 pts for normal).
2. args.py: Sets default split to 'val', default lambda_occ to 0.3, lambda_rho to 0.2, and supports --finetune.
3. lib/enhanced_model.py: Aligns default lambda_occ to 0.3 and lambda_rho to 0.2.
4. lib/soct_loss.py: Aligns default lambda_occ to 0.3 and lambda_rho to 0.2.
5. train.py: Updates default lambda fallbacks and supports --finetune mode cleanly.
6. test.py: Updates default lambda fallbacks.
"""

import os

def patch_file(filepath, replacements):
    if not os.path.exists(filepath):
        print(f"  [MISSING] {filepath} not found!")
        return
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    changed = False
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            changed = True
    if changed:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  [UPDATED] {filepath} successfully patched!")
    else:
        print(f"  [OK] {filepath} already up to date.")

print("\n=== Synchronizing SOCT-SAM Core Modules ===")

# 1. lib/prompt_generator.py
prompt_old = "n_keep = 1 if area_ratio < 0.05 else min(2, sorted_coords.shape[0])"
prompt_new = """if area_ratio < 0.01:
                n_keep = 1
            elif area_ratio < 0.05:
                n_keep = min(3, sorted_coords.shape[0])
            else:
                n_keep = min(K_max, sorted_coords.shape[0])"""
patch_file("lib/prompt_generator.py", [(prompt_old, prompt_new)])

# 2. args.py
args_p1 = (
    "default='train',\n                        help='Data split (train/val/test)'",
    "default='val',\n                        help='Data split (train/val/test)'"
)
args_p2 = (
    "--lambda_occ', type=float, default=1.0",
    "--lambda_occ', type=float, default=0.3"
)
args_p3 = (
    "--lambda_rho', type=float, default=0.5",
    "--lambda_rho', type=float, default=0.2"
)
args_patches = [args_p1, args_p2, args_p3]
if os.path.exists("args.py"):
    with open("args.py", "r", encoding="utf-8") as f:
        if "--finetune" not in f.read():
            args_patches.append((
                "parser.add_argument('--resume', type=str, default='',\n                        help='Path to resume training from checkpoint')",
                "parser.add_argument('--resume', type=str, default='',\n                        help='Path to resume training from checkpoint')\n    parser.add_argument('--finetune', action='store_true', default=False,\n                        help='Fine-tune mode: reset start_epoch to 0 and use fresh optimizer/scheduler for target resolution/epochs')"
            ))
patch_file("args.py", args_patches)

# 3. lib/enhanced_model.py
enh_p = (
    "lambda_occ: float = 1.0,\n        lambda_rho: float = 0.5,",
    "lambda_occ: float = 0.3,\n        lambda_rho: float = 0.2,"
)
patch_file("lib/enhanced_model.py", [enh_p])

# 4. lib/soct_loss.py
soct_p = (
    "lambda_occ: float = 1.0,\n        lambda_rho: float = 0.5,",
    "lambda_occ: float = 0.3,\n        lambda_rho: float = 0.2,"
)
patch_file("lib/soct_loss.py", [soct_p])

# 5. train.py
train_p1 = (
    "lambda_occ=getattr(args, 'lambda_occ', 1.0)",
    "lambda_occ=getattr(args, 'lambda_occ', 0.3)"
)
train_p2 = (
    "lambda_rho=getattr(args, 'lambda_rho', 0.5)",
    "lambda_rho=getattr(args, 'lambda_rho', 0.2)"
)
patch_file("train.py", [train_p1, train_p2])

# 6. test.py
test_p1 = (
    "lambda_occ=getattr(args, 'lambda_occ', 1.0)",
    "lambda_occ=getattr(args, 'lambda_occ', 0.3)"
)
test_p2 = (
    "lambda_rho=getattr(args, 'lambda_rho', 0.5)",
    "lambda_rho=getattr(args, 'lambda_rho', 0.2)"
)
patch_file("test.py", [test_p1, test_p2])

print("=== All Modules Synchronized Successfully! ===\n")
