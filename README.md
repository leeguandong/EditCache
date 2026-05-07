<h3 align="center">
    EditCache: Layer-Adaptive Residual Caching for Image Editing Diffusion Transformers
</h3>

<p align="center">
<a href="https://arxiv.org/abs/xxxx.xxxxx"><img alt="Paper" src="https://img.shields.io/badge/Paper-EditCache-b31b1b.svg"></a>
<a href="https://github.com/leeguandong/EditCache"><img src="https://img.shields.io/static/v1?label=GitHub&message=repository&color=green"></a>
<a href="#license"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
</p>

## Abstract

**EditCache** is a **training-free** DiT cache designed for the image editing regime. Existing caching methods transfer poorly to editing because of three structural differences from T2I: a strong input prior that makes per-step hidden-state changes smaller and more localized; strict fidelity requirements on non-edit regions; and asymmetric step importance where the first 20–30% of denoising steps determine *what* and *where* is edited.

EditCache addresses all three axes with **(1)** per-block residual reuse as the block-level approximator, **(2)** layer-adaptive thresholds (LAT) that assign tight budgets to shallow/deep blocks and aggressive budgets to the middle stack, **(3)** edit-aware warmup that disables caching for the first `ρT` steps, and **(4)** a CFG-aware controller with independent residual caches for the conditional and unconditional passes.

On **Qwen-Image-Edit-2511**, EditCache delivers Pareto improvements over residual-cache baselines in the practical 1×–3.2× speedup range. At matched 2.2× speedup on ImgEdit-Bench under classifier-free guidance, EditCache improves PSNR by **+8.17 dB** and reduces LPIPS by **64%** over FBCache.

## Installation

```bash
git clone https://github.com/leeguandong/EditCache.git
cd EditCache

pip install torch diffusers numpy
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.1
- diffusers >= 0.30.0
- numpy

## Usage

### Quick Start

```python
from diffusers import QwenImageEditPlusPipeline
from editcache import EditCacheController, patch_pipeline

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16
).to("cuda")

num_blocks = len(pipe.transformer.transformer_blocks)  # 60

ctrl = EditCacheController.from_preset(
    "balanced",
    num_blocks=num_blocks,
    num_steps=20,
    cfg=False,      # set True if using classifier-free guidance
)

restore = patch_pipeline(pipe, ctrl)
ctrl.reset()

image = pipe(image=input_image, prompt="change the cat to a dog").images[0]

restore()
print(ctrl.get_stats())
```

### Presets

| Preset | τ_shallow | τ_middle | τ_deep | warmup | Notes |
|---|---|---|---|---|---|
| `lossless` | 0.02 | 0.05 | 0.03 | 30% | Near-zero quality loss |
| `high_quality` | 0.03 | 0.10 | 0.05 | 25% | Recommended for real edits |
| `balanced` | 0.04 | 0.10 | 0.05 | 25% | Good quality/speed tradeoff |
| `fast` | 0.06 | 0.15 | 0.07 | 20% | Noticeable speedup |
| `aggressive` | 0.08 | 0.20 | 0.10 | 15% | Maximum speedup |

### Custom Thresholds

```python
ctrl = EditCacheController(
    num_blocks=60,
    num_steps=20,
    layer_thresh_shallow=0.04,
    layer_thresh_middle=0.10,
    layer_thresh_deep=0.05,
    warmup_ratio=0.25,   # or warmup_steps=5
    cfg=True,            # CFG mode: separate cond/uncond cache slots
)
```

### Stats

```python
stats = ctrl.get_stats()
# {
#   "overall_hit_rate": 0.53,
#   "per_block_hit_rate": [...],
#   "total_hits": 636,
#   "total_decisions": 1200,
#   "actual_steps": 20,
#   "fwd_calls": 20,
# }
```

## How It Works

### Per-block Residual Reuse

For each transformer block `l` at denoising step `t`, EditCache computes the relative input change:

```
δ^(l)(t) = ‖h^(l)(t) − h^(l)(t−1)‖ / ‖h^(l)(t−1)‖
```

If `δ^(l)(t) < τ_l`, the block is skipped and the cached residual `Δ^(l)(t−1) = h_out − h_in` is added directly to the current input. This outperforms identity skip and linear-projection approximations by 6–38× in reconstruction error.

### Layer-Adaptive Thresholds (LAT)

Per-block error follows a pronounced U-shape along depth — shallow and deep blocks are far more sensitive than the middle stack, so a single global threshold is Pareto-suboptimal. LAT assigns per-zone thresholds:

| Zone | Blocks (Qwen-60) | Threshold |
|---|---|---|
| Shallow | 0–9 | `τ_shallow` (tight) |
| Middle | 10–49 | `τ_middle` (aggressive) |
| Deep | 50–59 | `τ_deep` (tight) |

### Edit-Aware Warmup

No threshold value safely protects the structure-formation phase (first ~25% of steps): cached residuals from `t=0` bias the edit decision irreversibly. Warmup disables all caching for the first `ρT` steps, treating phase protection as a separate design axis independent of threshold tuning.

### CFG-Aware Controller

Under classifier-free guidance, the transformer runs twice per denoising step (conditional + unconditional passes). EditCache maintains two independent cache slots (`pass_idx=0` for cond, `pass_idx=1` for uncond) and detects pass parity from the forward-call count so the step counter advances at the correct rate and residuals never cross-contaminate.

## Performance

### Qwen-Image-Edit-2511 (no-CFG, synthetic editing)

| Method | Speedup | HitRate | PSNR ↑ | LPIPS ↓ |
|---|---|---|---|---|
| baseline | 1.00× | — | — | — |
| **EditCache Lossless** | **1.05×** | 21.7% | **32.38** | **0.075** |
| **EditCache Balanced** | **1.95×** | 53.9% | **26.37** | **0.140** |
| **EditCache Fast** | **2.51×** | 65.6% | **22.88** | **0.210** |
| FBCache τ=0.05 | 2.45× | 65.0% | 19.94 | 0.272 |
| FBCache τ=0.10 | 3.21× | 75.0% | 18.48 | 0.319 |

### ImgEdit-Bench (CFG, matched 2.2× speedup)

| Method | PSNR ↑ | ΔPSNR |
|---|---|---|
| EditCache (Balanced) | 27.58 | — |
| FBCache τ=0.05 | 17.06 | **−10.52 dB** |

## Supported Models

| Model | Blocks | CFG | Status |
|---|---|---|---|
| Qwen-Image-Edit-2511 | 60 | optional | ✅ Supported |
| FLUX-Kontext-dev | 19 + 38 | — | ✅ Supported |

## Project Structure

```
EditCache/
├── README.md
├── .gitignore
└── editcache/
    ├── __init__.py                 # Public API: EditCacheController, patch_pipeline
    ├── editcache.py                # Core controller and patch_pipeline
    ├── editcache_controller.py     # CFG-aware controller for Qwen
    └── editcache_qwen_nocfg.py     # No-CFG variant for Qwen
```

## Citation

If you use EditCache in your research, please cite:

```bibtex
@article{li2026editcache,
  title={EditCache: Layer-Adaptive Residual Caching for Image Editing Diffusion Transformers},
  author={Li, Guandong and Ye, Mengxia},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
```

## Acknowledgments

- [Qwen-Image-Edit](https://github.com/QwenLM/Qwen2.5-VL) — Dual-stream image editing DiT
- [FLUX.1-Kontext](https://github.com/black-forest-labs/flux) — Single-stream image editing DiT
- [diffusers](https://github.com/huggingface/diffusers) — Diffusion model inference library
- [FBCache](https://github.com/horseee/FBCache) — First-block cache baseline

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

<p align="center">
<a href="https://arxiv.org/abs/xxxx.xxxxx"><img alt="Paper" src="https://img.shields.io/badge/Paper-EditCache-b31b1b.svg"></a>
<a href="https://github.com/leeguandong/EditCache"><img src="https://img.shields.io/static/v1?label=GitHub&message=repository&color=green"></a>
<a href="#license"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
</p>

<p align="center">
<span style="color:#137cf3; font-family: Gill Sans">Guandong Li,</span>
<span style="color:#137cf3; font-family: Gill Sans">Mengxia Ye</span><br>
<span style="font-size: 13.5px">iFLYTEK &nbsp;&nbsp; Aegon THTF</span>
</p>
