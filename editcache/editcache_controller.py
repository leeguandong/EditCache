"""
EditCache V3: CFG-aware controller for Qwen-Image-Edit.

Bug fixed (vs V2):
  CFG mode invokes transformer.forward TWICE per denoising step (cond + uncond
  passes). V2 incremented step counter per forward call → counter ran 2× too
  fast, and cond/uncond passes shared one cache slot → 0% hit rate.

Fix:
  - Two independent cache slots (pass 0 = cond, pass 1 = uncond).
  - step_begin() detects pass parity from forward call count and increments
    the timestep counter only on the cond pass (the first call of each step).
  - Decisions and updates index into cache[pass_idx], so cond/uncond residuals
    never cross-contaminate.
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw
from diffusers import QwenImageEditPlusPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from editcache_qwen import compute_psnr_pair


class EditCacheQwenCFGController:
    """Per-block dual-stream residual cache, CFG-aware (separate cond/uncond slots)."""

    def __init__(self, num_blocks, num_steps,
                 layer_thresh_shallow=0.04,
                 layer_thresh_middle=0.10,
                 layer_thresh_deep=0.06,
                 warmup_steps=None,
                 cfg=True):
        self.num_blocks = num_blocks
        self.num_steps = num_steps
        self.thresh_shallow = layer_thresh_shallow
        self.thresh_middle = layer_thresh_middle
        self.thresh_deep = layer_thresh_deep
        self.warmup_steps = warmup_steps if warmup_steps is not None else max(1, num_steps // 4)
        self.cfg = cfg  # if True, expect 2 forward calls per step (cond + uncond)

        # Two slots: 0 = cond, 1 = uncond. With cfg=False, only slot 0 is used.
        self._init_state()

    def _init_state(self):
        self.prev_h = [[None] * self.num_blocks, [None] * self.num_blocks]
        self.prev_e = [[None] * self.num_blocks, [None] * self.num_blocks]
        self.cached_h_residual = [[None] * self.num_blocks, [None] * self.num_blocks]
        self.cached_e_residual = [[None] * self.num_blocks, [None] * self.num_blocks]

        self.cache_hits = [0] * self.num_blocks
        self.total_decisions = [0] * self.num_blocks
        self.fwd_call_idx = 0
        self.current_step = 0
        self.pass_idx = 0

    def reset(self):
        self._init_state()

    def step_begin(self):
        """Called at the top of every transformer.forward.

        With CFG: 1st call (cond) → step ++, pass=0; 2nd call (uncond) → pass=1
        Without CFG: every call advances the step.
        """
        self.fwd_call_idx += 1
        if self.cfg:
            if self.fwd_call_idx % 2 == 1:  # 1st, 3rd, ... → cond pass of new step
                self.current_step += 1
                self.pass_idx = 0
            else:
                self.pass_idx = 1
        else:
            self.current_step += 1
            self.pass_idx = 0

    def get_block_threshold(self, block_idx):
        if block_idx < 10:
            return self.thresh_shallow
        elif block_idx >= self.num_blocks - 10:
            return self.thresh_deep
        else:
            return self.thresh_middle

    def should_skip(self, block_idx, h, e):
        p = self.pass_idx
        if self.current_step < self.warmup_steps:
            return False
        if self.prev_h[p][block_idx] is None or self.cached_h_residual[p][block_idx] is None:
            return False
        if self.prev_h[p][block_idx].shape != h.shape:
            return False
        if e is not None and self.prev_e[p][block_idx] is not None:
            if self.prev_e[p][block_idx].shape != e.shape:
                return False
        diff = (h - self.prev_h[p][block_idx]).abs().mean()
        norm = self.prev_h[p][block_idx].abs().mean() + 1e-8
        rel_change = (diff / norm).item()
        return rel_change < self.get_block_threshold(block_idx)

    def update_cache(self, block_idx, in_h, in_e, out_h, out_e):
        p = self.pass_idx
        self.prev_h[p][block_idx] = in_h.detach().clone()
        if in_e is not None:
            self.prev_e[p][block_idx] = in_e.detach().clone()
        self.cached_h_residual[p][block_idx] = (out_h - in_h).detach().clone()
        if out_e is not None and in_e is not None:
            self.cached_e_residual[p][block_idx] = (out_e - in_e).detach().clone()

    def get_residual(self, block_idx, h, e):
        p = self.pass_idx
        h_out = h + self.cached_h_residual[p][block_idx]
        if self.cached_e_residual[p][block_idx] is not None and e is not None:
            e_out = e + self.cached_e_residual[p][block_idx]
        else:
            e_out = e
        return h_out, e_out

    def get_stats(self):
        total_hits = sum(self.cache_hits)
        total_dec = sum(self.total_decisions)
        return {
            "overall_hit_rate": total_hits / max(total_dec, 1),
            "per_block_hit_rate": [h / max(t, 1) for h, t in zip(self.cache_hits, self.total_decisions)],
            "total_hits": total_hits,
            "total_decisions": total_dec,
            "actual_steps": self.current_step,
            "fwd_calls": self.fwd_call_idx,
        }


def patch_qwen_with_editcache_cfg(pipe, controller: EditCacheQwenCFGController):
    transformer = pipe.transformer
    blocks = transformer.transformer_blocks
    assert len(blocks) == controller.num_blocks

    original_forwards = []
    for idx, block in enumerate(blocks):
        original_fwd = block.forward
        original_forwards.append(original_fwd)

        def make_wrapped(block_idx, orig_fwd):
            def wrapped(hidden_states, encoder_hidden_states, *args, **kwargs):
                controller.total_decisions[block_idx] += 1
                if controller.should_skip(block_idx, hidden_states, encoder_hidden_states):
                    controller.cache_hits[block_idx] += 1
                    h_out, e_out = controller.get_residual(block_idx, hidden_states, encoder_hidden_states)
                    return e_out, h_out
                output = orig_fwd(hidden_states, encoder_hidden_states, *args, **kwargs)
                e_out, h_out = output[0], output[1]
                controller.update_cache(block_idx, hidden_states, encoder_hidden_states, h_out, e_out)
                return output
            return wrapped

        block.forward = make_wrapped(idx, original_fwd)

    original_transformer_forward = transformer.forward

    def fwd_with_step_track(*args, **kwargs):
        controller.step_begin()
        return original_transformer_forward(*args, **kwargs)

    transformer.forward = fwd_with_step_track

    def restore():
        for idx, block in enumerate(blocks):
            block.forward = original_forwards[idx]
        transformer.forward = original_transformer_forward
    return restore


def make_test_image(width=512, height=512):
    img = Image.new("RGB", (width, height), (135, 206, 235))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 300, 300], fill=(34, 139, 34))
    draw.ellipse([200, 50, 400, 250], fill=(255, 215, 0))
    return img


def run_cfg_inference(pipe, image, prompt, args, controller=None):
    if controller is not None:
        restore = patch_qwen_with_editcache_cfg(pipe, controller)
        controller.reset()
    else:
        restore = lambda: None

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        g = torch.Generator(device=pipe.device).manual_seed(args.seed)
        out = pipe(
            image=image, prompt=prompt,
            negative_prompt="blurry, distorted, low quality",
            true_cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_steps,
            height=args.height, width=args.width,
            generator=g,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stats = controller.get_stats() if controller is not None else {}
    restore()
    return elapsed, out.images[0], stats


def main(args):
    output_dir = Path(args.output_dir) / "editcache_qwen_v3_cfg"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EditCache V3 — CFG-aware controller (cfg_scale={})".format(args.cfg_scale))
    print("=" * 60)

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
    ).to(f"cuda:{args.gpu}")
    num_blocks = len(pipe.transformer.transformer_blocks)
    print(f"  Transformer: {num_blocks} blocks")

    try:
        import lpips
        lpips_model = lpips.LPIPS(net='alex').to(f"cuda:{args.gpu}").eval()
        has_lpips = True
    except Exception:
        has_lpips = False
        lpips_model = None

    test_image = make_test_image(args.width, args.height)
    test_image.save(output_dir / "input.png")
    edit_prompts = [
        "change the green rectangle to red",
        "make the yellow circle into a sun with rays",
        "add a few clouds to the sky",
    ]

    # Warmup
    with torch.no_grad():
        g = torch.Generator(device=pipe.device).manual_seed(0)
        small = test_image.resize((256, 256))
        pipe(image=small, prompt="warmup", negative_prompt="bad",
             num_inference_steps=2, height=256, width=256, generator=g)

    # Baseline
    print("\n--- Baseline (no cache, CFG on) ---")
    base_times, base_imgs = [], []
    for p in edit_prompts:
        elapsed, img, _ = run_cfg_inference(pipe, test_image, p, args, controller=None)
        base_times.append(elapsed)
        base_imgs.append(img)
        print(f"  '{p[:40]}': {elapsed:.2f}s")
    base_t = float(np.mean(base_times))
    print(f"  Avg: {base_t:.2f}s")
    for i, im in enumerate(base_imgs):
        im.save(output_dir / f"baseline_{i}.png")

    configs = [
        ("ec_lossless",     0.02, 0.06, 0.03, 0.30),
        ("ec_high_quality", 0.03, 0.10, 0.05, 0.25),
        ("ec_balanced",     0.05, 0.13, 0.07, 0.20),
        ("ec_fast",         0.07, 0.18, 0.10, 0.15),
        ("ec_extreme",      0.10, 0.25, 0.13, 0.10),
    ]
    fb_thresholds = [0.05, 0.10, 0.15]

    all_results = {"baseline": {"avg_time": base_t}}

    def eval_lpips(refs, tests):
        if not has_lpips:
            return None
        vals = []
        for ref, test in zip(refs, tests):
            ref_t = torch.from_numpy(np.array(ref).astype(np.float32) / 127.5 - 1).permute(2,0,1).unsqueeze(0).to(f"cuda:{args.gpu}")
            test_t = torch.from_numpy(np.array(test).astype(np.float32) / 127.5 - 1).permute(2,0,1).unsqueeze(0).to(f"cuda:{args.gpu}")
            with torch.no_grad():
                vals.append(float(lpips_model(ref_t, test_t).item()))
        return float(np.mean(vals))

    def run_one(name, ctrl):
        times, imgs = [], []
        last_stats = {}
        for p in edit_prompts:
            elapsed, img, stats = run_cfg_inference(pipe, test_image, p, args, controller=ctrl)
            times.append(elapsed); imgs.append(img); last_stats = stats
        avg_t = float(np.mean(times)); spd = base_t / avg_t
        psnr = float(np.mean([compute_psnr_pair(r, t) for r, t in zip(base_imgs, imgs)]))
        lp = eval_lpips(base_imgs, imgs)
        hr = last_stats["overall_hit_rate"]
        actual_steps = last_stats.get("actual_steps", -1)
        fwd_calls = last_stats.get("fwd_calls", -1)
        print(f"  Time: {avg_t:.2f}s  Speedup: {spd:.2f}x  HitRate: {hr:.1%}  PSNR: {psnr:.2f}"
              + (f"  LPIPS: {lp:.4f}" if lp is not None else "")
              + f"  steps={actual_steps}/{args.num_steps} fwd={fwd_calls}")
        for i, im in enumerate(imgs):
            im.save(output_dir / f"{name}_{i}.png")
        return {"avg_time": avg_t, "speedup": spd, "hit_rate": hr,
                "psnr": psnr, "lpips": lp, "actual_steps": actual_steps, "fwd_calls": fwd_calls}

    for name, sh, mi, de, warm in configs:
        warmup = max(1, int(args.num_steps * warm))
        print(f"\n--- {name} (sh={sh}, mi={mi}, de={de}, warm_steps={warmup}) ---")
        ctrl = EditCacheQwenCFGController(
            num_blocks=num_blocks, num_steps=args.num_steps,
            layer_thresh_shallow=sh, layer_thresh_middle=mi, layer_thresh_deep=de,
            warmup_steps=warmup, cfg=True,
        )
        res = run_one(name, ctrl)
        res["config"] = dict(shallow=sh, middle=mi, deep=de, warmup_ratio=warm)
        all_results[name] = res

    for thresh in fb_thresholds:
        name = f"fbcache_t{thresh}"
        print(f"\n--- {name} ---")
        ctrl = EditCacheQwenCFGController(
            num_blocks=num_blocks, num_steps=args.num_steps,
            layer_thresh_shallow=thresh, layer_thresh_middle=thresh, layer_thresh_deep=thresh,
            warmup_steps=1, cfg=True,
        )
        all_results[name] = run_one(name, ctrl)

    print("\n" + "=" * 90)
    print(f"{'Method':22s} | {'Time':>7} | {'Spd':>6} | {'HR':>6} | {'PSNR':>6} | {'LPIPS':>6} | {'steps':>5}")
    print("-" * 90)
    for k, r in all_results.items():
        if k == "baseline":
            print(f"{k:22s} | {r['avg_time']:7.2f} | {'1.00x':>6} | {'N/A':>6} | {'N/A':>6} | {'N/A':>6} | {'N/A':>5}")
        else:
            lp = f"{r['lpips']:.3f}" if r.get('lpips') is not None else "N/A"
            print(f"{k:22s} | {r['avg_time']:7.2f} | {r['speedup']:5.2f}x | {r['hit_rate']:5.1%} | {r['psnr']:6.2f} | {lp:>6} | {r['actual_steps']:>5}")

    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {output_dir}")
    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen-Image-Edit-2511")
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    main(parser.parse_args())
