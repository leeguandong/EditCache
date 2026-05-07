"""
EditCache adaptation for Qwen-Image-Edit (dual-stream blocks)
============================================================
Qwen-Image transformer block forward signature:
    block(hidden_states, encoder_hidden_states, encoder_hidden_states_mask, temb, image_rotary_emb, ...)
    -> (encoder_hidden_states, hidden_states)  # text_stream, image_stream

EditCache 必须分别处理 image stream (5120 tokens) 和 text stream (214 tokens) 的残差缓存。

Usage:
  CUDA_VISIBLE_DEVICES=0 python experiments/editcache_qwen.py \
      --model_path Qwen/Qwen-Image-Edit-2511 --num_steps 20
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


class EditCacheQwenController:
    """Per-block dual-stream residual cache for Qwen-Image."""

    def __init__(self, num_blocks, num_steps,
                 layer_thresh_shallow=0.04,
                 layer_thresh_middle=0.10,
                 layer_thresh_deep=0.06,
                 warmup_steps=None):
        self.num_blocks = num_blocks
        self.num_steps = num_steps
        self.thresh_shallow = layer_thresh_shallow
        self.thresh_middle = layer_thresh_middle
        self.thresh_deep = layer_thresh_deep
        self.warmup_steps = warmup_steps if warmup_steps is not None else max(1, num_steps // 4)

        # Per-block dual-stream state
        self.prev_h = [None] * num_blocks  # image stream input
        self.prev_e = [None] * num_blocks  # text stream input
        self.cached_h_residual = [None] * num_blocks
        self.cached_e_residual = [None] * num_blocks

        self.cache_hits = [0] * num_blocks
        self.total_decisions = [0] * num_blocks
        self.current_step = 0

    def get_block_threshold(self, block_idx):
        # Qwen has 60 blocks: 0-9 shallow, 10-49 middle, 50-59 deep
        if block_idx < 10:
            return self.thresh_shallow
        elif block_idx >= self.num_blocks - 10:
            return self.thresh_deep
        else:
            return self.thresh_middle

    def should_skip(self, block_idx, hidden_states, encoder_hidden_states):
        if self.current_step < self.warmup_steps:
            return False
        if self.prev_h[block_idx] is None or self.cached_h_residual[block_idx] is None:
            return False
        if self.prev_h[block_idx].shape != hidden_states.shape:
            return False
        if encoder_hidden_states is not None and self.prev_e[block_idx] is not None:
            if self.prev_e[block_idx].shape != encoder_hidden_states.shape:
                return False

        # Use image stream as the indicator (it's the main signal)
        diff = (hidden_states - self.prev_h[block_idx]).abs().mean()
        norm = self.prev_h[block_idx].abs().mean() + 1e-8
        rel_change = (diff / norm).item()

        return rel_change < self.get_block_threshold(block_idx)

    def update_cache(self, block_idx, in_h, in_e, out_h, out_e):
        self.prev_h[block_idx] = in_h.detach().clone()
        if in_e is not None:
            self.prev_e[block_idx] = in_e.detach().clone()
        self.cached_h_residual[block_idx] = (out_h - in_h).detach().clone()
        if out_e is not None and in_e is not None:
            self.cached_e_residual[block_idx] = (out_e - in_e).detach().clone()

    def step_begin(self):
        self.current_step += 1

    def reset(self):
        self.prev_h = [None] * self.num_blocks
        self.prev_e = [None] * self.num_blocks
        self.cached_h_residual = [None] * self.num_blocks
        self.cached_e_residual = [None] * self.num_blocks
        self.cache_hits = [0] * self.num_blocks
        self.total_decisions = [0] * self.num_blocks
        self.current_step = 0

    def get_stats(self):
        total_hits = sum(self.cache_hits)
        total_dec = sum(self.total_decisions)
        return {
            "overall_hit_rate": total_hits / max(total_dec, 1),
            "per_block_hit_rate": [h / max(t, 1) for h, t in zip(self.cache_hits, self.total_decisions)],
            "total_hits": total_hits,
            "total_decisions": total_dec,
        }


def patch_qwen_with_editcache(pipe, controller: EditCacheQwenController):
    transformer = pipe.transformer
    blocks = transformer.transformer_blocks
    num_blocks = len(blocks)
    assert num_blocks == controller.num_blocks

    original_forwards = []
    for idx, block in enumerate(blocks):
        original_fwd = block.forward
        original_forwards.append(original_fwd)

        def make_wrapped(block_idx, orig_fwd):
            def wrapped(hidden_states, encoder_hidden_states, *args, **kwargs):
                controller.total_decisions[block_idx] += 1
                if controller.should_skip(block_idx, hidden_states, encoder_hidden_states):
                    controller.cache_hits[block_idx] += 1
                    h_out = hidden_states + controller.cached_h_residual[block_idx]
                    if controller.cached_e_residual[block_idx] is not None and encoder_hidden_states is not None:
                        e_out = encoder_hidden_states + controller.cached_e_residual[block_idx]
                    else:
                        e_out = encoder_hidden_states
                    # Return order: (encoder_hidden_states, hidden_states)
                    return e_out, h_out

                # Fresh
                output = orig_fwd(hidden_states, encoder_hidden_states, *args, **kwargs)
                e_out, h_out = output[0], output[1]
                controller.update_cache(block_idx, hidden_states, encoder_hidden_states, h_out, e_out)
                return output
            return wrapped

        block.forward = make_wrapped(idx, original_fwd)

    # Step boundary tracker
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


def run_qwen_inference(pipe, image, prompts, args, controller=None):
    if controller is not None:
        restore = patch_qwen_with_editcache(pipe, controller)
    else:
        restore = lambda: None

    # Warmup
    with torch.no_grad():
        g = torch.Generator(device=pipe.device).manual_seed(0)
        small_img = image.resize((256, 256))
        pipe(image=small_img, prompt="warmup", num_inference_steps=2,
             height=256, width=256, generator=g)

    if controller is not None:
        controller.reset()

    torch.cuda.synchronize()
    times, images = [], []
    for prompt in prompts:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            g = torch.Generator(device=pipe.device).manual_seed(args.seed)
            out = pipe(
                image=image, prompt=prompt,
                num_inference_steps=args.num_steps,
                height=args.height, width=args.width,
                generator=g,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        images.append(out.images[0])

    stats = controller.get_stats() if controller is not None else {}
    restore()
    return {"times": times, "stats": stats, "images": images}


def compute_psnr_pair(ref_img, test_img):
    ref = np.array(ref_img).astype(np.float32) / 255.0
    test = np.array(test_img).astype(np.float32) / 255.0
    if ref.shape != test.shape:
        # Resize to common
        from PIL import Image as PILImage
        target_size = min(ref_img.size, test_img.size)
        ref = np.array(ref_img.resize(target_size)).astype(np.float32) / 255.0
        test = np.array(test_img.resize(target_size)).astype(np.float32) / 255.0
    mse = np.mean((ref - test) ** 2)
    return 20 * np.log10(1.0 / max(np.sqrt(mse), 1e-10))


def main(args):
    output_dir = Path(args.output_dir) / "editcache_qwen"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EditCache on Qwen-Image-Edit-2511")
    print("=" * 60)

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
    ).to(f"cuda:{args.gpu}")
    num_blocks = len(pipe.transformer.transformer_blocks)
    print(f"  Transformer: {num_blocks} blocks")

    test_image = make_test_image(args.width, args.height)
    test_image.save(output_dir / "input.png")
    edit_prompts = [
        "change the green rectangle to red",
        "add a blue sky background with clouds",
        "make the yellow circle into a sun with rays",
    ]

    all_results = {}

    # Baseline
    print("\n--- Baseline (no cache) ---")
    res = run_qwen_inference(pipe, test_image, edit_prompts, args, controller=None)
    baseline_t = float(np.mean(res["times"]))
    ref_images = res["images"]
    print(f"  Avg time: {baseline_t:.2f}s")
    all_results["baseline"] = {"avg_time": baseline_t, "times": res["times"]}
    for i, img in enumerate(ref_images):
        img.save(output_dir / f"baseline_{i}.png")

    # EditCache configs
    configs = [
        ("ec_conservative", 0.03, 0.08, 0.04, 0.30),
        ("ec_v1",           0.04, 0.10, 0.06, 0.25),
        ("ec_balanced",     0.05, 0.13, 0.07, 0.25),
        ("ec_aggressive",   0.06, 0.16, 0.08, 0.20),
    ]
    for name, sh, mi, de, warm in configs:
        warmup = max(1, int(args.num_steps * warm))
        print(f"\n--- {name} (sh={sh}, mi={mi}, de={de}, warmup={warm}) ---")
        ctrl = EditCacheQwenController(
            num_blocks=num_blocks, num_steps=args.num_steps,
            layer_thresh_shallow=sh, layer_thresh_middle=mi, layer_thresh_deep=de,
            warmup_steps=warmup,
        )
        res = run_qwen_inference(pipe, test_image, edit_prompts, args, controller=ctrl)
        avg_t = float(np.mean(res["times"]))
        speedup = baseline_t / avg_t

        psnrs = [compute_psnr_pair(ref, test) for ref, test in zip(ref_images, res["images"])]
        avg_psnr = float(np.mean(psnrs))
        hr = res["stats"]["overall_hit_rate"]

        all_results[name] = {
            "avg_time": avg_t, "speedup": speedup,
            "hit_rate": hr, "psnr": avg_psnr,
            "config": dict(shallow=sh, middle=mi, deep=de, warmup_ratio=warm),
        }
        print(f"  Time: {avg_t:.2f}s  Speedup: {speedup:.2f}x  HitRate: {hr:.1%}  PSNR: {avg_psnr:.2f}")
        for i, img in enumerate(res["images"]):
            img.save(output_dir / f"{name}_{i}.png")

    # FBCache baselines
    for thresh in [0.05, 0.10, 0.15]:
        name = f"fbcache_t{thresh}"
        print(f"\n--- {name} ---")
        ctrl = EditCacheQwenController(
            num_blocks=num_blocks, num_steps=args.num_steps,
            layer_thresh_shallow=thresh, layer_thresh_middle=thresh, layer_thresh_deep=thresh,
            warmup_steps=1,
        )
        res = run_qwen_inference(pipe, test_image, edit_prompts, args, controller=ctrl)
        avg_t = float(np.mean(res["times"]))
        speedup = baseline_t / avg_t
        psnrs = [compute_psnr_pair(ref, test) for ref, test in zip(ref_images, res["images"])]
        avg_psnr = float(np.mean(psnrs))
        hr = res["stats"]["overall_hit_rate"]
        all_results[name] = {
            "avg_time": avg_t, "speedup": speedup,
            "hit_rate": hr, "psnr": avg_psnr,
        }
        print(f"  Time: {avg_t:.2f}s  Speedup: {speedup:.2f}x  HitRate: {hr:.1%}  PSNR: {avg_psnr:.2f}")
        for i, img in enumerate(res["images"]):
            img.save(output_dir / f"{name}_{i}.png")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (Qwen-Image-Edit)")
    print("=" * 70)
    print(f"{'Method':25s} | {'Time(s)':>8} | {'Speedup':>7} | {'HitRate':>7} | {'PSNR':>5}")
    print("-" * 70)
    for name, r in all_results.items():
        spd = f"{r.get('speedup',1):.2f}x" if 'speedup' in r else "1.00x"
        hr = f"{r.get('hit_rate',0):.2%}" if r.get('hit_rate') is not None else "N/A"
        psnr = f"{r.get('psnr',0):.2f}" if r.get('psnr') is not None else "N/A"
        print(f"{name:25s} | {r['avg_time']:8.2f} | {spd:>7} | {hr:>7} | {psnr:>5}")

    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_dir}")

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
    parser.add_argument("--output_dir", type=str, default="./outputs")
    main(parser.parse_args())
