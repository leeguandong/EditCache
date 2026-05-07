"""
EditCache — Layer-Adaptive Residual Caching for Image Editing DiTs
Public API (top-level import).

Usage:
    from editcache import EditCacheController, patch_pipeline, unpatch_pipeline

    ctrl = EditCacheController.from_preset("balanced", num_blocks=60, num_steps=20)
    restore = patch_pipeline(pipe, ctrl)
    ctrl.reset()
    out = pipe(image=img, prompt=prompt, ...)
    restore()
    print(ctrl.get_stats())
"""

from __future__ import annotations
import torch
import numpy as np
from typing import Callable, Optional


# ── Presets ──────────────────────────────────────────────────────────────────
_PRESETS = {
    # name: (tau_shallow, tau_middle, tau_deep, warmup_ratio)
    "lossless":     (0.02, 0.05, 0.03, 0.30),
    "high_quality": (0.03, 0.10, 0.05, 0.25),
    "balanced":     (0.04, 0.10, 0.05, 0.25),
    "fast":         (0.06, 0.15, 0.07, 0.20),
    "aggressive":   (0.08, 0.20, 0.10, 0.15),
}


# ── Controller ───────────────────────────────────────────────────────────────
class EditCacheController:
    """Per-block dual-stream residual cache with CFG-aware step tracking.

    Supports both no-CFG (single-stream or dual-stream) and CFG mode where
    transformer.forward is called twice per denoising step (cond + uncond).
    Two independent cache slots (pass_idx 0/1) prevent cross-contamination.
    """

    def __init__(
        self,
        num_blocks: int,
        num_steps: int,
        layer_thresh_shallow: float = 0.04,
        layer_thresh_middle: float = 0.10,
        layer_thresh_deep: float = 0.05,
        warmup_steps: Optional[int] = None,
        warmup_ratio: Optional[float] = None,
        cfg: bool = False,
    ):
        self.num_blocks = num_blocks
        self.num_steps = num_steps
        self.thresh_shallow = layer_thresh_shallow
        self.thresh_middle = layer_thresh_middle
        self.thresh_deep = layer_thresh_deep

        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = max(1, int(warmup_ratio * num_steps))
        else:
            self.warmup_steps = max(1, num_steps // 4)

        self.cfg = cfg
        self._init_state()

    @classmethod
    def from_preset(
        cls,
        preset: str,
        num_blocks: int,
        num_steps: int,
        cfg: bool = False,
    ) -> "EditCacheController":
        """Construct a controller from a named preset.

        Presets: lossless / high_quality / balanced / fast / aggressive
        """
        if preset not in _PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(_PRESETS)}")
        ts, tm, td, wr = _PRESETS[preset]
        return cls(
            num_blocks=num_blocks,
            num_steps=num_steps,
            layer_thresh_shallow=ts,
            layer_thresh_middle=tm,
            layer_thresh_deep=td,
            warmup_ratio=wr,
            cfg=cfg,
        )

    # ── State ────────────────────────────────────────────────────────────────

    def _init_state(self):
        n, b = 2, self.num_blocks
        self.prev_h           = [[None] * b, [None] * b]
        self.prev_e           = [[None] * b, [None] * b]
        self.cached_h_residual = [[None] * b, [None] * b]
        self.cached_e_residual = [[None] * b, [None] * b]
        self.cache_hits        = [0] * b
        self.total_decisions   = [0] * b
        self.fwd_call_idx      = 0
        self.current_step      = 0
        self.pass_idx          = 0

    def reset(self):
        """Call before each new inference (new image/prompt pair)."""
        self._init_state()

    # ── Step tracking ────────────────────────────────────────────────────────

    def step_begin(self):
        """Called once at the top of every transformer.forward invocation."""
        self.fwd_call_idx += 1
        if self.cfg:
            if self.fwd_call_idx % 2 == 1:   # cond pass → new timestep
                self.current_step += 1
                self.pass_idx = 0
            else:                              # uncond pass
                self.pass_idx = 1
        else:
            self.current_step += 1
            self.pass_idx = 0

    # ── Per-block threshold ──────────────────────────────────────────────────

    def get_block_threshold(self, block_idx: int) -> float:
        L = self.num_blocks
        if block_idx < L // 6:
            return self.thresh_shallow
        elif block_idx >= 5 * L // 6:
            return self.thresh_deep
        else:
            return self.thresh_middle

    # ── Cache logic ──────────────────────────────────────────────────────────

    def should_skip(self, block_idx: int, h: torch.Tensor, e=None) -> bool:
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
        rel_change = (
            (h - self.prev_h[p][block_idx]).abs().mean()
            / (self.prev_h[p][block_idx].abs().mean() + 1e-8)
        ).item()
        return rel_change < self.get_block_threshold(block_idx)

    def update_cache(self, block_idx: int, in_h, in_e, out_h, out_e):
        p = self.pass_idx
        self.prev_h[p][block_idx] = in_h.detach().clone()
        if in_e is not None:
            self.prev_e[p][block_idx] = in_e.detach().clone()
        self.cached_h_residual[p][block_idx] = (out_h - in_h).detach().clone()
        if out_e is not None and in_e is not None:
            self.cached_e_residual[p][block_idx] = (out_e - in_e).detach().clone()

    def get_residual(self, block_idx: int, h, e=None):
        p = self.pass_idx
        h_out = h + self.cached_h_residual[p][block_idx]
        if self.cached_e_residual[p][block_idx] is not None and e is not None:
            e_out = e + self.cached_e_residual[p][block_idx]
        else:
            e_out = e
        return h_out, e_out

    # ── Stats ────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        total_hits = sum(self.cache_hits)
        total_dec  = sum(self.total_decisions)
        return {
            "overall_hit_rate": total_hits / max(total_dec, 1),
            "per_block_hit_rate": [
                h / max(t, 1) for h, t in zip(self.cache_hits, self.total_decisions)
            ],
            "total_hits":       total_hits,
            "total_decisions":  total_dec,
            "actual_steps":     self.current_step,
            "fwd_calls":        self.fwd_call_idx,
        }

    def __repr__(self) -> str:
        return (
            f"EditCacheController(blocks={self.num_blocks}, steps={self.num_steps}, "
            f"tau=({self.thresh_shallow},{self.thresh_middle},{self.thresh_deep}), "
            f"warmup={self.warmup_steps}, cfg={self.cfg})"
        )


# ── Patch / unpatch helpers ──────────────────────────────────────────────────

def patch_pipeline(pipe, controller: EditCacheController) -> Callable:
    """Monkey-patch a HuggingFace diffusers pipeline with EditCache.

    Returns a `restore()` callable that removes the patches.
    Compatible with QwenImageEditPlusPipeline (dual-stream) and
    FluxPipeline (single-stream double blocks).
    """
    transformer = pipe.transformer
    blocks = transformer.transformer_blocks
    assert len(blocks) == controller.num_blocks, (
        f"Controller has num_blocks={controller.num_blocks} but "
        f"pipe.transformer has {len(blocks)} blocks."
    )

    original_forwards = []
    for idx, block in enumerate(blocks):
        orig_fwd = block.forward
        original_forwards.append(orig_fwd)

        def _make_wrapped(block_idx, orig):
            def wrapped(hidden_states, encoder_hidden_states=None, *args, **kwargs):
                controller.total_decisions[block_idx] += 1
                if controller.should_skip(block_idx, hidden_states, encoder_hidden_states):
                    controller.cache_hits[block_idx] += 1
                    h_out, e_out = controller.get_residual(
                        block_idx, hidden_states, encoder_hidden_states
                    )
                    # preserve original output tuple structure
                    if encoder_hidden_states is not None:
                        return e_out, h_out
                    return (h_out,)
                output = orig(hidden_states, encoder_hidden_states, *args, **kwargs)
                if encoder_hidden_states is not None:
                    e_out, h_out = output[0], output[1]
                else:
                    h_out = output[0] if isinstance(output, (tuple, list)) else output
                    e_out = None
                controller.update_cache(
                    block_idx, hidden_states, encoder_hidden_states, h_out, e_out
                )
                return output
            return wrapped

        block.forward = _make_wrapped(idx, orig_fwd)

    orig_transformer_fwd = transformer.forward

    def fwd_with_step_track(*args, **kwargs):
        controller.step_begin()
        return orig_transformer_fwd(*args, **kwargs)

    transformer.forward = fwd_with_step_track

    def restore():
        for idx, block in enumerate(blocks):
            block.forward = original_forwards[idx]
        transformer.forward = orig_transformer_fwd

    return restore


# Alias for symmetry
unpatch_pipeline = None   # use the `restore` callable returned by patch_pipeline
