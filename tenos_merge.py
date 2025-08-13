import copy
import json
import logging
import math
import re
import hashlib
from typing import Dict, Tuple, Any, Optional

import torch
import comfy.model_management as model_management


# =========================
# UI Labels (stable: do not rename, used in graphs)
# =========================
IMAGE_HINT = "Image Hint"
TIME_EMBEDDING = "Timestep Embedding"
TEXT_CONDITIONING = "Text Conditioning"
GUIDANCE_EMBEDDING = "Guidance Embedding"        
VECTOR_EMBEDDING = "Vector Embedding"            
EARLY_DOWNSAMPLING = "Early Downsampling (Composition)"
MID_DOWNSAMPLING = "Mid Downsampling (Subject & Concept)"
LATE_DOWNSAMPLING = "Late Downsampling (Refinement)"
CORE_MIDDLE_BLOCK = "Core Middle Block"
EARLY_UPSAMPLING = "Early Upsampling (Initial Style)"
MID_UPSAMPLING = "Mid Upsampling (Details)"
LATE_UPSAMPLING = "Late Upsampling (Final Textures)"
FINAL_OUTPUT_LAYER = "Final Output Layer"
OTHER = "Other"


# =========================
# Preset library for block weights
# =========================
PRESETS: Dict[str, Dict[str, float]] = {
    "Balanced": {
        IMAGE_HINT: 0.50, TIME_EMBEDDING: 0.50, TEXT_CONDITIONING: 0.50,
        GUIDANCE_EMBEDDING: 0.50, VECTOR_EMBEDDING: 0.50,
        EARLY_DOWNSAMPLING: 0.50, MID_DOWNSAMPLING: 0.50, LATE_DOWNSAMPLING: 0.50,
        CORE_MIDDLE_BLOCK: 0.50, EARLY_UPSAMPLING: 0.50, MID_UPSAMPLING: 0.50, LATE_UPSAMPLING: 0.50,
        FINAL_OUTPUT_LAYER: 0.50, OTHER: 0.50
    },
    "Style-lean": {
        IMAGE_HINT: 0.50, TIME_EMBEDDING: 0.50, TEXT_CONDITIONING: 0.50,
        GUIDANCE_EMBEDDING: 0.40, VECTOR_EMBEDDING: 0.40,
        EARLY_DOWNSAMPLING: 0.30, MID_DOWNSAMPLING: 0.40, LATE_DOWNSAMPLING: 0.50,
        CORE_MIDDLE_BLOCK: 0.80, EARLY_UPSAMPLING: 0.75, MID_UPSAMPLING: 0.85, LATE_UPSAMPLING: 0.90,
        FINAL_OUTPUT_LAYER: 0.60, OTHER: 0.50
    },
    "Subject-lean": {
        IMAGE_HINT: 0.50, TIME_EMBEDDING: 0.50, TEXT_CONDITIONING: 0.50,
        GUIDANCE_EMBEDDING: 0.60, VECTOR_EMBEDDING: 0.60,
        EARLY_DOWNSAMPLING: 0.80, MID_DOWNSAMPLING: 0.80, LATE_DOWNSAMPLING: 0.70,
        CORE_MIDDLE_BLOCK: 0.60, EARLY_UPSAMPLING: 0.45, MID_UPSAMPLING: 0.40, LATE_UPSAMPLING: 0.35,
        FINAL_OUTPUT_LAYER: 0.50, OTHER: 0.50
    },
    "Text-obedient": {
        IMAGE_HINT: 0.25, TIME_EMBEDDING: 0.60, TEXT_CONDITIONING: 0.85,
        GUIDANCE_EMBEDDING: 0.60, VECTOR_EMBEDDING: 0.50,
        EARLY_DOWNSAMPLING: 0.55, MID_DOWNSAMPLING: 0.55, LATE_DOWNSAMPLING: 0.50,
        CORE_MIDDLE_BLOCK: 0.55, EARLY_UPSAMPLING: 0.50, MID_UPSAMPLING: 0.50, LATE_UPSAMPLING: 0.50,
        FINAL_OUTPUT_LAYER: 0.50, OTHER: 0.50
    },
    "Structure-keeper": {
        IMAGE_HINT: 0.30, TIME_EMBEDDING: 0.40, TEXT_CONDITIONING: 0.50,
        GUIDANCE_EMBEDDING: 0.40, VECTOR_EMBEDDING: 0.40,
        EARLY_DOWNSAMPLING: 0.20, MID_DOWNSAMPLING: 0.25, LATE_DOWNSAMPLING: 0.30,
        CORE_MIDDLE_BLOCK: 0.40, EARLY_UPSAMPLING: 0.40, MID_UPSAMPLING: 0.40, LATE_UPSAMPLING: 0.40,
        FINAL_OUTPUT_LAYER: 0.40, OTHER: 0.20
    },
    "Detail-boost": {
        IMAGE_HINT: 0.50, TIME_EMBEDDING: 0.50, TEXT_CONDITIONING: 0.50,
        GUIDANCE_EMBEDDING: 0.50, VECTOR_EMBEDDING: 0.50,
        EARLY_DOWNSAMPLING: 0.35, MID_DOWNSAMPLING: 0.40, LATE_DOWNSAMPLING: 0.45,
        CORE_MIDDLE_BLOCK: 0.65, EARLY_UPSAMPLING: 0.75, MID_UPSAMPLING: 0.85, LATE_UPSAMPLING: 0.90,
        FINAL_OUTPUT_LAYER: 0.65, OTHER: 0.50
    },
    "Minimal-change": {
        IMAGE_HINT: 0.10, TIME_EMBEDDING: 0.10, TEXT_CONDITIONING: 0.10,
        GUIDANCE_EMBEDDING: 0.10, VECTOR_EMBEDDING: 0.10,
        EARLY_DOWNSAMPLING: 0.10, MID_DOWNSAMPLING: 0.10, LATE_DOWNSAMPLING: 0.10,
        CORE_MIDDLE_BLOCK: 0.10, EARLY_UPSAMPLING: 0.10, MID_UPSAMPLING: 0.10, LATE_UPSAMPLING: 0.10,
        FINAL_OUTPUT_LAYER: 0.10, OTHER: 0.10
    },
}


# =========================
# Helpers
# =========================
def _device() -> torch.device:
    """Comfy’s preferred device, with CPU fallback."""
    try:
        return model_management.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _clone_model(model_obj):
    """Clone without mutating the original graph model."""
    if hasattr(model_obj, "clone") and callable(getattr(model_obj, "clone")):
        return model_obj.clone()
    return copy.deepcopy(model_obj)


def _is_norm_param(name: str) -> bool:
    low = name.lower()
    return any(k in low for k in ["layernorm", "groupnorm", "batchnorm", "ln_", "norm", "bn.weight", "bn.bias"])


def _is_bias(name: str) -> bool:
    return name.endswith(".bias")


def _is_time_embed(name: str) -> bool:
    low = name.lower()
    return ("time_embed" in low) or ("time_embedding" in low) or ("time_in" in low)


def _is_conditioner(name: str) -> bool:
    low = name.lower()
    return any(k in low for k in ["text", "cond", "token", "clip", "context", "txt_in"])


def _is_out_proj(name: str) -> bool:
    low = name.lower()
    return any(k in low for k in ["to_rgb", "out.", "output_layer", "final_layer"])


def _is_guidance(name: str) -> bool:
    low = name.lower()
    return ("guidance_in" in low) or ("guidance" in low and "embed" in low)


def _is_vector(name: str) -> bool:
    low = name.lower()
    return ("vector_in" in low) or ("vector" in low and "embed" in low)


def _block_from_key(name: str) -> str:
    """
    Map parameter names to functional UI blocks.
    Supports Flux-style (double_blocks/single_blocks) and SD-style (input_blocks/output_blocks),
    plus Flux inlet heads and final head.
    """
    low = name.lower()

    # Flux-style UNet: down = double_blocks.*, up = single_blocks.*
    m_db = re.search(r"(?:^|[._])double_blocks\.(\d+)", low)
    if m_db:
        idx = int(m_db.group(1))
        if idx <= 2:
            return EARLY_DOWNSAMPLING
        elif idx <= 4:
            return MID_DOWNSAMPLING
        else:
            return LATE_DOWNSAMPLING

    m_sb = re.search(r"(?:^|[._])single_blocks\.(\d+)", low)
    if m_sb:
        idx = int(m_sb.group(1))
        if idx <= 2:
            return EARLY_UPSAMPLING
        elif idx <= 4:
            return MID_UPSAMPLING
        else:
            return LATE_UPSAMPLING

    # SD-style for broad compatibility
    m_in = re.search(r"input_blocks\.(\d+)", low)
    if m_in:
        idx = int(m_in.group(1))
        if idx <= 3:
            return EARLY_DOWNSAMPLING
        elif idx <= 8:
            return MID_DOWNSAMPLING
        else:
            return LATE_DOWNSAMPLING

    m_out = re.search(r"output_blocks\.(\d+)", low)
    if m_out:
        idx = int(m_out.group(1))
        if idx <= 3:
            return EARLY_UPSAMPLING
        elif idx <= 8:
            return MID_UPSAMPLING
        else:
            return LATE_UPSAMPLING

    if ("middle_block" in low) or ("mid_block" in low):
        return CORE_MIDDLE_BLOCK
    if _is_time_embed(low):
        return TIME_EMBEDDING
    if _is_conditioner(low):
        return TEXT_CONDITIONING
    if (("image" in low) and ("hint" in low)) or ("ipadapter" in low) or ("control" in low) or ("img_in" in low):
        return IMAGE_HINT
    if _is_guidance(low):      # NEW
        return GUIDANCE_EMBEDDING
    if _is_vector(low):        # NEW
        return VECTOR_EMBEDDING
    if _is_out_proj(low):
        return FINAL_OUTPUT_LAYER
    return OTHER


def _broadcast_mask_like(mask: torch.Tensor, target: torch.Tensor) -> Optional[torch.Tensor]:
    """Broadcast common shapes to target parameter shape."""
    if mask is None:
        return None
    if tuple(mask.shape) == tuple(target.shape):
        return mask
    if mask.numel() == 1:
        return torch.full_like(target, float(mask.item()))
    if target.ndim >= 3:
        c = target.shape[0]
        if mask.ndim == 3 and mask.shape == (c, 1, 1):
            return mask.expand_as(target)
        if mask.ndim == 1 and mask.shape[0] == c:
            shape = [c] + [1] * (target.ndim - 1)
            return mask.view(*shape).expand_as(target)
    return None


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Scalar cosine similarity in float32 for stability."""
    a_f = a.flatten().to(torch.float32)
    b_f = b.flatten().to(torch.float32)
    denom = (a_f.norm() * b_f.norm())
    if denom == 0:
        return torch.tensor(1.0, dtype=torch.float32, device=a.device)
    return (a_f @ b_f) / denom


def _sha1_of_state_dict_keys(sd: Dict[str, torch.Tensor]) -> str:
    """Fingerprint of keys + shapes (not values)."""
    h = hashlib.sha1()
    for k in sorted(sd.keys()):
        h.update(k.encode("utf-8"))
        t = sd[k]
        h.update(str(tuple(t.shape)).encode("utf-8"))
    return h.hexdigest()[:12]


def _safe_quantile_threshold(diff: torch.Tensor, q: float) -> torch.Tensor:
    """
    Deterministic, robust quantile threshold for very large tensors.
    Try device quantile -> CPU quantile -> CPU kthvalue.
    Always float32; returns scalar tensor on diff.device.
    """
    q = float(max(0.0, min(1.0, q)))
    flat = diff.reshape(-1).to(torch.float32)
    if q <= 0.0:
        return torch.tensor(0.0, dtype=torch.float32, device=diff.device)
    if q >= 1.0:
        return torch.tensor(float("inf"), dtype=torch.float32, device=diff.device)
    try:
        return torch.quantile(flat, q)
    except Exception:
        pass
    try:
        th = torch.quantile(flat.cpu(), q)
        return th.to(diff.device)
    except Exception:
        pass
    n = flat.numel()
    k = max(1, min(n, int(q * n)))
    th = flat.cpu().kthvalue(k).values
    return th.to(diff.device)


def _select_calc_dtype(requested: str) -> torch.dtype:
    """Respect user's calc dtype but fall back safely if bf16 isn't supported."""
    if requested == "bfloat16":
        is_supported = False
        try:
            is_supported = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        except Exception:
            is_supported = False
        if is_supported:
            return torch.bfloat16
        return torch.float32
    return torch.float32


class TenosaiMergeNode:
    """
    Flux-aware model merge for ComfyUI:
      * deterministic DARE (safe quantile fallback)
      * extra modes
      * mask broadcasting (model + regex)
      * safety skip/lock toggles
      * presets
      * analysis-only dry run
      * seed determinism + dtype guard

    Returns: (MODEL, STRING) -> merged model and JSON summary.
    """

    def __init__(self) -> None:
        self.device = _device()
        self.logger = logging.getLogger("TenosaiMerge")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    # ---- ComfyUI plumbing ----
    @classmethod
    def INPUT_TYPES(cls):
        return cls.get_input_definitions()

    @staticmethod
    def get_input_definitions():
        return {
            "required": {
                "model1": ("MODEL",),
                "model2": ("MODEL",),
                "base_model_choice": (["model1", "model2"], {"default": "model1"}),
                "merge_mode": (
                    ["simple", "dare", "weighted_sum", "sigmoid_average", "tensor_addition",
                     "difference_maximization", "auto_similarity"],
                    {"default": "simple"}
                ),

                # Presets
                "block_preset": (
                    ["Custom", "Balanced", "Style-lean", "Subject-lean", "Text-obedient",
                     "Structure-keeper", "Detail-boost", "Minimal-change"],
                    {"default": "Custom"}
                ),
                "apply_preset": ("BOOLEAN", {"default": False}),

                # Block weights
                IMAGE_HINT: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                TIME_EMBEDDING: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                TEXT_CONDITIONING: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                GUIDANCE_EMBEDDING: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),  
                VECTOR_EMBEDDING: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),    
                EARLY_DOWNSAMPLING: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                MID_DOWNSAMPLING: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                LATE_DOWNSAMPLING: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                CORE_MIDDLE_BLOCK: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                EARLY_UPSAMPLING: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                MID_UPSAMPLING: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                LATE_UPSAMPLING: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                FINAL_OUTPUT_LAYER: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                OTHER: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),

                # Mode-specific
                "dare_prune_amount": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.001}),
                "dare_merge_amount": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 1.0, "step": 0.001}),
                "weight_1": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.001}),
                "sigmoid_strength": ("FLOAT", {"default": 2.5, "min": 0.1, "max": 12.0, "step": 0.1}),
                "auto_k": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.5}),

                # Safety toggles
                "skip_bias": ("BOOLEAN", {"default": False}),
                "skip_norms": ("BOOLEAN", {"default": True}),
                "lock_time_embed": ("BOOLEAN", {"default": False}),
                "lock_conditioner": ("BOOLEAN", {"default": False}),
                "lock_output_layer": ("BOOLEAN", {"default": False}),

                # Execution
                "analysis_only": ("BOOLEAN", {"default": False}),
                "calc_dtype": (["float32", "bfloat16"], {"default": "float32"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "mask_model": ("MODEL",),
                "mask_regex": ("STRING", {"default": ""}),
                "mask_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "summary")
    FUNCTION = "tenosai_merge"
    CATEGORY = "Tenos.ai/Model Surgery"

    def tenosai_merge(
        self,
        model1,
        model2,
        base_model_choice,
        merge_mode,
        **kwargs,
    ) -> Tuple[Any, str]:

        # Presets
        block_preset = kwargs.get("block_preset", "Custom")
        apply_preset = bool(kwargs.get("apply_preset", False))

        # Block weights
        block_weights = {
            IMAGE_HINT: float(kwargs.get(IMAGE_HINT, 0.5)),
            TIME_EMBEDDING: float(kwargs.get(TIME_EMBEDDING, 0.5)),
            TEXT_CONDITIONING: float(kwargs.get(TEXT_CONDITIONING, 0.5)),
            GUIDANCE_EMBEDDING: float(kwargs.get(GUIDANCE_EMBEDDING, 0.5)),
            VECTOR_EMBEDDING: float(kwargs.get(VECTOR_EMBEDDING, 0.5)),
            EARLY_DOWNSAMPLING: float(kwargs.get(EARLY_DOWNSAMPLING, 0.5)),
            MID_DOWNSAMPLING: float(kwargs.get(MID_DOWNSAMPLING, 0.5)),
            LATE_DOWNSAMPLING: float(kwargs.get(LATE_DOWNSAMPLING, 0.5)),
            CORE_MIDDLE_BLOCK: float(kwargs.get(CORE_MIDDLE_BLOCK, 0.5)),
            EARLY_UPSAMPLING: float(kwargs.get(EARLY_UPSAMPLING, 0.5)),
            MID_UPSAMPLING: float(kwargs.get(MID_UPSAMPLING, 0.5)),
            LATE_UPSAMPLING: float(kwargs.get(LATE_UPSAMPLING, 0.5)),
            FINAL_OUTPUT_LAYER: float(kwargs.get(FINAL_OUTPUT_LAYER, 0.5)),
            OTHER: float(kwargs.get(OTHER, 0.5)),
        }

        preset_applied = False
        preset_name = str(block_preset)
        if apply_preset and preset_name in PRESETS:
            block_weights = {k: float(v) for k, v in PRESETS[preset_name].items()}
            preset_applied = True

        # Other settings
        dare_prune_amount = float(kwargs.get("dare_prune_amount", 0.10))
        dare_merge_amount = float(kwargs.get("dare_merge_amount", 1.00))
        weight_1 = float(kwargs.get("weight_1", 0.50))
        sigmoid_strength = float(kwargs.get("sigmoid_strength", 2.5))
        auto_k = float(kwargs.get("auto_k", 6.0))

        skip_bias = bool(kwargs.get("skip_bias", False))
        skip_norms = bool(kwargs.get("skip_norms", True))
        lock_time_embed = bool(kwargs.get("lock_time_embed", False))
        lock_conditioner = bool(kwargs.get("lock_conditioner", False))
        lock_output_layer = bool(kwargs.get("lock_output_layer", False))

        analysis_only = bool(kwargs.get("analysis_only", False))
        calc_dtype_str = kwargs.get("calc_dtype", "float32")
        calc_dtype = _select_calc_dtype(calc_dtype_str)
        seed = int(kwargs.get("seed", 0))

        mask_model = kwargs.get("mask_model", None)
        mask_regex = kwargs.get("mask_regex", "") or ""
        mask_value = float(kwargs.get("mask_value", 1.0))

        if seed is not None and seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        base_model = model1 if base_model_choice == "model1" else model2
        secondary_model = model2 if base_model_choice == "model1" else model1
        base_is_model1 = base_model_choice == "model1"

        merged_model = _clone_model(base_model)
        base_sd = merged_model.model.state_dict()
        sec_sd = secondary_model.model.state_dict()
        mask_sd = mask_model.model.state_dict() if hasattr(mask_model, "model") else {}

        def _size_gb(sd: Dict[str, torch.Tensor]) -> float:
            return sum(t.numel() * t.element_size() for t in sd.values()) / (1024 ** 3)

        size1 = _size_gb(model1.model.state_dict())
        size2 = _size_gb(model2.model.state_dict())

        missing_in_sec = [k for k in base_sd.keys() if k not in sec_sd]
        missing_in_base = [k for k in sec_sd.keys() if k not in base_sd]
        shape_mismatches = [k for k in base_sd.keys() if (k in sec_sd) and (tuple(base_sd[k].shape) != tuple(sec_sd[k].shape))]

        merged = 0
        kept = 0
        errors = []

        if analysis_only:
            final_size = _size_gb(base_sd)
            summary = self._build_summary_json(
                base_model_choice=base_model_choice,
                merge_mode=merge_mode,
                block_weights=block_weights,
                dare_prune_amount=dare_prune_amount,
                dare_merge_amount=dare_merge_amount,
                weight_1=weight_1,
                sigmoid_strength=sigmoid_strength,
                auto_k=auto_k,
                size1=size1,
                size2=size2,
                final_size=final_size,
                changed_ratio_per_block={},
                deltas_per_block={},
                missing_in_secondary=len(missing_in_sec),
                missing_in_base=len(missing_in_base),
                shape_mismatches=len(shape_mismatches),
                seed=seed,
                base_keys_fingerprint=_sha1_of_state_dict_keys(base_sd),
                sec_keys_fingerprint=_sha1_of_state_dict_keys(sec_sd),
                errors=None,
                merged_count=None,
                kept_count=None,
                preset_name=preset_name,
                preset_applied=preset_applied,
            )
            return merged_model, summary

        with torch.no_grad():
            for name, param in merged_model.model.named_parameters():
                try:
                    if name not in sec_sd:
                        kept += 1
                        continue

                    t1 = param.data.to(self.device)
                    t2 = sec_sd[name].to(self.device)

                    if tuple(t1.shape) != tuple(t2.shape):
                        kept += 1
                        continue

                    if (skip_bias and _is_bias(name)) or (skip_norms and _is_norm_param(name)) \
                       or (lock_time_embed and _is_time_embed(name)) \
                       or (lock_conditioner and _is_conditioner(name)) \
                       or (lock_output_layer and _is_out_proj(name)):
                        kept += 1
                        continue

                    block = _block_from_key(name)
                    amount = float(block_weights.get(block, block_weights[OTHER]))
                    if amount <= 0.0:
                        kept += 1
                        continue

                    mask_tensor = None
                    if mask_sd and name in mask_sd:
                        mask_tensor = mask_sd[name].to(self.device)
                    if (mask_tensor is None) and mask_regex and re.search(mask_regex, name):
                        mask_tensor = torch.tensor([mask_value], device=self.device, dtype=torch.float32)

                    merged_tensor = self._merge_tensors(
                        t1=t1,
                        t2=t2,
                        amount=amount,
                        mode=merge_mode,
                        dare_prune=dare_prune_amount,
                        dare_merge=dare_merge_amount,
                        weight1=weight_1,
                        sigmoid_strength=sigmoid_strength,
                        auto_k=auto_k,
                        base_is_model1=base_is_model1,
                        mask=mask_tensor,
                        calc_dtype=calc_dtype,
                    )
                    param.data.copy_(merged_tensor)
                    merged += 1

                except Exception as ex:
                    errors.append(f"{name}: {repr(ex)}")
                    continue

        final_size = _size_gb(merged_model.model.state_dict())
        summary = self._build_summary_json(
            base_model_choice=base_model_choice,
            merge_mode=merge_mode,
            block_weights=block_weights,
            dare_prune_amount=dare_prune_amount,
            dare_merge_amount=dare_merge_amount,
            weight_1=weight_1,
            sigmoid_strength=sigmoid_strength,
            auto_k=auto_k,
            size1=size1,
            size2=size2,
            final_size=final_size,
            changed_ratio_per_block={},
            deltas_per_block={},
            missing_in_secondary=len(missing_in_sec),
            missing_in_base=len(missing_in_base),
            shape_mismatches=len(shape_mismatches),
            seed=seed,
            base_keys_fingerprint=_sha1_of_state_dict_keys(base_sd),
            sec_keys_fingerprint=_sha1_of_state_dict_keys(sec_sd),
            errors=errors,
            merged_count=merged,
            kept_count=kept,
            preset_name=preset_name,
            preset_applied=preset_applied,
        )
        return merged_model, summary

    def _merge_tensors(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
        amount: float,
        mode: str,
        dare_prune: float,
        dare_merge: float,
        weight1: float,
        sigmoid_strength: float,
        auto_k: float,
        base_is_model1: bool,
        mask: Optional[torch.Tensor],
        calc_dtype: torch.dtype,
    ) -> torch.Tensor:

        t1_calc = t1.to(calc_dtype, copy=False)
        t2_calc = t2.to(calc_dtype, copy=False)
        original_dtype = t1.dtype

        def _lerp(a: torch.Tensor, b: torch.Tensor, w):
            if isinstance(w, torch.Tensor):
                return torch.lerp(a, b, w.to(dtype=a.dtype, device=a.device))
            return torch.lerp(a, b, torch.tensor(w, dtype=a.dtype, device=a.device))

        if mode == "simple":
            merged = _lerp(t1_calc, t2_calc, amount)

        elif mode == "weighted_sum":
            w1 = torch.tensor(weight1, dtype=t1_calc.dtype, device=t1_calc.device)
            w2 = torch.tensor(1.0 - weight1, dtype=t1_calc.dtype, device=t1_calc.device)
            merged = t1_calc * w1 + t2_calc * w2

        elif mode == "sigmoid_average":
            centered = 2.0 * amount - 1.0
            eff = 1.0 / (1.0 + math.exp(-sigmoid_strength * centered))
            merged = _lerp(t1_calc, t2_calc, eff)

        elif mode == "tensor_addition":
            merged = t1_calc + amount * (t2_calc - t1_calc)

        elif mode == "difference_maximization":
            choose = (t2_calc.abs() > t1_calc.abs()).to(t1_calc.dtype)
            merged = t1_calc + amount * choose * (t2_calc - t1_calc)

        elif mode == "auto_similarity":
            cos = _cosine_similarity(t1_calc, t2_calc)
            w = 1.0 / (1.0 + torch.exp(torch.tensor(-auto_k * (1.0 - float(cos)), dtype=t1_calc.dtype, device=t1_calc.device)))
            eff = amount * float(w)
            merged = _lerp(t1_calc, t2_calc, eff)

        elif mode == "dare":
            diff = (t2_calc - t1_calc).abs()
            q = min(max(dare_prune, 0.0), 1.0)
            if q <= 0.0:
                prune_mask = torch.zeros_like(diff, dtype=torch.bool, device=diff.device)
            elif q >= 1.0:
                prune_mask = torch.ones_like(diff, dtype=torch.bool, device=diff.device)
            else:
                threshold = _safe_quantile_threshold(diff, q)
                prune_mask = diff < threshold

            candidate = _lerp(t1_calc, t2_calc, amount * dare_merge)
            merged = t1_calc.clone()
            merged[~prune_mask] = candidate[~prune_mask]

        else:
            merged = _lerp(t1_calc, t2_calc, amount)

        if mask is not None:
            mask_on_device = mask.to(t1_calc.device)
            mask_b = _broadcast_mask_like(mask_on_device, merged)
            if mask_b is not None:
                mask_b = mask_b.to(merged.dtype)
                merged = torch.lerp(t1_calc, merged, mask_b)
            else:
                logging.getLogger("TenosaiMerge").warning(
                    f"Mask shape {tuple(mask_on_device.shape)} not broadcastable to {tuple(merged.shape)}; ignoring mask."
                )

        return merged.to(original_dtype, copy=False)

    def _build_summary_json(
        self,
        base_model_choice: str,
        merge_mode: str,
        block_weights: Dict[str, float],
        dare_prune_amount: float,
        dare_merge_amount: float,
        weight_1: float,
        sigmoid_strength: float,
        auto_k: float,
        size1: float,
        size2: float,
        final_size: float,
        changed_ratio_per_block: Dict[str, float],
        deltas_per_block: Dict[str, Dict[str, float]],
        missing_in_secondary: int,
        missing_in_base: int,
        shape_mismatches: int,
        seed: int,
        base_keys_fingerprint: str,
        sec_keys_fingerprint: str,
        errors: Optional[list] = None,
        merged_count: Optional[int] = None,
        kept_count: Optional[int] = None,
        preset_name: str = "Custom",
        preset_applied: bool = False,
    ) -> str:
        summary = {
            "base_model": base_model_choice,
            "sizes_gb": {
                "model1": round(size1, 3),
                "model2": round(size2, 3),
                "final": round(final_size, 3)
            },
            "merge_mode": merge_mode,
            "settings": {
                "preset": {"name": preset_name, "applied": bool(preset_applied)},
                "weights": {k: round(float(v), 4) for k, v in block_weights.items()},
                "dare": {"prune_amount": float(dare_prune_amount), "merge_amount": float(dare_merge_amount)},
                "weighted_sum": {"weight_1": float(weight_1)},
                "sigmoid_average": {"strength": float(sigmoid_strength)},
                "auto_similarity": {"k": float(auto_k)},
            },
            "preflight": {
                "missing_in_secondary": int(missing_in_secondary),
                "missing_in_base": int(missing_in_base),
                "shape_mismatches": int(shape_mismatches),
            },
            "run": {
                "seed": int(seed),
                "base_keys_fingerprint": base_keys_fingerprint,
                "secondary_keys_fingerprint": sec_keys_fingerprint,
            },
        }
        if merged_count is not None and kept_count is not None:
            summary["result"] = {
                "merged_params": int(merged_count),
                "kept_from_base": int(kept_count),
                "errors": len(errors or []),
            }
            if errors:
                summary["errors_detail"] = [str(e) for e in errors[:50]]
        return json.dumps(summary, indent=2)


# ---- ComfyUI registration ----
NODE_CLASS_MAPPINGS = {"TenosaiMergeNode": TenosaiMergeNode}
NODE_DISPLAY_NAME_MAPPINGS = {"TenosaiMergeNode": "Tenos.ai Model Merge Node (Flux)"}
