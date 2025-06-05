import torch
import json
from collections import defaultdict
import comfy.model_management as model_management
import random
import logging
from typing import Dict, Tuple, Any, Optional

# --- Constants for Block Names (for maintainability) ---
DOUBLE_BLOCKS_0_5 = "double_blocks_0_5"
DOUBLE_BLOCKS_6_12 = "double_blocks_6_12"
DOUBLE_BLOCKS_13_18 = "double_blocks_13_18"
SINGLE_BLOCKS_0_15 = "single_blocks_0_15"
SINGLE_BLOCKS_16_25 = "single_blocks_16_25"
SINGLE_BLOCKS_26_37 = "single_blocks_26_37"
FINAL_LAYER = "final_layer"
TIME_IN = "time_in"
TXT_IN = "txt_in"
IMG_IN = "img_in"
OTHER = "other"

class TenosaiMergeNode:
    """
    A custom ComfyUI node for merging Stable Diffusion models, specialized for FLUX.1 Dev.
    Provides various merge methods and fine-grained control over block-specific merge amounts.
    """

    def __init__(self):
        self.device = model_management.get_torch_device()
        self.setup_logger()

    def setup_logger(self):
        logging.basicConfig(level=logging.DEBUG)  # Use DEBUG for detailed logs during development
        self.logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model1": ("MODEL",),
                "model2": ("MODEL",),
                "use_smaller_model": ("BOOLEAN", {"default": False}),
                "merge_mode": (
                    ["simple", "dare", "weighted_sum", "sigmoid_average", "tensor_addition", "difference_maximization"],
                ),
                "img_in": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "time_in": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guidance_in": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vector_in": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "txt_in": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "double_blocks_0_5_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "double_blocks_6_12_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "double_blocks_13_18_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "single_blocks_0_15_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "single_blocks_16_25_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "single_blocks_26_37_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "final_layer_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "other_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}), # Configurable "other"
                "force_keep_dim": ("BOOLEAN", {"default": False}),
                "random_drop_probability": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),  # For DARE
                "weight_1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),  # for weighted_sum
                "sigmoid_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),  # for sigmoid_average, adjusted range
            },
           "optional": {
                "mask_model": ("MODEL",),
           }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    FUNCTION = "tenosai_merge"
    CATEGORY = "Tenos.ai"

    def get_model_size(self, model) -> float:
        """Calculates the model size in GB, preferably from metadata, otherwise estimates."""
        if hasattr(model, 'metadata') and 'file_size' in model.metadata:
            return model.metadata['file_size'] / (1024 * 1024 * 1024)
        param_size = sum(p.numel() * p.element_size() for p in model.model.parameters())
        estimated_size = param_size / (1024 * 1024 * 1024)
        self.logger.warning(f"Model size not found in metadata. Estimated size: {estimated_size:.2f} GB")
        return estimated_size

    def tenosai_merge(self, model1, model2, use_smaller_model: bool, merge_mode: str,
                      img_in: float, time_in: float, guidance_in: float, vector_in: float, txt_in: float,
                      double_blocks_0_5_amount: float, double_blocks_6_12_amount: float, double_blocks_13_18_amount: float,
                      single_blocks_0_15_amount: float, single_blocks_16_25_amount: float, single_blocks_26_37_amount: float,
                      final_layer_amount: float, other_amount: float, force_keep_dim: bool, random_drop_probability: float,
                      weight_1: float, sigmoid_strength: float,
                      mask_model=None) -> Tuple[Any, str]:
        """Merges two models using the specified method and block-wise control."""

        # Input Validation includes sigmoid_strength now
        self._validate_inputs(img_in, time_in, guidance_in, vector_in, txt_in, double_blocks_0_5_amount,
                             double_blocks_6_12_amount, double_blocks_13_18_amount, single_blocks_0_15_amount,
                             single_blocks_16_25_amount, single_blocks_26_37_amount, final_layer_amount, other_amount,
                             random_drop_probability, weight_1, sigmoid_strength)


        size1 = self.get_model_size(model1)
        size2 = self.get_model_size(model2)
        self.logger.info(f"Model 1 size: {size1:.2f} GB, Model 2 size: {size2:.2f} GB")
        self.logger.info(f"Model 1/2 parameter count: {sum(p.numel() for p in model1.model.parameters())}/{sum(p.numel() for p in model2.model.parameters())}")
        self.logger.info(f"Use smaller model as base: {use_smaller_model}")

        base_model, secondary_model = (model1, model2) if (size1 <= size2 and use_smaller_model) or (size1 > size2 and not use_smaller_model) else (model2, model1)
        self.logger.info(f"Selected {'Model 1' if base_model == model1 else 'Model 2'} as base model")

        m1 = base_model.clone()
        m2 = secondary_model.clone()

        self.logger.debug(f"Base/Secondary model attributes: {dir(m1)}/{dir(m2)}")


        merge_info = self.perform_block_merge(m1, m2, merge_mode, img_in, time_in, guidance_in, vector_in, txt_in,
                                              double_blocks_0_5_amount, double_blocks_6_12_amount, double_blocks_13_18_amount,
                                              single_blocks_0_15_amount, single_blocks_16_25_amount, single_blocks_26_37_amount,
                                              final_layer_amount, other_amount, force_keep_dim, random_drop_probability,
                                              weight_1, sigmoid_strength, mask_model)


        final_size = self.get_model_size(m1)
        self.logger.info(f"Final model size: {final_size:.2f} GB")

        analysis_summary = self.generate_minimal_summary(size1, size2, final_size, use_smaller_model, merge_info)
        return m1, analysis_summary

    def _validate_inputs(self, *args):
        """Validates that all input float amounts are within [0.0, 1.0]."""
        for arg in args:
            if not 0.0 <= arg <= 1.0:
                raise ValueError(f"Input value {arg} is out of range [0.0, 1.0]")

    def merge_tensors(self, tp1: torch.Tensor, tp2: torch.Tensor, amount: float, mode: str, force_keep_dim: bool,
                      random_drop_probability: float, weight_1: float, sigmoid_strength: float,
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Merges two tensors using the specified merge mode."""

        if tp1.shape != tp2.shape:
            if force_keep_dim:
                return tp1
            else:
                raise ValueError(f"Tensor shapes do not match: {tp1.shape} vs {tp2.shape}")

        if mode == "dare":
            abs_diff = torch.abs(tp1 - tp2)
            max_diff = torch.max(abs_diff)
            importance = abs_diff / max_diff if max_diff > 0 else torch.ones_like(abs_diff)
            rand_tensor = torch.rand_like(tp1)
            merge_mask = (rand_tensor > random_drop_probability) & (rand_tensor < importance + random_drop_probability)
            merged = torch.where(merge_mask, tp1 * (1 - amount) + tp2 * amount, tp1)
        elif mode == "weighted_sum":
            merged = tp1 * weight_1 + tp2 * (1 - weight_1)
        elif mode == "sigmoid_average":
            # Correct sigmoid calculation for 0.0-1.0 range
            weight = 1 / (1 + torch.exp(-10 * (sigmoid_strength - 0.5)))  # Scale and shift for 0-1 range
            merged = tp1 * weight + tp2 * (1 - weight)
        elif mode == "tensor_addition":
            merged = tp1 + tp2
        elif mode == "difference_maximization":
            diff = torch.abs(tp1 - tp2)
            normalized_diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
            merged = tp1 * (1 - normalized_diff) + tp2 * normalized_diff
        else:  # Default: simple
            merged = tp1 * (1 - amount) + tp2 * amount

        if mask is not None:  # Keep the mask application, but don't expand on it
            merged = merged * mask + tp1 * (1 - mask)

        return merged


    def _get_double_blocks_key(self, key: str) -> str:
        """Helper function to determine the correct double block key."""
        if any(f"model.diffusion_model.input_blocks.{i}." in key for i in range(4)):
            return DOUBLE_BLOCKS_0_5
        elif any(f"model.diffusion_model.input_blocks.{i}." in key for i in range(4, 9)):
            return DOUBLE_BLOCKS_6_12
        elif any(f"model.diffusion_model.input_blocks.{i}." in key for i in range(9, 12)):
            return DOUBLE_BLOCKS_13_18
        return OTHER # Default

    def _get_single_blocks_key(self, key: str) -> str:
        """Helper function to determine the correct single block key."""

        if any(f"model.diffusion_model.output_blocks.{i}." in key for i in range(3)):
            return SINGLE_BLOCKS_0_15
        if any(f"model.diffusion_model.output_blocks.{i}." in key for i in range(3,6)):
            return  SINGLE_BLOCKS_0_15
        elif any(f"model.diffusion_model.output_blocks.{i}." in key for i in range(6, 9)):
            return SINGLE_BLOCKS_16_25
        elif any(f"model.diffusion_model.output_blocks.{i}." in key for i in range(9, 12)):
            return SINGLE_BLOCKS_26_37
        return OTHER  #default

    def get_block_from_key(self, key: str) -> str:
        """Determines the block type based on the parameter key."""

        if "model.diffusion_model.input_blocks." in key:
            return self._get_double_blocks_key(key)
        if "model.diffusion_model.output_blocks." in key:
            return self._get_single_blocks_key(key)
        if "model.diffusion_model.out." in key:
            return FINAL_LAYER
        if "model.diffusion_model.middle_block" in key:
            return SINGLE_BLOCKS_16_25  # As per your original logic

        # Input-specific handling
        if ".time_embed." in key:
            return TIME_IN
        if ".emb." in key or any(x in key for x in [".cond_stage_model.", ".cond_proj", "conditioner"]):
            return TXT_IN
        if ".input_hint_block." in key:
            return IMG_IN

        return OTHER

    def perform_block_merge(self, m1: Any, m2: Any, merge_mode: str,
                            img_in: float, time_in: float, guidance_in: float, vector_in: float, txt_in: float,
                            double_blocks_0_5_amount: float, double_blocks_6_12_amount: float, double_blocks_13_18_amount: float,
                            single_blocks_0_15_amount: float, single_blocks_16_25_amount: float, single_blocks_26_37_amount: float,
                            final_layer_amount: float, other_amount:float, force_keep_dim: bool, random_drop_probability: float,
                            weight_1: float, sigmoid_strength: float, mask_model: Optional[Any]) -> Dict:
        """Performs the block-wise merging of the two models."""

        merge_info = {
            "merge_mode": merge_mode,
            "img_in": img_in,
            "time_in": time_in,
            "guidance_in": guidance_in,
            "vector_in": vector_in,
            "txt_in": txt_in,
            "double_blocks_0_5_amount": double_blocks_0_5_amount,
            "double_blocks_6_12_amount": double_blocks_6_12_amount,
            "double_blocks_13_18_amount": double_blocks_13_18_amount,
            "single_blocks_0_15_amount": single_blocks_0_15_amount,
            "single_blocks_16_25_amount": single_blocks_16_25_amount,
            "single_blocks_26_37_amount": single_blocks_26_37_amount,
            "final_layer_amount": final_layer_amount,
            "other_amount": other_amount,
            "force_keep_dim": force_keep_dim,
            "random_drop_probability": random_drop_probability,
            "weight_1": weight_1,
            "sigmoid_strength": sigmoid_strength,
            "components": defaultdict(dict),
            "errors": []
        }

        block_merge_amounts = {
            IMG_IN: img_in,
            TIME_IN: time_in,
            "guidance_in": guidance_in,  #String literal to avoid future refactoring if the constant name changes
            "vector_in": vector_in,
            TXT_IN: txt_in,
            DOUBLE_BLOCKS_0_5: double_blocks_0_5_amount,
            DOUBLE_BLOCKS_6_12: double_blocks_6_12_amount,
            DOUBLE_BLOCKS_13_18: double_blocks_13_18_amount,
            SINGLE_BLOCKS_0_15: single_blocks_0_15_amount,
            SINGLE_BLOCKS_16_25: single_blocks_16_25_amount,
            SINGLE_BLOCKS_26_37: single_blocks_26_37_amount,
            FINAL_LAYER: final_layer_amount,
            OTHER: other_amount,
        }

        total_params = 0
        merged_params = 0
        kept_params = 0
        error_params = 0

        with torch.no_grad():  # Disable gradient calculations for efficiency
            for name, param in m1.model.named_parameters():
                total_params += 1
                self.logger.debug(f"Processing parameter: {name}")
                if name in m2.model.state_dict():
                    block = self.get_block_from_key(name)
                    t1 = param.data
                    t2 = m2.model.state_dict()[name].data
                    mask = mask_model.model.state_dict()[name] if mask_model and name in mask_model.model.state_dict() else None

                    try:
                        self.logger.debug(f"Shapes - t1: {t1.shape}, t2: {t2.shape}, Types - t1: {t1.dtype}, t2: {t2.dtype}")
                        merged_tensor = self.merge_tensors(t1, t2, block_merge_amounts[block], merge_mode,
                                                           force_keep_dim, random_drop_probability, weight_1,
                                                           sigmoid_strength, mask)

                        if not torch.allclose(merged_tensor, t1, rtol=1e-05, atol=1e-08):
                            param.data.copy_(merged_tensor)
                            merge_info["components"][name] = self.get_merge_component_info(block_merge_amounts[block], t1, t2, merged_tensor)
                            self.logger.debug(f"Successfully merged: {name}")
                            merged_params += 1
                        else:
                            merge_info["components"][name] = "Kept from base model (no significant change)"
                            self.logger.debug(f"No significant change after merge: {name}")
                            kept_params += 1

                    except (ValueError, TypeError, RuntimeError) as e:  # Catch specific exceptions
                        self.logger.error(f"Error merging {name}: {str(e)}")
                        merge_info["errors"].append(f"Error merging {name}: {str(e)}")
                        merge_info["components"][name] = "Kept from base model due to error"
                        error_params += 1
                else:
                    self.logger.debug(f"Parameter not found in secondary model: {name}")
                    merge_info["components"][name] = "Kept from base model (not in secondary)"
                    kept_params += 1

        merge_info["components_merged"] = merged_params
        merge_info["components_kept"] = kept_params
        merge_info["components_errored"] = error_params

        self.logger.info(f"Merge complete. Total: {total_params}, Merged: {merged_params}, Kept: {kept_params}, Errors: {error_params}")
        return merge_info

    def generate_minimal_summary(self, size1: float, size2: float, final_size: float, use_smaller_model: bool,
                                 merge_info: Dict) -> str:
        """Generates a concise summary of the merge operation."""
        base_size = min(size1, size2) if use_smaller_model else max(size1, size2)
        summary = {
            "model_sizes": {
                "model1": f"{size1:.2f}GB",
                "model2": f"{size2:.2f}GB",
                "final": f"{final_size:.2f}GB"
            },
            "merge_stats": {
                "components_merged": merge_info["components_merged"],
                "components_kept": merge_info["components_kept"],
                "errors": len(merge_info["errors"])
            },
            "settings": {
                "use_smaller_model": use_smaller_model,
                "merge_mode": merge_info["merge_mode"],
                "merge_amounts": {
                    "img_in": merge_info["img_in"],
                    "time_in": merge_info["time_in"],
                    "guidance_in": merge_info["guidance_in"],
                    "vector_in": merge_info["vector_in"],
                    "txt_in": merge_info["txt_in"],
                    "double_blocks_0_5_amount": merge_info["double_blocks_0_5_amount"],
                    "double_blocks_6_12_amount": merge_info["double_blocks_6_12_amount"],
                    "double_blocks_13_18_amount": merge_info["double_blocks_13_18_amount"],
                    "single_blocks_0_15_amount": merge_info["single_blocks_0_15_amount"],
                    "single_blocks_16_25_amount": merge_info["single_blocks_16_25_amount"],
                    "single_blocks_26_37_amount": merge_info["single_blocks_26_37_amount"],
                    "final_layer_amount": merge_info["final_layer_amount"],
                    "other_amount": merge_info["other_amount"],
                },
                "weight_1": merge_info["weight_1"],
                "sigmoid_strength": merge_info["sigmoid_strength"],
            }
        }
        if abs(final_size - base_size) / base_size > 0.01:  # 1% tolerance for size check
            summary["warning"] = "Unexpected final model size"
        return json.dumps(summary, indent=2)

    def get_merge_component_info(self, merge_amount: float, t1: torch.Tensor, t2: torch.Tensor,
                                 merged_tensor: torch.Tensor) -> Dict:
        """Collects statistics about a merged component."""
        return {
            "merge_amount": merge_amount,
            "original_mean": float(t1.mean()),
            "original_std": float(t1.std()),
            "secondary_mean": float(t2.mean()),
            "secondary_std": float(t2.std()),
            "merged_mean": float(merged_tensor.mean()),
            "merged_std": float(merged_tensor.std())
        }

NODE_CLASS_MAPPINGS = {
    "TenosaiMergeNode": TenosaiMergeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TenosaiMergeNode": "Tenosai Merge Node (FLUX)"
}