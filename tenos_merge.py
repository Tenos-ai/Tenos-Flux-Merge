import torch
import json
from collections import defaultdict
import comfy.model_management as model_management
import logging
from typing import Dict, Tuple, Any, Optional

IMAGE_HINT = "Image Hint"
TIME_EMBEDDING = "Timestep Embedding"
TEXT_CONDITIONING = "Text Conditioning"
EARLY_DOWNSAMPLING = "Early Downsampling (Composition)"
MID_DOWNSAMPLING = "Mid Downsampling (Subject & Concept)"
LATE_DOWNSAMPLING = "Late Downsampling (Refinement)"
CORE_MIDDLE_BLOCK = "Core/Middle Block (Style Focus)"
EARLY_UPSAMPLING = "Early Upsampling (Initial Style)"
MID_UPSAMPLING = "Mid Upsampling (Detail Generation)"
LATE_UPSAMPLING = "Late Upsampling (Final Textures)"
FINAL_OUTPUT_LAYER = "Final Output Layer (Latent Projection)"
OTHER = "Other Tensors"

class TenosaiMergeNode:
    """
    A robust custom node for merging models, specialized for FLUX.1.
    Features intuitive block controls, corrected merge math, and hardened against VRAM and state management issues.
    """
    def __init__(self):
        self.device = model_management.get_torch_device()
        self.setup_logger()

    def setup_logger(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def INPUT_TYPES(cls):
        return cls.get_input_definitions()

    @staticmethod
    def get_input_definitions():
        return {
            "required": {
                "model1": ("MODEL",), "model2": ("MODEL",),
                "base_model_choice": (["model1", "model2"], {"default": "model1"}),
                "merge_mode": (["simple", "dare", "weighted_sum", "sigmoid_average", "tensor_addition", "difference_maximization"],),
                IMAGE_HINT: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Controls influence from IPAdapters or ControlNets."}),
                TIME_EMBEDDING: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Controls how the model interprets the noise level at each generation step."}),
                TEXT_CONDITIONING: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Controls influence of the prompt."}),
                EARLY_DOWNSAMPLING: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Input/Double Blocks 0-3: Defines basic composition."}),
                MID_DOWNSAMPLING: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Input/Double Blocks 4-8: Develops core concepts."}),
                LATE_DOWNSAMPLING: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Input/Double Blocks 9-15: Refines concepts before styling."}),
                CORE_MIDDLE_BLOCK: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The central block, critical for style and subject."}),
                EARLY_UPSAMPLING: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Output/Single Blocks 0-3: Applies initial style."}),
                MID_UPSAMPLING: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Output/Single Blocks 4-8: Refines details."}),
                LATE_UPSAMPLING: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Output/Single Blocks 9-15: Finalizes textures."}),
                FINAL_OUTPUT_LAYER: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Final conversion to latent image."}),
                OTHER: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Controls any other tensors."}),
                "dare_prune_amount": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dare_merge_amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight_1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sigmoid_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }, "optional": { "mask_model": ("MODEL",) }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    FUNCTION = "tenosai_merge"
    CATEGORY = "Tenos.ai"

    def tenosai_merge(self, **kwargs) -> Tuple[Any, str]:
        model1, model2 = kwargs['model1'], kwargs['model2']
        base_model_choice = kwargs['base_model_choice']
        
        base_model_obj, secondary_model_obj = (model1, model2) if base_model_choice == "model1" else (model2, model1)
        merged_model = base_model_obj.clone()
        
        self.logger.info(f"Base: '{base_model_choice}'. Mode: '{kwargs['merge_mode']}'.")

        merge_info = self.perform_block_merge(merged_model, secondary_model_obj, kwargs)
        
        size1 = sum(p.numel() * p.element_size() for p in model1.model.parameters()) / (1024**3)
        size2 = sum(p.numel() * p.element_size() for p in model2.model.parameters()) / (1024**3)
        final_size = sum(p.numel() * p.element_size() for p in merged_model.model.parameters()) / (1024**3)

        analysis_summary = self.generate_minimal_summary(size1, size2, final_size, base_model_choice, merge_info)
        return (merged_model, analysis_summary)

    def merge_tensors(self, tp1: torch.Tensor, tp2: torch.Tensor, amount: float, mode: str, 
                      dare_prune: float, dare_merge: float, weight1: float, sigmoid_str: float,
                      base_is_model1: bool, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if tp1.shape != tp2.shape:
             self.logger.warning(f"Tensor shape mismatch, skipping merge for this tensor. Base: {tp1.shape}, Secondary: {tp2.shape}")
             return tp1

        original_dtype = tp1.dtype
        calc_dtype = torch.float32
        t1_calc = tp1.to(calc_dtype, copy=False)
        t2_calc = tp2.to(calc_dtype, copy=False)

        if mode == "dare":
            diff = torch.abs(t1_calc - t2_calc)
            threshold = torch.quantile(diff.flatten(), dare_prune)
            prune_mask = diff < threshold
            merged = torch.lerp(t1_calc, t2_calc, dare_merge)
            merged[prune_mask] = t1_calc[prune_mask]
        elif mode == "weighted_sum":
            weight_for_tp2 = 1.0 - weight1
            merged = (t1_calc * (weight1 if base_is_model1 else weight_for_tp2)) + \
                     (t2_calc * (weight_for_tp2 if base_is_model1 else weight1))
        elif mode == "sigmoid_average":
            weight = 1 / (1 + torch.exp(-12 * (sigmoid_str - 0.5)))
            merged = torch.lerp(t1_calc, t2_calc, weight)
        else:
            merged = torch.lerp(t1_calc, t2_calc, amount)
        
        if mask is not None:
             if mask.shape == merged.shape:
                mask_calc = mask.to(merged.dtype, copy=False)
                merged = torch.lerp(t1_calc, merged, mask_calc)
             else:
                self.logger.warning(f"Mask shape mismatch ignored. Mask: {mask.shape}, Tensor: {merged.shape}.")

        return merged.to(original_dtype)

    def get_block_from_key(self, key: str) -> str:
        if "time_embed" in key: return TIME_EMBEDDING
        if "conditioner" in key or "cond_proj" in key: return TEXT_CONDITIONING
        if "input_hint_block" in key: return IMAGE_HINT
        if "model.diffusion_model." not in key and "diffusion_model." not in key: return OTHER
        if "middle_block" in key: return CORE_MIDDLE_BLOCK
        if "output_blocks" in key:
            try:
                idx = int(key.split(".")[3 if key.startswith('model.') else 2])
                if idx < 4: return EARLY_UPSAMPLING
                if idx < 9: return MID_UPSAMPLING
                return LATE_UPSAMPLING
            except (ValueError, IndexError): return OTHER
        if "input_blocks" in key:
            try:
                idx = int(key.split(".")[3 if key.startswith('model.') else 2])
                if idx < 4: return EARLY_DOWNSAMPLING
                if idx < 9: return MID_DOWNSAMPLING
                return LATE_DOWNSAMPLING
            except (ValueError, IndexError): return OTHER
        if "out." in key: return FINAL_OUTPUT_LAYER
        return OTHER

    def perform_block_merge(self, merged_model: Any, secondary_model: Any, settings: Dict) -> Dict:
        merged_params, kept_params, error_params = 0, 0, 0
        errors = []
        mask_sd = settings.get('mask_model').model.state_dict() if settings.get('mask_model') else None
        base_is_model1 = settings['base_model_choice'] == 'model1'

        secondary_sd = secondary_model.model.state_dict()
        
        with torch.no_grad():
            for name, param in merged_model.model.named_parameters():
                if name in secondary_sd:
                    try:
                        block_name = self.get_block_from_key(name)
                        amount = settings.get(block_name, 0.5)
                        
                        t1_base = param.data
                        t2_secondary = secondary_sd[name]
                        
                        merged_tensor = self.merge_tensors(t1_base, t2_secondary, amount, settings['merge_mode'],
                                                           settings['dare_prune_amount'], settings['dare_merge_amount'],
                                                           settings['weight_1'], settings['sigmoid_strength'], 
                                                           base_is_model1, mask_sd.get(name) if mask_sd else None)
                        param.data.copy_(merged_tensor)
                        merged_params += 1
                    except Exception as e:
                        self.logger.error(f"Error merging tensor '{name}': {e}", exc_info=True)
                        errors.append(f"{name}: {e}")
                        error_params += 1
                else:
                    self.logger.debug(f"Parameter '{name}' not found in secondary model. Kept from base.")
                    kept_params += 1

        self.logger.info(f"Partial merge complete. Merged {merged_params} common tensors. Kept {kept_params} tensors unique to the base model. Errors: {error_params}")
        summary_info = settings.copy()
        summary_info.update({"components_merged": merged_params, "components_kept_from_base": kept_params, "errors": errors})
        return summary_info

    def generate_minimal_summary(self, size1: float, size2: float, final_size: float, base_model_choice: str, merge_info: Dict) -> str:
        
        block_weights = {
            key: val for key, val in merge_info.items() if isinstance(val, (float, int)) and key in 
            [IMAGE_HINT, TIME_EMBEDDING, TEXT_CONDITIONING, EARLY_DOWNSAMPLING, MID_DOWNSAMPLING, LATE_DOWNSAMPLING,
             CORE_MIDDLE_BLOCK, EARLY_UPSAMPLING, MID_UPSAMPLING, LATE_UPSAMPLING, FINAL_OUTPUT_LAYER, OTHER]
        }
        summary = {
            "base_model": base_model_choice,
            "model_sizes": {"model1": f"{size1:.2f}GB", "model2": f"{size2:.2f}GB", "final": f"{final_size:.2f}GB"},
            "merge_stats": {"merged": merge_info["components_merged"], "kept_from_base": merge_info["components_kept_from_base"], "errors": len(merge_info["errors"])},
            "settings": { "merge_mode": merge_info["merge_mode"] },
            "block_weights": block_weights
        }
        if merge_info['merge_mode'] == 'dare':
            summary['settings']['dare_settings'] = {"prune_amount": merge_info['dare_prune_amount'], "merge_amount": merge_info['dare_merge_amount']}
        elif merge_info['merge_mode'] == 'weighted_sum':
            summary['settings']['weight_for_model1'] = merge_info['weight_1']
        elif merge_info['merge_mode'] == 'sigmoid_average':
             summary['settings']['sigmoid_strength'] = merge_info['sigmoid_strength']
        return json.dumps(summary, indent=4)


NODE_CLASS_MAPPINGS = { "TenosaiMergeNode": TenosaiMergeNode }
NODE_DISPLAY_NAME_MAPPINGS = { "TenosaiMergeNode": "Tenosai Merge (FLUX)" }
