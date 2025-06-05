# Tenos.ai Merge Node (FLUX) for ComfyUI

The Tenos.ai Merge Node is a powerful custom node for ComfyUI designed for merging Flux Diffusion models, with a special focus and block definitions tailored for the **FLUX.1 Dev model architecture**. It offers a variety of merge methods and allows fine-grained, block-specific control over how much each model contributes to the final merged result.

## Key Features

*   **FLUX.1 Dev Optimized:** Block definitions and naming conventions are specifically designed for the FLUX.1 Dev model structure (e.g., "double blocks," "single blocks" with specific layer groupings).
*   **Multiple Merge Methods:**
    *   `simple`: Basic linear interpolation.
    *   `dare` (Drop and ReScale): Selectively merges based on weight differences and drops less important weights.
    *   `weighted_sum`: A straightforward weighted average of the two models.
    *   `sigmoid_average`: A weighted average where the weight is determined by a sigmoid function, offering smoother transitions.
    *   `tensor_addition`: Direct summation of tensor values.
    *   `difference_maximization`: Emphasizes areas where the models differ most.
*   **Granular Block Control:** Adjust merge amounts for distinct sections of the FLUX.1 Dev model:
    *   Image Input (`img_in`)
    *   Time Embeddings (`time_in`)
    *   Text/Conditioning Embeddings (`txt_in`)
    *   Guidance/Vector Embeddings (configurable)
    *   Specific groups of "Double Blocks" (input blocks of the UNet)
    *   Specific groups of "Single Blocks" (output blocks and middle block of the UNet)
    *   Final Output Layer (`final_layer`)
    *   Other/Uncategorized Layers (`other_amount`)
*   **Base Model Selection:** Option to automatically use the smaller model as the base for merging.
*   **Dimension Mismatch Handling:** `force_keep_dim` option to keep the base model's tensor if dimensions mismatch, preventing errors.
*   **Masking (Optional):** Use a third model as a mask to selectively apply merging. *(Note: This feature's effectiveness depends on the mask model's structure and how it aligns with the tensors being merged.)*
*   **Detailed JSON Summary:** Outputs a string containing a JSON summary of the merge operation, including settings, model sizes, and merge statistics.
*   **Logging:** Provides debug-level logging for detailed insight into the merging process (check your ComfyUI console).

## Installation

1.  **Clone or Download:**
    *   **Option 1 (Git):** Navigate to your ComfyUI `custom_nodes` directory (`ComfyUI/custom_nodes/`) and run:
        ```bash
        git clone https://github.com/Tenos-ai/Tenos-Flux-Merge
        ```
    *   **Option 2 (Manual):** Download the `tenos_merge.py` file. Place it directly into your `ComfyUI/custom_nodes/` directory.

2.  **Restart ComfyUI:** Ensure ComfyUI is restarted to recognize the new custom node.

## Usage

1.  **Find the Node:** Right-click on the ComfyUI canvas, navigate to `Add Node` -> `Tenos.ai`, and select `Tenosai Merge Node (FLUX)`.
2.  **Connect Models:**
    *   Connect your primary FLUX.1 Dev model to `model1`.
    *   Connect your secondary FLUX.1 Dev model to `model2`.
    *   Optionally, connect a third model to `mask_model` if you wish to use masking.
3.  **Configure Parameters:** Adjust the merge mode, block-specific amounts, and other settings as desired.
4.  **Output:**
    *   The `MODEL` output is the merged model, ready to be used by a KSampler or other nodes.
    *   The `STRING` output provides a JSON summary of the merge process, which can be useful for tracking experiments.

## Inputs

**Required:**

*   **`model1`**: (`MODEL`) The first FLUX.1 Dev model to merge.
*   **`model2`**: (`MODEL`) The second FLUX.1 Dev model to merge.
*   **`use_smaller_model`**: (`BOOLEAN`, Default: `False`) If `True`, the smaller of `model1` and `model2` (by file size or estimated parameter size) will be used as the base model for the merge. Otherwise, `model1` is the base if it's larger or equal, and `model2` if `model1` is smaller.
*   **`merge_mode`**: (`COMBO`) The merging algorithm to use:
    *   `simple`
    *   `dare`
    *   `weighted_sum`
    *   `sigmoid_average`
    *   `tensor_addition`
    *   `difference_maximization`
*   **`img_in`**: (`FLOAT`, 0.0-1.0, Default: 0.5) Merge amount for image input related blocks (e.g., `input_hint_block`).
*   **`time_in`**: (`FLOAT`, 0.0-1.0, Default: 0.5) Merge amount for time embedding blocks.
*   **`guidance_in`**: (`FLOAT`, 0.0-1.0, Default: 0.5) Merge amount for guidance/vector related blocks (configurable, general category).
*   **`vector_in`**: (`FLOAT`, 0.0-1.0, Default: 0.5) Merge amount for additional vector inputs (if identified as such).
*   **`txt_in`**: (`FLOAT`, 0.0-1.0, Default: 0.5) Merge amount for text/conditioning related blocks (e.g., `cond_stage_model`, `cond_proj`).
*   **`double_blocks_0_5_amount`**: (`FLOAT`, 0.0-1.0, Default: 0.5) Merge amount for FLUX.1 input blocks 0-3 (referred to as Double Blocks group 1).
*   **`double_blocks_6_12_amount`**: (`FLOAT`, 0.0-1.0, Default: 0.5) Merge amount for FLUX.1 input blocks 4-8 (referred to as Double Blocks group 2).
*   **`double_blocks_13_18_amount`**: (`FLOAT`, 0.0-1.0, Default: 0.5) Merge amount for FLUX.1 input blocks 9-11 (referred to as Double Blocks group 3).
*   **`single_blocks_0_15_amount`**: (`FLOAT`, 0.0-1.0, Default: 0.5) Merge amount for FLUX.1 output blocks 0-5 (referred to as Single Blocks group 1).
*   **`single_blocks_16_25_amount`**: (`FLOAT`, 0.0-1.0, Default: 0.5) Merge amount for FLUX.1 output blocks 6-8 and the middle block (referred to as Single Blocks group 2).
*   **`single_blocks_26_37_amount`**: (`FLOAT`, 0.0-1.0, Default: 0.5) Merge amount for FLUX.1 output blocks 9-11 (referred to as Single Blocks group 3).
*   **`final_layer_amount`**: (`FLOAT`, 0.0-1.0, Default: 0.5) Merge amount for the final output convolution of the UNet.
*   **`other_amount`**: (`FLOAT`, 0.0-1.0, Default: 0.5) Merge amount for any parameters not categorized into the above blocks.
*   **`force_keep_dim`**: (`BOOLEAN`, Default: `False`) If `True` and tensor shapes mismatch, the base model's tensor is kept, preventing errors. If `False`, a ValueError is raised.
*   **`random_drop_probability`**: (`FLOAT`, 0.0-1.0, Default: 0.1) Used by the `dare` merge mode. Probability of dropping weights.
*   **`weight_1`**: (`FLOAT`, 0.0-1.0, Default: 0.5) Used by `weighted_sum` and (indirectly by `simple` if amount != 0.5). The weight applied to `model1`'s tensors (or base model's tensors). `model2` (secondary model) gets `1 - weight_1`.
*   **`sigmoid_strength`**: (`FLOAT`, 0.0-1.0, Default: 0.5) Used by `sigmoid_average`. Controls the steepness and center of the sigmoid curve, influencing the blending. A value of 0.5 results in an equal (0.5) weight. Values closer to 0 or 1 will heavily favor one model over the other.

**Optional:**

*   **`mask_model`**: (`MODEL`) A third model whose parameters can be used as a mask during merging. Where the mask tensor is 1, merging occurs; where 0, the base model's tensor is kept. *Note: Structural similarity is important for this to be effective.*

## Outputs

*   **`MODEL`**: The merged FLUX.1 Dev model.
*   **`STRING`**: A JSON formatted string containing a summary of the merge operation, including model sizes, merge statistics (components merged, kept, errored), and all applied settings.

## Merge Methods Explained

*   **`simple`**: `merged = base * (1 - amount) + secondary * amount`. A linear interpolation.
*   **`dare`**: (Drop and ReScale) Identifies important differences between tensors. Randomly drops some weights based on `random_drop_probability` and this importance, then merges the rest. Aims to preserve unique features while reducing redundancy.
*   **`weighted_sum`**: `merged = base * weight_1 + secondary * (1 - weight_1)`.
*   **`sigmoid_average`**: `weight = 1 / (1 + exp(-10 * (sigmoid_strength - 0.5)))`, then `merged = base * weight + secondary * (1 - weight)`. This creates a smoother, non-linear weighting.
*   **`tensor_addition`**: `merged = base + secondary`. Can lead to large values; use with caution.
*   **`difference_maximization`**: Normalizes the absolute difference between tensors and uses this normalized difference to weight the secondary model more heavily where differences are large: `merged = base * (1 - normalized_diff) + secondary * normalized_diff`.

## FLUX.1 Dev Block Definitions (Approximate)

The block categories in this node are designed to align with the FLUX.1 Dev architecture:

*   **Input Embeddings:**
    *   `img_in`: Hint/image encoder related.
    *   `time_in`: Time step embeddings.
    *   `txt_in`: Text/prompt conditioning embeddings.
    *   `guidance_in`/`vector_in`: Other forms of conditioning or vector inputs.
*   **UNet Backbone (Diffusion Model):**
    *   `double_blocks_0_5`: Early input blocks of the UNet (e.g., `input_blocks.0` to `input_blocks.3`). These are "DoubleTransformerBlock" in FLUX.
    *   `double_blocks_6_12`: Mid input blocks (e.g., `input_blocks.4` to `input_blocks.8`).
    *   `double_blocks_13_18`: Late input blocks (e.g., `input_blocks.9` to `input_blocks.11`).
    *   `single_blocks_0_15`: Early output blocks (e.g., `output_blocks.0` to `output_blocks.5`). These are "SingleTransformerBlock" in FLUX.
    *   `single_blocks_16_25`: Mid output blocks (e.g., `output_blocks.6` to `output_blocks.8`) AND the `middle_block`.
    *   `single_blocks_26_37`: Late output blocks (e.g., `output_blocks.9` to `output_blocks.11`).
    *   `final_layer`: The final convolutional layer (`out`).
*   **`other_amount`**: Catches any layers not fitting the above, such as normalization layers not directly part of a transformer block, or other utility layers.

*(Note: The exact mapping of layer indices to these groups is based on the internal logic of `get_block_from_key` in the script.)*

## Tips & Use Cases

*   **Targeted Merging:** This node excels when you want to, for example, take the text understanding (`txt_in`) from Model A but the style from the early UNet blocks (`double_blocks_...`) of Model B.
*   **Experimentation:** Start with `simple` or `weighted_sum` and small deviations from 0.5 for block amounts. Then, explore more complex methods like `dare` or `difference_maximization`.
*   **Iterative Refinement:** Use the JSON output to track your settings and results. Merge, test, adjust.
*   **Understanding FLUX:** Familiarity with the FLUX.1 Dev model architecture will help in making informed decisions about block-specific merge amounts.
*   **Console Logs:** Check the ComfyUI console for detailed logs during the merge, especially if you encounter unexpected behavior or want to see which parameters fall into the `other` category.

## Troubleshooting & Notes

*   **FLUX.1 Dev Specificity:** While it might work on other models, the block definitions are highly specific to FLUX.1 Dev. Results on other architectures may be unpredictable.
*   **Memory Usage:** Merging large models can be memory-intensive. Ensure you have sufficient RAM/VRAM.
*   **Parameter Mismatches:** The `force_keep_dim` option is a safeguard. If disabled, ensure your models have compatible architectures for layers you intend to merge.
*   **Mask Model:** The `mask_model` should ideally have a similar structure to the models being merged for the masking to be meaningful at a per-parameter level.
