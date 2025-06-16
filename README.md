# Tenos.ai Merge Node (FLUX) for ComfyUI

The Tenos.ai Merge Node is a robust, memory-optimized custom node for ComfyUI designed for merging diffusion models. It is highly specialized for **FLUX-style architectures**, featuring an intuitive, block-based control system that allows for granular and predictable merging.

This node has been hardened through extensive testing to handle VRAM limitations, device-offloading state changes, and cross-precision model merging, ensuring a stable and reliable user experience even across repeated runs.

## Key Features

*   **Memory Optimized:** Merges are performed **in-place** without cloning the base model, drastically reducing VRAM consumption and preventing allocation errors on consecutive runs.
*   **Intuitive Functional Blocks:** No more guessing which layers you're merging. Blocks are named by their function in the image generation process (e.g., "Early Downsampling (Composition)," "Core/Middle Block (Style Focus)"), with tooltips explaining each one.
*   **Robust Error Handling:**
    *   Immune to ComfyUI's model offloading (`cuda vs cpu` errors).
    *   Handles massive models (like FLUX) without crashing in `DARE` mode.
    *   Gracefully handles cross-precision merging (e.g., FP16 and FP8).
*   **Advanced Merge Methods:** A full suite of mathematically correct merge algorithms:
    *   `simple`: Linear interpolation.
    *   `dare`: Robustly prunes the smallest changes and merges the most significant ones.
    *   `weighted_sum`: A straightforward weighted average.
    *   `sigmoid_average`: Non-linear averaging for smoother blending.
    *   `tensor_addition`: Adds scaled tensor values from the secondary model.
    *   `difference_maximization`: Emphasizes areas where the models differ most.
*   **Explicit Base Model Selection:** You have direct control to choose which model (`model1` or `model2`) serves as the foundation for the merge.
*   **Detailed JSON Summary:** Outputs a clean JSON string summarizing the entire merge operation, including all settings and statistics, for perfect reproducibility.

## Installation

1.  **Clone or Download:**
    *   **Option 1 (Git):** Navigate to your ComfyUI `custom_nodes` directory (`ComfyUI/custom_nodes/`) and run:
        ```bash
        git clone https://github.com/Tenos-ai/Tenos-Flux-Merge
        ```
    *   **Option 2 (Manual):** Download the `tenos_merge.py` file. Place it directly into your `ComfyUI/custom_nodes/` directory.

2.  **Restart ComfyUI:** Ensure ComfyUI is restarted to recognize the new custom node.

## Usage

1.  **Find the Node:** Right-click on the ComfyUI canvas, navigate to `Add Node` -> `Tenos.ai`, and select `Tenosai Merge (FLUX)`.
2.  **Connect Models:**
    *   Connect your primary FLUX model to `model1`.
    *   Connect your secondary FLUX model to `model2`.
3.  **Configure Parameters:**
    *   Choose your desired `base_model_choice` and `merge_mode`.
    *   Adjust the sliders for each functional block. Hover over a slider's name to see its tooltip and understand what it controls.
4.  **Output:**
    *   The `MODEL` output is the merged model, ready for a KSampler.
    *   The `STRING` output provides the JSON summary for your records.

## Inputs Explained

**Required:**

*   **`model1` / `model2`**: (`MODEL`) The two models to be merged.
*   **`base_model_choice`**: (`COMBO`) Explicitly choose `model1` or `model2` to be the base for the merge.
*   **`merge_mode`**: (`COMBO`) The merging algorithm to use.
*   **Block Weights**: (`FLOAT`, 0.0-1.0) Sliders to control the merge ratio for each part of the model. A value of 0.0 keeps the base model's block, while 1.0 uses the secondary model's block.
    *   `Image Hint`: Controls influence from IPAdapters or ControlNets.
    *   `Timestep Embedding`: Controls how the model interprets the noise level at each step.
    *   `Text Conditioning`: Controls the influence of the prompt.
    *   `Early Downsampling (Composition)`: Defines the image's basic composition and structure.
    *   `Mid Downsampling (Subject & Concept)`: Develops the core concepts and subjects.
    *   `Late Downsampling (Refinement)`: Refines abstract concepts before the style core.
    *   `Core/Middle Block (Style Focus)`: The central block, critical for style and subject matter.
    *   `Early Upsampling (Initial Style)`: First upscaling layers where style is initially applied.
    *   `Mid Upsampling (Detail Generation)`: Main upscaling layers that add and refine details.
    *   `Late Upsampling (Final Textures)`: Final upscaling layers for fine details and textures.
    *   `Final Output Layer (Latent Projection)`: The final conversion to the latent image.
    *   `Other Tensors`: Controls any remaining tensors not covered by the main blocks.
*   **Mode-Specific Parameters:**
    *   **For `dare` mode:**
        *   `dare_prune_amount`: (0.0-1.0) The percentage of the *least significant* weight changes to ignore (prune).
        *   `dare_merge_amount`: (0.0-1.0) The strength of the merge applied to the *most significant* changes that were not pruned.
    *   **For `weighted_sum` mode:**
        *   `weight_1`: (0.0-1.0) The weight for `model1`. `model2` receives `1.0 - weight_1`.
    *   **For `sigmoid_average` mode:**
        *   `sigmoid_strength`: (0.0-1.0) Controls the non-linear blending curve.

**Optional:**

*   **`mask_model`**: (`MODEL`) A third model whose parameters can be used as a mask. *Note: Structural similarity is important for this to be effective.*

## Outputs

*   **`MODEL`**: The merged FLUX model.
*   **`STRING`**: A JSON formatted string containing a complete summary of the merge operation.

## Block Mapping Explained

The intuitive labels in the UI correspond to the technical block names in FLUX-style models as follows:

| UI Label                             | Technical Block Name(s)                                    |
| ------------------------------------ | ---------------------------------------------------------- |
| `Early Downsampling (Composition)`   | `input_blocks` / `double_blocks` **0-3**                   |
| `Mid Downsampling (Subject & Concept)` | `input_blocks` / `double_blocks` **4-8**                   |
| `Late Downsampling (Refinement)`     | `input_blocks` / `double_blocks` **9-15**                  |
| `Core/Middle Block (Style Focus)`    | `middle_block`                                             |
| `Early Upsampling (Initial Style)`   | `output_blocks` / `single_blocks` **0-3**                  |
| `Mid Upsampling (Detail Generation)` | `output_blocks` / `single_blocks` **4-8**                  |
| `Late Upsampling (Final Textures)`   | `output_blocks` / `single_blocks` **9-15**                 |
| `Final Output Layer`                 | `out`                                                      |
| `Text Conditioning`                  | `conditioner`, `cond_proj`                                 |

## Tips & Use Cases

*   **Targeted Style Transfer:** To transfer the style from `model2` to the structure of `model1`, set `model1` as the base and increase the weights for `Core/Middle Block` and the `Upsampling` blocks.
*   **Concept Blending:** For blending concepts, try adjusting the `Downsampling` blocks, as this is where the model interprets the core subject matter.
*   **DARE for Fine-Tuning:** The `dare` mode is excellent for merging a fine-tuned model (e.g., one trained on a character) into a base model. Use a small `dare_prune_amount` (like 0.05) and a `dare_merge_amount` of 1.0 to add only the most important changes from the fine-tune.
*   **Iterative Workflow:** This node is designed for repeated runs. You can merge two models, send the output to the sampler, and then feed the *same output* back into the `model1` slot to merge it with a third model, all without crashing.

## Troubleshooting & Notes

*   **Memory Usage:** This node is highly memory-optimized due to its in-place merging strategy. It is significantly more stable than nodes that clone the model in memory.
*   **Architecture Compatibility:** The node is optimized for FLUX models but may work on other architectures that use similar block naming (`input_blocks`, `output_blocks`, etc.).
*   **Parameter Mismatches:** If a tensor exists in one model but not the other (or has a different shape), it is automatically skipped, and the base model's tensor is kept. An error will not be raised. Check the console for warnings.
