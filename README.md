# Tenos.ai Merge Node (Flux+ Deterministic) for ComfyUI

A robust, **Flux-aware** merge node for ComfyUI with **deterministic DARE**, extra merge modes, a **block-weight preset library**, regex and model-based **masks**, safety **skip/lock** toggles, and an **analysis-only** dry run. It’s designed for **FLUX-style UNets** (and generally works with SD-like models that follow similar block naming).

The node prioritizes **predictability, reproducibility, and VRAM sanity**. You choose the base model, control per-block weights (or one-click presets), and get a clean **JSON summary** with fingerprints so you can rerun the exact merge later.

---

# Quick Start

1. **Install**

   * Copy `tenos_merge.py` into `ComfyUI/custom_nodes/`
   * Restart ComfyUI (kill the process; don’t just Reload Nodes)

2. **Add the node**

   * Right-click → *Add Node* → **Tenos.ai** → **Tenosai Merge (Flux+ Deterministic)**

3. **Wire models**

   * `model1` = primary
   * `model2` = secondary

4. **Pick base & mode**

   * `base_model_choice` = the weights you start from
   * `merge_mode` = algorithm (see list below)

5. **Choose weights**

   * Either pick a **preset** (`block_preset` + toggle `apply_preset`)
   * …or dial each **block slider** manually (0.0 = keep base, 1.0 = take secondary)

6. **(Optional) Safety & Masks**

   * Use **skip\_/lock\_** toggles to protect sensitive parts
   * Use a **mask\_model** or **mask\_regex** to target specific params

7. **Run**

   * `MODEL` → merged model (feed to KSampler or SaveModel)
   * `STRING` → JSON summary (for audit + reproducibility)

---

# Installation & Compatibility

* Copy `tenos_merge.py` to `ComfyUI/custom_nodes/` and restart
* Node appears as **“Tenosai Merge (Flux+ Deterministic)”**
* Optimized for **Flux-style** UNets; works broadly with SD-like models using:

  * `input_blocks.*`, `output_blocks.*`, `middle_block` (or `mid_block`)

---

# Inputs & What They Mean

## Required

* `model1`, `model2` *(MODEL)* – the two models to merge
* `base_model_choice` *(model1 | model2)* – which model’s weights you start from
* `merge_mode`

  * `simple` – Linear interpolation (lerp)
  * `dare` – Deterministic DARE: prune smallest diffs, merge the rest
  * `weighted_sum` – `weight_1 * model1 + (1-weight_1) * model2` (independent of base)
  * `sigmoid_average` – Nonlinear remap of `amount` via sigmoid for softer blending
  * `tensor_addition` – Explicit delta: `t1 + amount * (t2 - t1)`
  * `difference_maximization` – Emphasize where `|t2| > |t1|`
  * `auto_similarity` – Cosine similarity per param scales effective blend automatically

## Preset Library (Block Weights)

* `block_preset` *(Custom, Balanced, Style-lean, Subject-lean, Text-obedient, Structure-keeper, Detail-boost, Minimal-change)*
* `apply_preset` *(bool)* – when **on**, preset **overrides all sliders** before merging

### What each preset is for

| Preset           | Intent                                              |
| ---------------- | --------------------------------------------------- |
| Balanced         | Neutral 50/50 baseline                              |
| Style-lean       | Lean into **style**: middle + upsampling            |
| Subject-lean     | Lean into **subject/identity**: downsampling        |
| Text-obedient    | Stronger prompt adherence (conditioning-heavy)      |
| Structure-keeper | Keep composition/structure; conservative everywhere |
| Detail-boost     | Sharpen details; upsampling heavy                   |
| Minimal-change   | Very light touch; exploratory                       |

**Exact weights** (0.0 keep base ←→ 1.0 take secondary):

```json
{
  "Balanced": {
    "Image Hint": 0.50, "Timestep Embedding": 0.50, "Text Conditioning": 0.50,
    "Early Downsampling (Composition)": 0.50, "Mid Downsampling (Subject & Concept)": 0.50, "Late Downsampling (Refinement)": 0.50,
    "Core Middle Block": 0.50, "Early Upsampling (Initial Style)": 0.50, "Mid Upsampling (Details)": 0.50, "Late Upsampling (Final Textures)": 0.50,
    "Final Output Layer": 0.50, "Other": 0.50
  },
  "Style-lean": {
    "Image Hint": 0.50, "Timestep Embedding": 0.50, "Text Conditioning": 0.50,
    "Early Downsampling (Composition)": 0.30, "Mid Downsampling (Subject & Concept)": 0.40, "Late Downsampling (Refinement)": 0.50,
    "Core Middle Block": 0.80, "Early Upsampling (Initial Style)": 0.75, "Mid Upsampling (Details)": 0.85, "Late Upsampling (Final Textures)": 0.90,
    "Final Output Layer": 0.60, "Other": 0.50
  },
  "Subject-lean": {
    "Image Hint": 0.50, "Timestep Embedding": 0.50, "Text Conditioning": 0.50,
    "Early Downsampling (Composition)": 0.80, "Mid Downsampling (Subject & Concept)": 0.80, "Late Downsampling (Refinement)": 0.70,
    "Core Middle Block": 0.60, "Early Upsampling (Initial Style)": 0.45, "Mid Upsampling (Details)": 0.40, "Late Upsampling (Final Textures)": 0.35,
    "Final Output Layer": 0.50, "Other": 0.50
  },
  "Text-obedient": {
    "Image Hint": 0.25, "Timestep Embedding": 0.60, "Text Conditioning": 0.85,
    "Early Downsampling (Composition)": 0.55, "Mid Downsampling (Subject & Concept)": 0.55, "Late Downsampling (Refinement)": 0.50,
    "Core Middle Block": 0.55, "Early Upsampling (Initial Style)": 0.50, "Mid Upsampling (Details)": 0.50, "Late Upsampling (Final Textures)": 0.50,
    "Final Output Layer": 0.50, "Other": 0.50
  },
  "Structure-keeper": {
    "Image Hint": 0.30, "Timestep Embedding": 0.40, "Text Conditioning": 0.50,
    "Early Downsampling (Composition)": 0.20, "Mid Downsampling (Subject & Concept)": 0.25, "Late Downsampling (Refinement)": 0.30,
    "Core Middle Block": 0.40, "Early Upsampling (Initial Style)": 0.40, "Mid Upsampling (Details)": 0.40, "Late Upsampling (Final Textures)": 0.40,
    "Final Output Layer": 0.40, "Other": 0.20
  },
  "Detail-boost": {
    "Image Hint": 0.50, "Timestep Embedding": 0.50, "Text Conditioning": 0.50,
    "Early Downsampling (Composition)": 0.35, "Mid Downsampling (Subject & Concept)": 0.40, "Late Downsampling (Refinement)": 0.45,
    "Core Middle Block": 0.65, "Early Upsampling (Initial Style)": 0.75, "Mid Upsampling (Details)": 0.85, "Late Upsampling (Final Textures)": 0.90,
    "Final Output Layer": 0.65, "Other": 0.50
  },
  "Minimal-change": {
    "Image Hint": 0.10, "Timestep Embedding": 0.10, "Text Conditioning": 0.10,
    "Early Downsampling (Composition)": 0.10, "Mid Downsampling (Subject & Concept)": 0.10, "Late Downsampling (Refinement)": 0.10,
    "Core Middle Block": 0.10, "Early Upsampling (Initial Style)": 0.10, "Mid Upsampling (Details)": 0.10, "Late Upsampling (Final Textures)": 0.10,
    "Final Output Layer": 0.10, "Other": 0.10
  }
}
```

> Presets **don’t** bypass safety toggles. If `skip_norms` is on, those params stay from the base regardless of preset weights.

## Block Weights (Manual)

* `Image Hint` – IP-Adapter / control influence
* `Timestep Embedding` – how the model interprets noise level
* `Text Conditioning` – prompt adherence
* `Early Downsampling (Composition)` – composition & layout
* `Mid Downsampling (Subject & Concept)` – subject identity & concepts
* `Late Downsampling (Refinement)` – pre-style refinement
* `Core Middle Block` – global style/identity
* `Early/Mid/Late Upsampling` – detail creation & textures
* `Final Output Layer` – output head / latent projection
* `Other` – anything not matched by the heuristics

**Rule of thumb:** Downsampling = *what*, Upsampling = *how it looks*, Middle = *style core*.

## Mode-Specific Settings

* **DARE**

  * `dare_prune_amount` (0.0–1.0): fraction of the **smallest** diffs pruned by quantile of `|t2 - t1|`
  * `dare_merge_amount` (0.0–1.0): strength applied to the unpruned elements
* **weighted\_sum**

  * `weight_1` (0.0–1.0): fraction of `model1`; `model2` gets `1 - weight_1` (independent of base)
* **sigmoid\_average**

  * `sigmoid_strength` (0.1–12.0): slope of the nonlinear blend
* **auto\_similarity**

  * `auto_k` (1.0–20.0): steepness mapping `(1 - cosine)` to effective weight

## Safety Toggles (Skip / Lock)

| Toggle              | Protects (keeps base)                                       | Why/When                                                          |
| ------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------- |
| `skip_bias`         | All `.bias`                                                 | Reduce drift; maintain activation centering                       |
| `skip_norms`        | LayerNorm, GroupNorm, BatchNorm (weights + biases)          | Preserve calibration; strongly recommended                        |
| `lock_time_embed`   | Timestep/time-embedding layers                              | Keep denoising schedule interpretation stable                     |
| `lock_conditioner`  | Text conditioning & related projections                     | Maintain prompt adherence while changing style/identity elsewhere |
| `lock_output_layer` | Final projection / head (e.g., `to_rgb`, `out.`, `final_*`) | Avoid breaking the last conversion stage                          |

---

# Masks (Targeted Control)

**Semantics:** after the per-block merge computes a candidate `merged`, the mask blends it with the original base tensor:

```
final = lerp(base_param, merged_param, mask)
# mask=0.0 → keep base; mask=1.0 → take merged
```

### Two ways to provide a mask

1. **mask\_model (per-param masks by name)**
   If `mask_model` has a tensor with the same name, that tensor is used as the mask.

2. **mask\_regex + mask\_value (constant mask on matches)**
   If a param name matches `mask_regex`, a constant mask filled with `mask_value` is used.

**Precedence:** `mask_model` overrides for names it covers; regex applies to the rest.

**Broadcasting supported:**

* Exact same shape – used as-is
* `[C,1,1]` or `[C]` → broadcast to conv weights `[C,H,W,...]`
* Scalar → broadcast to any shape
  If a mask can’t be broadcast, it’s ignored with a warning.

### Regex cookbook (Python syntax; escape dots)

* Protect all norms (keep base):

  ```
  mask_regex: (?i)(layernorm|groupnorm|batchnorm|ln_|norm)
  mask_value: 0.0
  ```
* Force final head from secondary:

  ```
  mask_regex: (?i)(to_rgb|out\.|final_layer|output_layer)
  mask_value: 1.0
  ```
* Freeze text conditioning:

  ```
  mask_regex: (?i)(text|cond|token|clip|context)
  mask_value: 0.0
  ```
* Hit only early downsample blocks:

  ```
  mask_regex: (?i)^input_blocks\.(?:0|1|2|3)
  mask_value: 1.0
  ```

---

# Execution Controls

* `analysis_only` – **dry run**: returns summary, doesn’t modify weights
* `calc_dtype` – math dtype (`float32` or `bfloat16`). Final params keep their original dtype.
* `seed` – sets torch seeds for deterministic behavior

---

# Outputs

* `MODEL` – the merged model
* `STRING` – JSON summary with:

  * sizes (GB) of model1, model2, final
  * merge mode + full settings (including preset name/applied)
  * preflight counts (missing/shape-mismatch)
  * run info (seed + fingerprints of key sets)
  * result stats (merged/kept counts, up to 50 error lines)

**Example JSON (trimmed):**

```json
{
  "base_model": "model1",
  "sizes_gb": {"model1": 3.21, "model2": 3.21, "final": 3.21},
  "merge_mode": "dare",
  "settings": {
    "preset": {"name": "Style-lean", "applied": true},
    "weights": {"Core Middle Block": 0.8, "...": 0.5},
    "dare": {"prune_amount": 0.1, "merge_amount": 1.0}
  },
  "preflight": {"missing_in_secondary": 0, "missing_in_base": 0, "shape_mismatches": 0},
  "run": {
    "seed": 0,
    "base_keys_fingerprint": "a1b2c3d4e5f6",
    "secondary_keys_fingerprint": "f6e5d4c3b2a1"
  },
  "result": {"merged_params": 1234, "kept_from_base": 56, "errors": 0}
}
```

---

# Block Mapping (Heuristic)

| UI Label                             | Typical param name patterns                     |
| ------------------------------------ | ----------------------------------------------- |
| Early Downsampling (Composition)     | `input_blocks.0–3`                              |
| Mid Downsampling (Subject & Concept) | `input_blocks.4–8`                              |
| Late Downsampling (Refinement)       | `input_blocks.9+`                               |
| Core Middle Block                    | `middle_block`, `mid_block`                     |
| Early Upsampling (Initial Style)     | `output_blocks.0–3`                             |
| Mid Upsampling (Details)             | `output_blocks.4–8`                             |
| Late Upsampling (Final Textures)     | `output_blocks.9+`                              |
| Final Output Layer                   | `to_rgb`, `out.`, `final_layer`, `output_layer` |
| Text Conditioning                    | `text`, `cond`, `token`, `clip`, `context`      |
| Image Hint                           | `ipadapter`, `control`, `image.*hint`           |
| Other                                | Anything else                                   |

Heuristics are designed to work across Flux and SD-like repos.

---

# Recipes

* **Style over structure**

  * Base = structure model
  * Raise **Core Middle Block** + **Upsampling** weights
  * Keep `skip_norms` on

* **Concept blending**

  * Raise **Downsampling** weights (subject/identity lives there)

* **Auto as smart default**

  * Use `auto_similarity` with moderate weights; it preserves layers that agree and leans into layers that differ

* **Fine-tune into base (DARE)**

  * `dare_prune_amount` 0.05–0.15, `dare_merge_amount` 1.0
  * Adjust per-block amounts normally

* **Keep fragile parts steady**

  * Turn on `skip_norms` and (often) `skip_bias`
  * Lock `time_embed`, `conditioner`, `output_layer` for minimal drift

---

# Troubleshooting

* **Node not found / old UI appears**

  * You’re loading an older file. Remove duplicates in `custom_nodes/`, delete `__pycache__`, restart Comfy.

* **Nothing seems to change**

  * Check per-block sliders; `0.0` means “keep base”.
  * In DARE, large `dare_prune_amount` can hide changes.

* **VRAM pressure**

  * Use `calc_dtype = bfloat16`. The math uses less memory; write-back preserves original dtype.

* **Weird behavior in weighted\_sum**

  * By design, `weight_1` always maps to **model1**, independent of base choice (predictable semantics).

* **Import error at startup**

  * Comfy will skip the node silently. Check the console for a traceback; fix the file or send the error here.

---

# Reproducibility & Licensing

* Set `seed` for deterministic merges.
* Summary includes **key-set fingerprints**. If fingerprints differ, you are not merging the same models.
* Respect the licenses of the models you’re merging. You are responsible for redistribution rights.

---

# Changelog (highlights)

* Deterministic DARE (quantile in float32)
* Extra modes: `difference_maximization`, `auto_similarity`, `sigmoid_average`
* Skip/lock safety toggles
* Regex + model-based masks with broadcasting
* **Preset library** with UI controls
* Analysis-only dry run and detailed JSON summary

---
