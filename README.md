# 🌐 Egregora-ARMD for ComfyUI

Regional adaptive diffusion nodes for **semantic upscaling** and **region-based conditioning** inside ComfyUI. ✨

Egregora-ARMD lets you split an image into planned regions, assign **one prompt per region**, and denoise the canvas in controlled batches while blending the results back together into a shared latent space. 🧠🧩

This is especially useful for workflows where a single global prompt is not enough — for example, when different parts of the image need different semantic instructions during upscale or re-detailing. 🔍

---

## 🚀 What this package is for

Egregora-ARMD is designed for workflows where you want:

- 🖼️ **regional semantic upscaling**
- 🧠 **different text conditioning for different areas of the image**
- 📦 **batched region processing** to balance speed and VRAM usage
- 🌊 **smooth overlap blending** between neighboring regions
- 🔌 optional routing of **extra runtime payloads** such as images, latents, or static values

Rather than treating the full canvas with one prompt, ARMD lets you define a **region plan** and then apply **regional conditioning** on top of that plan. Each region is processed in context of the full planned canvas, then merged back with weighted blending. 🎯

---

## ✨ Main use case: upscaling

ARMD is primarily aimed at **upscaling and detail enhancement workflows** where different image areas benefit from different prompt instructions.

Examples:

- a character face needs one description 👤
- clothing or accessories need another 👗
- background architecture or scenery needs another 🏛️🌄
- fine local refinement should happen without forcing one global caption across the whole frame 🧩

In practice, this makes ARMD a strong fit for:

- high-resolution regional prompt workflows
- image-to-image upscale pipelines
- tiled enhancement flows with semantic control
- caption-driven regional prompt generation

---

## 🧠 What makes ARMD different

ARMD is not just a simple tile splitter.

It combines:

- 📐 **explicit region planning**
- 📝 **one prompt per region**
- 🔁 **runtime region batching**
- 🌊 **adaptive overlap blending**
- 📦 optional **spatial and static payload adapters**

This means the package can act as a small **regional conditioning system** for ComfyUI, not only a geometry helper.

---

## ✅ Current validation status

At the moment, the safest public claim is:

- ✅ **Tested:** SDXL, Z-Image Turbo
- ⚠️ **Optional but not required:** captioners, ControlNet
- 🧪 **Not officially validated yet:** Flux, AnimateDiff, other video/temporal workflows

ControlNet is **optional**. Captioners are **optional**. ARMD can be used in a fully manual workflow. 🙌

---

## 🧩 Included nodes

### 🧭 Egregora Region Plan
Creates the canonical region layout for the run.

**Outputs:**
- aligned image
- batch preview of regions
- list preview of regions
- region plan object
- region count
- region order text

This node is the spatial foundation of the workflow. It decides:

- aligned image size
- latent size derived from compression
- region width / height
- overlap
- row/column arrangement
- region ordering

---

### 🧠 Egregora Regional Conditioning
Encodes **one positive prompt per region** and optionally negative prompts.

**Outputs:**
- regional positive conditioning
- regional negative conditioning
- placeholder positive conditioning
- placeholder negative conditioning
- prompt count

This is the node that connects text prompts to the region plan. ✍️

---

### 🌐 Egregora Adaptive Diffusion Apply
Applies the ARMD runtime patch to a model.

This is where the model becomes region-aware during denoising. It consumes:

- the model
- the region plan
- regional positive conditioning
- regional negative conditioning
- region batch size
- optional runtime payload adapter

---

### 🔎 Egregora Region Select
Selects a single region preview from the region list.

Useful for inspection and debugging of region order. 🧐

---

### 🧩 Egregora Spatial Tensor Pack
Packs optional spatial runtime payloads.

Can carry:
- image tensors
- latent tensors

Use this when another part of your workflow needs image-like or latent-like data routed regionally at runtime. 📦

---

### 📦 Egregora Static Payload Pack
Packs optional non-spatial runtime values.

Use this for simple runtime flags or named values. 🏷️

---

### 🔗 Egregora Runtime Adapter Merge
Merges multiple runtime payload adapters into one.

Useful when you want to combine:
- spatial payloads
- static payloads
- multiple adapter sources

---

## 🛠️ Basic workflow structure

A minimal ARMD workflow usually looks like this:

1. 🖼️ Load your image
2. 🧭 Create a **Region Plan**
3. 📝 Create **Regional Conditioning**
4. 🌐 Apply **Adaptive Diffusion** to the model
5. 🎛️ Run your normal sampler / denoising flow

Optional extras:

- 🤖 use an external captioner to generate region prompts
- 🎚️ use ControlNet-compatible runtime payloads
- 📦 inject additional spatial or static runtime data

---

## 📝 Manual prompts: how to send them correctly

This is the most important rule when using ARMD manually:

> **You must provide one non-empty prompt line per region, in the exact order generated by the Region Plan node.**

### ✅ Positive prompts
The `positive_prompts` field must contain:

- **exactly one non-empty line per region**
- in the **same order** shown by `region_order_text`

Example for a plan with **4 regions**:

```text
prompt for region 1
prompt for region 2
prompt for region 3
prompt for region 4
```

### ✅ Negative prompts
The `negative_prompts` field is optional.

You can use it in three ways:

#### 1. Leave it empty
ARMD will use blank negatives for all regions.

#### 2. Use a single negative line
That single negative prompt will be repeated to all regions.

Example:

```text
blurry, deformed, low quality
```

#### 3. Use one negative line per region
The number of lines must exactly match the region count.

Example for 4 regions:

```text
negative for region 1
negative for region 2
negative for region 3
negative for region 4
```

### ⚠️ Important notes

- empty lines are ignored
- region order matters
- prompt count must match the region plan
- the safest way is to copy the order from `region_order_text`

---

## 📋 Understanding `region_order_text`

The `Region Plan` node outputs a text block called `region_order_text`.

This tells you the exact order of the regions, including:

- region index
- row
- column
- pixel bounding box
- latent bounding box

Use this output as your reference when writing manual prompts. 🧭

In other words:

- line 1 in `positive_prompts` → region 1
- line 2 in `positive_prompts` → region 2
- line 3 in `positive_prompts` → region 3
- and so on

---

## 🤖 Captioners are optional

ARMD does **not** require an external captioner.

You can use it in two different ways:

### Manual mode ✍️
You write the region prompts yourself, one line per region.

### Captioner-assisted mode 🤖
You use any external captioner / VLM / custom prompt-generation workflow to produce one caption per region, then paste or route those prompts into `Egregora Regional Conditioning`.

ARMD only needs the final ordered prompt list. It does not depend on a specific captioner implementation. 🔓

---

## 🎛️ ControlNet is optional

ARMD does **not** require ControlNet.

You can use ARMD perfectly well with just:

- a model
- a region plan
- regional prompts

When used with compatible runtime payload workflows, ARMD can also transport additional regional payloads through its adapter system. This makes it possible to build more advanced regional pipelines without making ControlNet mandatory for basic usage. 🔌

---

## 📐 Alignment, compression, and region geometry

The `Egregora Region Plan` node aligns the input image before planning regions.

Available alignment modes:

- `keep_proportion_resize` 📏
- `stretch_resize` ↔️
- `floor_crop` ✂️

The plan is then built from:

- `region_width`
- `region_height`
- `region_overlap`
- `compression`

This determines both the pixel-space and latent-space layout used by the runtime. 🧠

---

## ⚡ Region batching

ARMD processes regions in batches using `region_batch_size`.

This lets you trade:

- 🚀 more throughput with larger batches
- 💾 lower memory usage with smaller batches

The spatial plan stays the same; only runtime grouping changes.

---

## 🧪 Advanced runtime payloads

ARMD includes optional payload nodes for more advanced workflows.

### 🧩 Spatial Tensor Pack
For image or latent data that should be routed as runtime payload.

### 📦 Static Payload Pack
For string-based named values or flags.

### 🔗 Runtime Adapter Merge
For combining multiple runtime payload sources into one adapter.

These nodes are optional and mainly intended for advanced workflow composition. Most users can start with just:

- Region Plan
- Regional Conditioning
- Adaptive Diffusion Apply

---

## 📦 Installation

1. Copy this folder into:

```text
ComfyUI/custom_nodes/
```

2. Restart ComfyUI 🔄
3. Search for `Egregora` in the node menu 🔍

---

## 🪄 Recommended starting point

For a first test:

- use **Egregora Region Plan** on your source image
- inspect the generated regions
- read `region_order_text`
- write exactly one prompt line per region
- apply ARMD to the model
- sample as usual

This is the fastest way to validate whether your regional prompt layout is working as intended. ✅

---

## 📚 Attribution

This project is conceptually aligned with region-based diffusion research such as:

- **MultiDiffusion**
- **Mixture of Diffusers**

See `ATTRIBUTION.md` for details. 📄

---

## ❤️ Status

Egregora-ARMD is in the stage where the core nodes are ready for real-world workflow validation and broader user testing.

The current focus is:

- validating behavior in practical upscale workflows
- refining prompt ergonomics
- preparing public examples and workflows
