# Egregora ARMD

**ARMD** stands for **Adaptive Regional Mixture of Diffusers**.

Egregora ARMD is a regional diffusion workflow for **semantic upscaling**, **multi-prompt image generation**, and **adaptive regional reconstruction** in ComfyUI. It is especially useful when you want the continuity benefits of shared-canvas denoising together with **different prompts for different image regions**.

Instead of treating each tile as an isolated img2img job, ARMD applies regional conditioning over a shared latent canvas. This makes it much more suitable for creative upscaling, where the goal is not only to sharpen textures, but also to create or improve **local structures** without producing obvious seams, conflicting forms, or tile-by-tile color drift.

---

## ✨ What this node pack is for

ARMD is designed for workflows where a single global prompt is not enough.

Typical use cases include:

- **Creative upscaling** of existing images with region-specific prompts
- **Regional prompt automation** using captioners / VLMs
- **Manual per-region prompting** for local control
- **Shared-canvas denoising** with stronger inter-tile coherence than independent tiled img2img
- **Blank-canvas multi-prompt generation** using a region plan and denoise `1.0`

This is especially useful for:

- old AI images that need a more intelligent upscale
- artworks that benefit from local reinterpretation
- scenes with multiple semantic zones
- compositions where each tile should “know” what it is supposed to become

---

## 🧠 Core idea behind ARMD

Many tiled upscaling workflows split an image into tiles, run each tile separately, and recombine the results. That can work for low creativity settings, but once the model starts changing local structure, tiles often stop agreeing with each other.

That usually leads to:

- visible seams
- mismatched local structures
- different color decisions across neighboring tiles
- objects or patterns that do not align at recombination boundaries

ARMD takes a different route.

It uses a **shared-canvas regional denoising strategy** inspired by Mixture of Diffusers style workflows, while replacing prompt uniformity with **regional conditioning**.

The result is a workflow where:

- neighboring regions communicate better during denoising
- each region can receive its own prompt
- captioners can automate regional prompt creation
- ControlNet can be added as an optional structural guide
- the output is better suited for **creative** upscaling rather than simple restoration

---

## ✅ Why this can work better than independent tiled img2img

ARMD is not just “tile upscaling with prompts”.

The difference is where the main bottleneck is addressed.

In classic tiled img2img:

- each tile invents structure largely on its own
- recombination happens afterward
- even with overlap and seam reduction tricks, tiles may still disagree structurally
- color drift across tiles is common, especially in backgrounds

In ARMD:

- denoising is performed with a regional shared-canvas logic
- tile boundaries are blended through weighted accumulation
- prompts are assigned region by region
- the workflow is better able to maintain coherence while still allowing creative structural change

This is why ARMD is particularly valuable at **higher denoise values**.

Low denoise tiled workflows can already look acceptable in many cases because the model is not changing enough structure to expose their weaknesses. ARMD becomes most useful when you want to unlock a higher level of reinterpretation, detail invention, and local structural enhancement.

---

## 🧩 Included nodes

### 🧭 Egregora Region Plan
Builds the regional layout used by ARMD.

It:
- aligns the image to a diffusion-friendly working size
- divides the image into regions
- outputs the aligned image
- outputs the list of region crops
- outputs the region plan metadata
- outputs a region order text reference

This node is the backbone of the whole workflow.

### 🧠 Egregora Regional Conditioning
Encodes one prompt per region.

It creates:
- regional positive conditioning
- regional negative conditioning
- placeholder positive conditioning
- placeholder negative conditioning

These placeholders are useful when another node expects a standard conditioning object even though the real conditioning logic is handled regionally.

### 🌐 Egregora Adaptive Diffusion Apply
Applies ARMD to a model.

This is the main node that patches the model so regional conditioning can be used during denoising.

It:
- reads the region plan
- reads the per-region conditioning
- builds regional batches
- runs weighted shared-canvas accumulation
- optionally accepts runtime payload adapters

### 🔎 Egregora Region Select
Lets you preview or inspect a single planned region.

Useful for:
- debugging
- manual prompt writing
- captioning workflows
- checking region order

### 🧩 Egregora Spatial Tensor Pack
Packs spatial tensors into a runtime payload adapter.

Useful for advanced workflows that want to pass extra spatial inputs during runtime.

### 📦 Egregora Static Payload Pack
Packs static values into a runtime payload adapter.

### 🔗 Egregora Runtime Adapter Merge
Merges multiple runtime payload adapters.

### 📐 Egregora Restore Original Size
Restores the final image back to the original framing after `pad_reflect` alignment.

This is important because ARMD may internally pad the image to make the latent canvas compatible with region planning, but you usually still want the final image in the original size and framing.

---

## 🖼️ Alignment modes

ARMD intentionally keeps alignment simple.

### `pad_reflect` ✅ recommended
This is the default and recommended mode.

It:
- preserves the original image content
- avoids deformation
- avoids cropping away image content
- pads only where needed so the image can be processed safely

This is the best general-purpose option for most users.

### `floor_crop`
This crops the image down to the nearest compatible size.

Use it only if you explicitly prefer cropping over padding.

---

## ✍️ Manual prompt input rule

**ARMD reads regional prompts line by line.**

That means:

- **one non-empty line = one region prompt**
- blank lines are ignored
- the number of positive prompts must match the number of regions exactly

Example:

```text
prompt for region 1
prompt for region 2
prompt for region 3
prompt for region 4
```

If your region plan has 4 regions, you must provide 4 positive prompt lines.

### Negative prompts
Negative prompts can be used in three ways:

- leave empty → ARMD will use empty negatives for all regions
- provide **one** negative line → that single negative will be reused for all regions
- provide one negative line per region

---

## 🤖 Captioner / VLM support

Captioners are **optional**.

ARMD works with:
- fully manual prompts
- prompts generated by external captioners
- VLM-based region descriptions
- hybrid workflows where you edit VLM outputs before encoding

If your captioner outputs **one caption per line**, it can usually connect directly to **Egregora Regional Conditioning**.

If it outputs one long block of text instead, you must first reformat that text into **newline-separated prompts**.

In practice, ARMD works especially well with region-first captioning workflows:
1. split image into regions
2. caption each region
3. send the newline-separated result to Regional Conditioning
4. run ARMD

---

## 🎛️ ControlNet support

ControlNet is **optional**, but often useful.

It can help preserve:
- structure
- composition
- edges
- depth logic
- local guidance from the source image

Examples of useful structural controls include:
- tile-based ControlNet
- canny
- depth
- line / edge based preprocessors

Important note:
ControlNet is not a replacement for regional prompting.

A global prompt plus ControlNet can still produce local hallucinations. ARMD is strongest when ControlNet is used as **structural support**, while regional prompts define what each part of the image should become.

In many workflows:
- **ARMD without ControlNet** is already better than global-prompt MoD-style upscale
- **ARMD with ControlNet** is even better when the base image structure matters

---

## 🔍 What “creative upscaling” means here

Creative upscaling is not just about generating sharper textures.

In this project, it means allowing the model to improve or reinterpret **local structures** in the image, such as:

- clothing forms
- objects
- architectural details
- background elements
- texture transitions that imply new structure
- shape refinement at meaningful semantic locations

This is why ARMD becomes much more valuable at medium and high denoise values.

At very low denoise, many tiled methods can look acceptable because they barely change the image.  
At higher denoise, their weaknesses become obvious.

ARMD is designed for the stage where you want:
- more detail
- more structure
- more semantic control
- fewer regional conflicts

---

## 🧪 Two main workflow modes

### 1. Creative upscaling from a source image
This is the main use case.

Typical logic:
- load image
- create region plan
- generate prompts manually or with a captioner
- optionally apply ControlNet
- run ARMD
- restore original size

For this mode, denoise is usually **below 1.0**, depending on how much reinterpretation you want.

### 2. Blank-canvas regional generation
ARMD can also be used for multi-prompt image generation from scratch.

In this case:
- use a blank or minimal base image to define aspect ratio and spatial layout
- create a region plan
- assign prompts region by region
- run with **denoise = 1.0**

This turns ARMD into a practical shared-canvas multi-prompt composition workflow.

---

## 🔧 Recommended starting workflow

A simple starting pipeline is:

1. **Load image**
2. **Resize to target working size**
3. **Egregora Region Plan**
4. **Caption each region** or write prompts manually
5. **Egregora Regional Conditioning**
6. **Egregora Adaptive Diffusion Apply**
7. **KSampler**
8. **VAE Decode Tiled**
9. **Egregora Restore Original Size**

Optional additions:
- ControlNet
- color consistency tools
- runtime payload adapters
- caption cleanup / manual prompt edits

---

## 📌 Notes on tiled VAE encode / decode

Tiled VAE encode and decode are complementary tools in ARMD workflows.

They are useful because they:
- help process large images safely
- reduce memory pressure
- keep VAE conversion more practical at higher resolutions

But tiled VAE alone does **not** solve the semantic coordination problem between regions.

ARMD addresses that missing piece by adding region-aware conditioning on top of a shared-canvas denoising strategy.

---

## 🛠️ Related practical tools

ARMD was developed in the context of practical experimentation with tiled creative upscaling workflows.

Related tools worth knowing include:

- **[ComfyUI-Egregora-Divide-And-Enhance](https://github.com/lucasgattas/ComfyUI-Egregora-Divide-And-Enhance)**  
  An earlier approach for tiled creative upscaling with prompt lists and per-tile prompting, but still based on separate img2img-style tile generation and recombination.

- **[ComfyUI-Egregora-Adaptive-Colorfix](https://github.com/lucasgattas/ComfyUI-Egregora-Adaptive-Colorfix)**  
  A previous attempt to improve per-tile color consistency before recombination.

- **[comfyui-colorfix-v3](https://github.com/ihorpankin/comfyui-colorfix-v3)**  
  Useful especially for models that tend to reinterpret global palette and contrast more aggressively.

- **[ComfyUI_UltimateSDUpscaleGuider](https://github.com/Blakeem/ComfyUI_UltimateSDUpscaleGuider)**  
  Adds improvements to Ultimate SD Upscale style workflows, including context-oriented overlap strategies.

These tools are relevant because ARMD is best understood not as a rejection of prior tiled practice, but as a refinement of where the real bottleneck lies.

Once seam blending is improved, the next major problem is no longer only border visibility.  
It becomes the question of how neighboring regions can invent the **right structure together**.

---

## 📚 Research background

ARMD is inspired by practical gaps between public tools and shared-canvas diffusion research.

Important references include:
- MultiDiffusion
- Mixture of Diffusers
- DemoFusion
- SpotDiffusion
- region-captioning and regional super-resolution papers such as C-Upscale and RAGSR

ARMD does **not** claim to invent all of these ideas from scratch.

Its purpose is to make this kind of regional conditioning workflow **usable in practice**, especially for:
- artists
- creative developers
- image pipeline builders
- advanced ComfyUI users

---

## ✅ Current status

ARMD is already usable for practical experimentation.

It has been tested primarily with:
- **SDXL**
- **Z-Image Turbo**

Other backbones may work too, but behavior can differ depending on:
- model architecture
- color reinterpretation strength
- ControlNet compatibility
- denoise range

---

## 💛 Acknowledgements

This project was developed through practical experimentation inside the wider ComfyUI ecosystem.

Thanks to the open-source diffusion community for building the tools, workflows, discussions, and research foundations that make this kind of work possible.

---

## 🔗 Repository

**GitHub:** [lucasgattas/ComfyUI-Egregora-ARMD](https://github.com/lucasgattas/ComfyUI-Egregora-ARMD)
