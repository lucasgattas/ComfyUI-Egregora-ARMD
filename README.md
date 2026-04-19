# 🌐 Egregora ARMD

**ARMD** stands for **Adaptive Regional Mixture of Diffusers**.

Egregora ARMD is a regional diffusion workflow for **semantic upscaling**, **multi-prompt image generation**, and **adaptive regional reconstruction** in ComfyUI.

It is designed around a simple idea: keep the benefits of **shared-canvas denoising**, but give different regions of the image **different prompts** without letting those prompts fight over the same space.

Instead of treating each tile as a fully independent img2img job, ARMD uses:

- **core regions** for local semantic ownership
- **context regions** so neighboring areas can still inform each other
- **feathered write-back** to reduce seams during recombination

This makes it especially useful for creative upscaling, regional prompting, and structured scene generation.

---

## ✨ What this node pack is for

ARMD is meant for workflows where a single global prompt is not enough.

Typical use cases:

- **Creative upscaling** of existing images with per-region prompts
- **Regional prompt automation** using captioners / VLMs
- **Manual per-region prompting**
- **Shared-canvas denoising** with better local coherence than independent tiled img2img
- **Blank-canvas multi-prompt generation** using an Empty Latent or aligned base image

This is useful when you want to create or improve:

- skies, clouds, sun, mountains, forests, gardens, buildings, vehicles, subjects, or objects in different parts of the same image
- scenes that need different semantic instructions across the frame
- outputs that should remain coherent without strong tile-by-tile disagreement

---

## 🧠 Core idea behind ARMD

Many tiled workflows improve memory usage and resolution, but they often fail when different tiles start inventing structure independently.

That usually creates:

- visible seams
- ghosting
- inconsistent structures across tile boundaries
- tile-by-tile color drift
- conflicting semantic decisions in overlap zones

ARMD addresses this by separating **context** from **ownership**.

Each region has:

- a **core region**: the area that actually belongs to that prompt
- a **context region**: an expanded area that gives surrounding information to the model
- a **write-back region** with feathering: so the tile can blend back into the canvas smoothly without taking full control of neighboring semantic zones

This makes ARMD much more suitable for **creative** diffusion workflows than naive tiled recombination.

---

## ✅ Why this works better than independent tile generation

ARMD is not just “tile upscaling with prompts”.

In classic tiled img2img:

- tiles are often generated too independently
- overlap zones may receive different structures from different prompts
- recombination tries to blend incompatible hypotheses
- ghosting and seams appear when prompts diverge too much

In ARMD:

- prompts stay tied to **local core regions**
- tiles still see more context through expanded context crops
- write-back is feathered rather than fully competitive
- neighboring tiles can remain aware of each other without mixing prompt ownership too aggressively

That balance is what makes ARMD effective both for:

- **upscaling from a source image**
- **generation from scratch**

---

## 🧩 Included nodes

### 🧭 Egregora Region Plan
Builds the spatial layout used by ARMD.

It:
- aligns the working canvas
- divides the image into **core regions**
- expands them into **context regions**
- prepares batching metadata
- outputs region order text for prompt assignment
- optionally works from an **IMAGE**, a **LATENT**, or both

This node is the backbone of the workflow.

### 🧠 Egregora Regional Conditioning
Encodes one prompt per region.

It creates:
- regional positive conditioning
- regional negative conditioning
- placeholder positive conditioning
- placeholder negative conditioning

These placeholders are useful for compatibility with nodes that still expect standard conditioning objects.

### 🌐 Egregora Adaptive Diffusion Apply
Applies ARMD to a model.

This is the main node that patches the model so regional conditioning can be used during denoising.

It:
- reads the region plan
- reads per-region conditioning
- builds regional runtime batches
- uses context regions for reading
- writes back using feathered local ownership

### 🔎 Egregora Region Select
Lets you inspect a single region for:
- manual prompt writing
- captioning workflows
- debugging region order

### 🧩 Egregora Spatial Tensor Pack
Packs spatial tensors into a runtime payload adapter.

Useful for advanced workflows that pass extra runtime tensors.

### 📦 Egregora Static Payload Pack
Packs static runtime values into a runtime payload adapter.

### 🔗 Egregora Runtime Adapter Merge
Merges multiple runtime payload adapters.

### 📐 Egregora Restore Original Size
Restores the final image back to the original framing after `pad_reflect` alignment.

This is useful because ARMD may internally pad the image to a safer working canvas, while you still want the final output to match the original framing.

---

## 🖼️ Alignment modes

ARMD keeps alignment intentionally simple.

### `pad_reflect` ✅ recommended
This is the default and recommended mode.

It:
- preserves the original framing better
- avoids deformation
- avoids discarding image content
- pads only where necessary

This is the safest general-purpose option.

### `floor_crop`
This crops the image down to the nearest compatible size.

Use it only if you explicitly prefer cropping over padding.

---

## 📏 Region planning and defaults

ARMD now works best with a **core + context + feather** strategy.

Recommended starting values:

- `region_width = 1024`
- `region_height = 1024`
- `region_overlap = 384`
- `blend_feather = 64`

How to think about them:

- **region_width / region_height** define the region scale
- **region_overlap** acts as **context padding**
- **blend_feather** controls the soft transition on write-back

Practical rule of thumb:

- context overlap around **1/3 to 3/8** of tile size
- feather around **1/16 to 1/8** of tile size

Examples:
- for `1024` tiles → `384 / 64`
- for `768` tiles → `256 / 48` or `256 / 64`
- for `512` tiles → `192 / 32` or `192 / 48`

---

## ✍️ Manual prompt input rule

**ARMD reads regional prompts line by line.**

That means:

- **one non-empty line = one region prompt**
- blank lines are ignored
- the number of positive prompts must match the number of regions exactly

Example:

```text
the sky
the sky
the sun
a beautiful mansion house
a beautiful garden
a beautiful forest
grass and flowers
a nice white ferrari on a grassy field
grass
```

If your region plan produces 9 regions, you must provide 9 positive prompt lines.

### Negative prompts
Negative prompts can be used in three ways:

- leave empty → ARMD uses empty negatives for all regions
- provide **one** negative line → it is reused for all regions
- provide one negative line per region

---

## 🤖 Captioner / VLM support

Captioners are **optional**.

ARMD works with:
- manual prompts
- external captioners
- VLM-based regional descriptions
- hybrid workflows where VLM output is edited manually before encoding

If your captioner outputs **one caption per line**, it can usually connect directly to **Egregora Regional Conditioning**.

If it outputs one large paragraph or one single string, reformat it into newline-separated prompts first.

A typical regional caption workflow is:

1. build a region plan
2. inspect or batch the regions
3. caption each region
4. send the newline-separated captions to Regional Conditioning
5. run ARMD

---

## 🎛️ ControlNet support

ControlNet is **optional**, but often useful.

It can help preserve:
- composition
- edge structure
- subject pose
- depth logic
- local geometry from a source image

Examples of useful controls:
- tile
- canny
- depth
- other structural preprocessors

Important:
ControlNet does **not** replace regional prompting.

A global prompt plus ControlNet can still produce local hallucinations. ARMD works best when:

- regional prompts define what each zone should become
- ControlNet acts as structural support

In many cases:
- **ARMD without ControlNet** is already stronger than a global-prompt tiled workflow
- **ARMD with ControlNet** is stronger still when source structure matters

---

## 🔍 What “creative upscaling” means here

Creative upscaling is not only about adding texture.

In ARMD, it means allowing the model to improve or reinterpret **local structures**, such as:

- clouds, sun, sky transitions
- buildings and architectural details
- gardens, vegetation, terrain
- vehicles and objects
- semantic transitions between neighboring regions

This is why ARMD becomes especially valuable beyond very low denoise values.

At low denoise, many tiled workflows can appear acceptable because they are not changing enough to expose their weaknesses.  
At higher denoise, independent or poorly coordinated tiles tend to break down much more easily.

ARMD is aimed at the point where you want:
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
- optionally resize to a working target
- create region plan
- generate prompts manually or with a captioner
- optionally apply ControlNet
- run ARMD
- restore original size

For this mode, denoise is usually **below 1.0**, depending on how much reinterpretation you want.

### 2. Blank-canvas regional generation
ARMD can also be used for multi-prompt image generation from scratch.

This now works best by using:
- an **Empty Latent** to define canvas resolution
- optionally a base/noise image for canvas guidance if desired
- a region plan aligned to that latent canvas
- one prompt per region

For true from-scratch generation, use:

- **denoise = 1.0**
- a latent-driven canvas
- regional prompts assigned line by line

---

## 🧱 IMAGE and LATENT input support in Region Plan

`Egregora Region Plan` can now work with:

- **IMAGE only**
- **LATENT only**
- **IMAGE + LATENT**

Why this matters:

- for **upscaling**, IMAGE-only is often enough
- for **generation from scratch**, LATENT is often the best source of truth for canvas size
- for mixed workflows, IMAGE + LATENT helps keep resolution synchronized between visual reference and diffusion canvas

If a LATENT is provided, ARMD can use it to define the exact working resolution even when there is no strong source image.

---

## 🔧 Recommended starting workflow

### Upscaling workflow
1. **Load image**
2. optionally resize to working target
3. **Egregora Region Plan**
4. caption each region or write prompts manually
5. **Egregora Regional Conditioning**
6. optionally add **ControlNet**
7. **Egregora Adaptive Diffusion Apply**
8. **KSampler**
9. **VAE Decode Tiled**
10. **Egregora Restore Original Size**

### From-scratch generation workflow
1. **Empty Latent Image**
2. optionally create or pass a base/noise image
3. **Egregora Region Plan**
4. prepare one prompt per region
5. **Egregora Regional Conditioning**
6. **Egregora Adaptive Diffusion Apply**
7. **KSampler** with `denoise = 1.0`
8. **VAE Decode Tiled**

---

## 📌 Notes on tiled VAE encode / decode

Tiled VAE encode and decode remain complementary tools in ARMD workflows.

They help:
- process large images safely
- reduce memory pressure
- keep VAE conversion practical at higher resolutions

But tiled VAE alone does **not** solve the semantic coordination problem between regions.

ARMD addresses that by combining:
- region planning
- regional conditioning
- context-aware denoising
- controlled write-back

---

## 📚 Research background

ARMD is inspired by practical gaps between public tools and shared-canvas diffusion research.

Important references include:
- MultiDiffusion
- Mixture of Diffusers
- DemoFusion
- SpotDiffusion
- regional captioning and regional super-resolution work such as C-Upscale and RAGSR

ARMD does **not** claim to invent all of these ideas from scratch.

Its purpose is to make this kind of regional conditioning workflow **usable in practice**, especially for:
- artists
- creative developers
- image pipeline builders
- advanced ComfyUI users

---

## 💛 Acknowledgements

This project was developed through practical experimentation inside the wider ComfyUI ecosystem.

Thanks to the open-source diffusion community for building the tools, workflows, discussions, and research foundations that make this kind of work possible.
