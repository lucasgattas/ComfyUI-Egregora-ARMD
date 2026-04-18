# Attribution

## Algorithmic basis

Egregora-ARMD is an independent implementation of region-based diffusion orchestration for ComfyUI workflows. Its algorithmic inspiration comes from academic work on multi-region diffusion and canvas-level composition, especially:

1. **MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation**  
   Omer Bar-Tal, Lior Yariv, Yaron Lipman, Tali Dekel.  
   arXiv:2302.08113, 2023.

2. **Mixture of Diffusers for scene composition and high resolution image generation**  
   Álvaro Barbero Jiménez.  
   arXiv:2302.02412, 2023.

These references are cited as conceptual and algorithmic background for tiled / regional diffusion, multi-window denoising, and spatially controlled composition.

## Scope of attribution

This attribution is provided to acknowledge the academic ideas that informed the design goals of this package.

It does **not** imply that this repository is a redistribution of any third-party implementation. The package is intended to stand on its own as a ComfyUI custom-node implementation with its own runtime layout, batching logic, regional planning, and workflow integration.

## Recommended citation text

If you reference Egregora-ARMD in documentation, presentations, or derivative technical material, cite the underlying academic basis as:

- Bar-Tal, O., Yariv, L., Lipman, Y., & Dekel, T. (2023). *MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation*. arXiv:2302.08113.
- Barbero Jiménez, Á. (2023). *Mixture of Diffusers for scene composition and high resolution image generation*. arXiv:2302.02412.

## Practical note

This file is intentionally focused on paper-level attribution and conceptual lineage. Repository-specific installation, usage, node catalog, and workflow examples should live in the project README and example workflows.
