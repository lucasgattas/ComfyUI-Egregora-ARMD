from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence
import math

import torch
import torch.nn.functional as F
from comfy.model_patcher import ModelPatcher
from comfy.utils import repeat_to_batch_size


# ============================================================
# version
# ============================================================

__version__ = "0.2.0"


# ============================================================
# math primitives
# ============================================================

def ceildiv(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("b must be non-zero")
    return -(a // -b)



def floor_multiple(v: int, mult: int) -> int:
    if mult <= 0:
        raise ValueError("mult must be positive")
    return max(mult, int(math.floor(v / mult)) * mult)


def ceil_multiple(v: int, mult: int) -> int:
    if mult <= 0:
        raise ValueError("mult must be positive")
    return max(mult, int(math.ceil(v / mult)) * mult)



def axis_positions(full_size: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if tile_size > full_size:
        return [0]
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("tile_size must be larger than overlap")

    count = ceildiv(max(1, full_size - overlap), stride)
    if count <= 1:
        return [0]

    last_start = full_size - tile_size
    positions = []
    for i in range(count):
        alpha = i / (count - 1)
        pos = int(round(alpha * last_start))
        positions.append(pos)
    return positions



def split_bounds(full_size: int, target_size: int) -> list[int]:
    if full_size <= 0:
        raise ValueError("full_size must be positive")
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    count = max(1, ceildiv(full_size, target_size))
    return [int(round((i * full_size) / count)) for i in range(count + 1)]


def expand_bbox_clamped(bbox: RegionBBox, pad_x: int, pad_y: int, full_w: int, full_h: int) -> RegionBBox:
    x1 = max(0, bbox.x - pad_x)
    y1 = max(0, bbox.y - pad_y)
    x2 = min(full_w, bbox.x2 + pad_x)
    y2 = min(full_h, bbox.y2 + pad_y)
    return RegionBBox(x=x1, y=y1, w=max(1, x2 - x1), h=max(1, y2 - y1))


# ============================================================
# region planning
# ============================================================

@dataclass(frozen=True)
class RegionBBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def box(self) -> list[int]:
        return [self.x, self.y, self.x2, self.y2]



@dataclass(frozen=True)
class RegionPlan:
    original_width: int
    original_height: int
    aligned_width: int
    aligned_height: int
    latent_width: int
    latent_height: int
    compression: int
    alignment_mode: str
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    region_width: int
    region_height: int
    region_overlap: int
    region_width_latent: int
    region_height_latent: int
    region_overlap_latent: int
    blend_feather: int
    blend_feather_latent: int
    rows: int
    cols: int
    bboxes_latent: list[RegionBBox]
    bboxes_pixel: list[RegionBBox]
    core_bboxes_latent: list[RegionBBox]
    core_bboxes_pixel: list[RegionBBox]
    write_bboxes_latent: list[RegionBBox]
    write_bboxes_pixel: list[RegionBBox]

    @property
    def region_count(self) -> int:
        return len(self.bboxes_pixel)


def image_bhwc_to_bchw(img: torch.Tensor) -> torch.Tensor:
    return img.movedim(-1, 1)


def image_bchw_to_bhwc(img: torch.Tensor) -> torch.Tensor:
    return img.movedim(1, -1)


def resize_bhwc(image: torch.Tensor, width: int, height: int, mode: str = "bilinear") -> torch.Tensor:
    x = image_bhwc_to_bchw(image)
    x = F.interpolate(
        x,
        size=(height, width),
        mode=mode,
        align_corners=False if mode in ("bilinear", "bicubic") else None,
    )
    return image_bchw_to_bhwc(x)


def pad_bhwc(
    image: torch.Tensor,
    *,
    pad_left: int,
    pad_right: int,
    pad_top: int,
    pad_bottom: int,
    mode: str = "reflect",
    value: float = 0.0,
) -> torch.Tensor:
    """Pad a BHWC image while staying compatible with older call sites.

    Supported modes:
    - "reflect": default and recommended; falls back to "replicate" when
      PyTorch reflect padding would be invalid for the current image/pad sizes.
    - "replicate" / "edge": edge padding.
    - "constant": constant-value padding.

    The simplified ARMD UI only exposes reflect-based padding, but keeping these
    arguments prevents older internal calls or saved workflows from breaking.
    """
    if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
        return image

    x = image_bhwc_to_bchw(image)
    _, _, h, w = x.shape
    pad = (pad_left, pad_right, pad_top, pad_bottom)

    normalized_mode = str(mode).lower().strip()
    if normalized_mode == "edge":
        normalized_mode = "replicate"

    if normalized_mode == "reflect":
        reflect_invalid = (
            h < 2 or w < 2 or pad_top >= h or pad_bottom >= h or pad_left >= w or pad_right >= w
        )
        torch_mode = "replicate" if reflect_invalid else "reflect"
        x = F.pad(x, pad, mode=torch_mode)
    elif normalized_mode == "replicate":
        x = F.pad(x, pad, mode="replicate")
    elif normalized_mode == "constant":
        x = F.pad(x, pad, mode="constant", value=float(value))
    else:
        raise ValueError(f"unsupported pad mode: {mode}")

    return image_bchw_to_bhwc(x)



def align_image_to_compression(
    image: torch.Tensor,
    compression: int,
    alignment_mode: str,
) -> tuple[torch.Tensor, dict[str, int | str]]:
    _, h, w, _ = image.shape

    # Keep legacy values loadable for old workflows, but simplify the public UI
    # to the two modes that are actually useful in practice.
    if alignment_mode in {"pad_bottom_right_reflect", "pad_symmetric_reflect"}:
        normalized_mode = "pad_reflect"
    elif alignment_mode == "floor_crop":
        normalized_mode = "floor_crop"
    else:
        normalized_mode = alignment_mode

    meta: dict[str, int | str] = {
        "original_width": w,
        "original_height": h,
        "alignment_mode": normalized_mode,
        "pad_left": 0,
        "pad_top": 0,
        "pad_right": 0,
        "pad_bottom": 0,
    }

    if normalized_mode == "floor_crop":
        new_w = floor_multiple(w, compression)
        new_h = floor_multiple(h, compression)
        return image[:, :new_h, :new_w, :], meta

    if normalized_mode != "pad_reflect":
        raise ValueError(f"unsupported alignment_mode: {alignment_mode}")

    pad_w = (ceil_multiple(w, compression) - w) % compression
    pad_h = (ceil_multiple(h, compression) - h) % compression
    pad_left = 0
    pad_top = 0
    pad_right = pad_w
    pad_bottom = pad_h

    meta.update({
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom,
    })

    aligned = pad_bhwc(
        image,
        pad_left=pad_left,
        pad_right=pad_right,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        mode="reflect",
        value=0.0,
    )
    return aligned, meta



def build_region_plan(
    aligned_width: int,
    aligned_height: int,
    region_width: int,
    region_height: int,
    region_overlap: int,
    compression: int,
    *,
    blend_feather: int = 64,
    original_width: int | None = None,
    original_height: int | None = None,
    alignment_mode: str = "unknown",
    pad_left: int = 0,
    pad_top: int = 0,
    pad_right: int = 0,
    pad_bottom: int = 0,
) -> RegionPlan:
    if region_width % compression != 0:
        raise ValueError("region_width must be divisible by compression")
    if region_height % compression != 0:
        raise ValueError("region_height must be divisible by compression")
    if region_overlap % compression != 0:
        raise ValueError("region_overlap must be divisible by compression")
    if blend_feather % compression != 0:
        raise ValueError("blend_feather must be divisible by compression")

    latent_width = aligned_width // compression
    latent_height = aligned_height // compression

    rw_l_target = min(max(1, region_width // compression), latent_width)
    rh_l_target = min(max(1, region_height // compression), latent_height)
    context_pad_l = max(0, region_overlap // compression)
    blend_feather_l = max(0, blend_feather // compression)

    effective_region_width = rw_l_target * compression
    effective_region_height = rh_l_target * compression
    effective_region_overlap = context_pad_l * compression
    effective_blend_feather = blend_feather_l * compression

    x_bounds = split_bounds(latent_width, rw_l_target)
    y_bounds = split_bounds(latent_height, rh_l_target)
    cols = len(x_bounds) - 1
    rows = len(y_bounds) - 1

    bboxes_latent: list[RegionBBox] = []
    bboxes_pixel: list[RegionBBox] = []
    core_bboxes_latent: list[RegionBBox] = []
    core_bboxes_pixel: list[RegionBBox] = []
    write_bboxes_latent: list[RegionBBox] = []
    write_bboxes_pixel: list[RegionBBox] = []

    for row in range(rows):
        for col in range(cols):
            core_lat = RegionBBox(
                x=x_bounds[col],
                y=y_bounds[row],
                w=max(1, x_bounds[col + 1] - x_bounds[col]),
                h=max(1, y_bounds[row + 1] - y_bounds[row]),
            )
            context_lat = expand_bbox_clamped(core_lat, context_pad_l, context_pad_l, latent_width, latent_height)
            write_lat = expand_bbox_clamped(core_lat, blend_feather_l, blend_feather_l, latent_width, latent_height)

            core_pix = RegionBBox(
                x=core_lat.x * compression,
                y=core_lat.y * compression,
                w=core_lat.w * compression,
                h=core_lat.h * compression,
            )
            context_pix = RegionBBox(
                x=context_lat.x * compression,
                y=context_lat.y * compression,
                w=context_lat.w * compression,
                h=context_lat.h * compression,
            )
            write_pix = RegionBBox(
                x=write_lat.x * compression,
                y=write_lat.y * compression,
                w=write_lat.w * compression,
                h=write_lat.h * compression,
            )

            bboxes_latent.append(context_lat)
            bboxes_pixel.append(context_pix)
            core_bboxes_latent.append(core_lat)
            core_bboxes_pixel.append(core_pix)
            write_bboxes_latent.append(write_lat)
            write_bboxes_pixel.append(write_pix)

    return RegionPlan(
        original_width=aligned_width if original_width is None else original_width,
        original_height=aligned_height if original_height is None else original_height,
        aligned_width=aligned_width,
        aligned_height=aligned_height,
        latent_width=latent_width,
        latent_height=latent_height,
        compression=compression,
        alignment_mode=alignment_mode,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        region_width=effective_region_width,
        region_height=effective_region_height,
        region_overlap=effective_region_overlap,
        region_width_latent=rw_l_target,
        region_height_latent=rh_l_target,
        region_overlap_latent=context_pad_l,
        blend_feather=effective_blend_feather,
        blend_feather_latent=blend_feather_l,
        rows=rows,
        cols=cols,
        bboxes_latent=bboxes_latent,
        bboxes_pixel=bboxes_pixel,
        core_bboxes_latent=core_bboxes_latent,
        core_bboxes_pixel=core_bboxes_pixel,
        write_bboxes_latent=write_bboxes_latent,
        write_bboxes_pixel=write_bboxes_pixel,
    )



def region_order_text(plan: RegionPlan) -> str:
    lines = []
    for idx, (cpb, clb, kpb, klb, wpb, wlb) in enumerate(
        zip(
            plan.bboxes_pixel,
            plan.bboxes_latent,
            plan.core_bboxes_pixel,
            plan.core_bboxes_latent,
            plan.write_bboxes_pixel,
            plan.write_bboxes_latent,
        ),
        start=1,
    ):
        row = (idx - 1) // plan.cols
        col = (idx - 1) % plan.cols
        lines.append(
            f"region {idx}: row={row}, col={col}, "
            f"core_pixel_bbox={kpb.box}, core_latent_bbox={klb.box}, "
            f"write_pixel_bbox={wpb.box}, write_latent_bbox={wlb.box}, "
            f"context_pixel_bbox={cpb.box}, context_latent_bbox={clb.box}"
        )
    return "\n".join(lines)


def extract_region_images(image: torch.Tensor, plan: RegionPlan, *, use_core: bool = True) -> list[torch.Tensor]:
    bboxes = plan.core_bboxes_pixel if use_core else plan.bboxes_pixel
    out = []
    for bbox in bboxes:
        out.append(image[:, bbox.y:bbox.y2, bbox.x:bbox.x2, :])
    return out


def stack_region_images_with_padding(regions: Sequence[torch.Tensor], *, pad_mode: str = "edge") -> torch.Tensor:
    if not regions:
        raise ValueError("regions must be non-empty")
    max_h = max(int(r.shape[1]) for r in regions)
    max_w = max(int(r.shape[2]) for r in regions)
    padded = []
    for region in regions:
        _, h, w, _ = region.shape
        pad_right = max_w - int(w)
        pad_bottom = max_h - int(h)
        if pad_right or pad_bottom:
            region = pad_bhwc(
                region,
                pad_left=0,
                pad_right=pad_right,
                pad_top=0,
                pad_bottom=pad_bottom,
                mode=pad_mode,
                value=0.0,
            )
        padded.append(region)
    return torch.cat(padded, dim=0)


# ============================================================
# regional conditioning
# ============================================================

def collapse_prompt_text(s: str) -> str:
    return " ".join(str(s).split()).strip()


def normalize_prompt_input(raw: Any) -> list[str]:
    def _normalize_one(value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, str):
            return [line.strip() for line in value.splitlines() if line.strip()]

        if isinstance(value, (list, tuple)):
            out: list[str] = []
            for item in value:
                out.extend(_normalize_one(item))
            return out

        t = collapse_prompt_text(str(value))
        return [t] if t else []

    return _normalize_one(raw)


def pad_crossattn_to_length(t: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad a cross-attention tensor to target_len along the sequence axis.

    Uses zero-padding so that existing token semantics are unchanged.
    This replaces the previous repeat strategy which caused semantic
    duplication and could trigger OOM when the LCM of mixed prompt
    lengths grew very large.
    """
    if t.shape[1] == target_len:
        return t
    if t.shape[1] > target_len:
        return t[:, :target_len, :]
    pad_len = target_len - t.shape[1]
    return F.pad(t, (0, 0, 0, pad_len))


def conditioning_to_local_entry(conditioning: Any) -> dict[str, Any]:
    """Extract c_crossattn and pooled_output from a ComfyUI conditioning list.

    Stores tensors on their current device (no .cpu() transfer) to avoid
    repeated CPU↔GPU copies at every denoising step.

    The 'extra' dict preserves pooled_output (SDXL CLIP-G pool) and any
    other fields returned by CLIPTextEncode so they can be injected as
    per-region conditioning in the UNet call.
    """
    if not isinstance(conditioning, list) or len(conditioning) == 0:
        raise ValueError("unexpected conditioning format")
    base = conditioning[0]
    if not isinstance(base, (list, tuple)) or len(base) < 1:
        raise ValueError("unexpected conditioning entry format")
    cross = base[0]
    if not isinstance(cross, torch.Tensor):
        raise ValueError("conditioning c_crossattn tensor not found")
    extra: dict[str, Any] = {}
    if len(base) > 1 and isinstance(base[1], dict):
        for k, v in base[1].items():
            extra[k] = v.detach() if isinstance(v, torch.Tensor) else v
    return {"c_crossattn": cross.detach(), "extra": extra}


# ============================================================
# runtime adapters
# ============================================================

class RuntimePayloadAdapter(Protocol):
    def build_payload(self, latent, timestep, cond_dict) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class StaticRuntimePayloadAdapter:
    payload: dict[str, Any]

    def build_payload(self, latent, timestep, cond_dict) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True)
class CompositeRuntimePayloadAdapter:
    adapters: tuple[RuntimePayloadAdapter, ...]

    def build_payload(self, latent, timestep, cond_dict) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for adapter in self.adapters:
            payload = adapter.build_payload(latent, timestep, cond_dict)
            if not isinstance(payload, dict):
                raise ValueError("runtime adapter payload must be a dict")
            for key, value in payload.items():
                if key in merged:
                    raise ValueError(f"duplicate runtime payload key: {key}")
                merged[key] = value
        return merged


def merge_runtime_adapters(*adapters: RuntimePayloadAdapter | None) -> RuntimePayloadAdapter:
    live = tuple(a for a in adapters if a is not None)
    if len(live) == 0:
        return StaticRuntimePayloadAdapter({})
    if len(live) == 1:
        return live[0]
    return CompositeRuntimePayloadAdapter(live)


# ============================================================
# spatial controls / adapters
# ============================================================


def scale_region_extent_to_tensor(
    *,
    full_latent_extent: int,
    tensor_extent: int,
    region_extent: int,
) -> int:
    if full_latent_extent <= 0 or tensor_extent <= 0:
        raise ValueError("full_latent_extent and tensor_extent must be positive")
    scaled = int(math.floor((region_extent * tensor_extent) / full_latent_extent))
    return max(1, min(tensor_extent, scaled))


def fixed_count_axis_positions(full_size: int, tile_size: int, count: int) -> list[int]:
    if count <= 1 or tile_size >= full_size:
        return [0]
    last_start = max(0, full_size - tile_size)
    return [int(round((i / (count - 1)) * last_start)) for i in range(count)]


def scale_bbox_list_to_tensor(
    bboxes: Sequence[RegionBBox],
    *,
    latent_height: int,
    latent_width: int,
    tensor_height: int,
    tensor_width: int,
) -> list[RegionBBox]:
    if latent_height <= 0 or latent_width <= 0:
        raise ValueError("latent dimensions must be positive")
    if tensor_height <= 0 or tensor_width <= 0:
        raise ValueError("tensor dimensions must be positive")

    scaled: list[RegionBBox] = []
    for bbox in bboxes:
        x1 = int(math.floor((bbox.x * tensor_width) / latent_width))
        y1 = int(math.floor((bbox.y * tensor_height) / latent_height))
        x2 = int(math.ceil((bbox.x2 * tensor_width) / latent_width))
        y2 = int(math.ceil((bbox.y2 * tensor_height) / latent_height))
        x1 = max(0, min(max(0, tensor_width - 1), x1))
        y1 = max(0, min(max(0, tensor_height - 1), y1))
        x2 = max(x1 + 1, min(tensor_width, x2))
        y2 = max(y1 + 1, min(tensor_height, y2))
        scaled.append(RegionBBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1))
    return scaled


def build_scaled_bboxes_for_plan(
    region_plan: RegionPlan,
    *,
    tensor_height: int,
    tensor_width: int,
) -> list[RegionBBox]:
    return scale_bbox_list_to_tensor(
        region_plan.bboxes_latent,
        latent_height=region_plan.latent_height,
        latent_width=region_plan.latent_width,
        tensor_height=tensor_height,
        tensor_width=tensor_width,
    )


def validate_spatial_tensor(
    tensor: torch.Tensor,
    *,
    name: str = "tensor",
) -> None:
    """Validate tensors whose final two axes represent spatial H and W.

    Supported examples:
    - BCHW
    - BCTHW
    - layouts with additional non-spatial axes before H and W

    ARMD always preserves every leading axis and tiles only the final two
    spatial dimensions.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")

    if tensor.ndim < 4:
        raise ValueError(
            f"{name} must have at least 4 dimensions with H and W "
            f"in the final two axes; received shape {tuple(tensor.shape)}"
        )

    if int(tensor.shape[-2]) <= 0 or int(tensor.shape[-1]) <= 0:
        raise ValueError(
            f"{name} has invalid spatial dimensions: "
            f"{tuple(tensor.shape)}"
        )


def spatial_slice_for_bbox(
    bbox: RegionBBox,
) -> tuple:
    """Build an ellipsis-based slice preserving all leading axes."""
    return (
        ...,
        slice(int(bbox.y), int(bbox.y2)),
        slice(int(bbox.x), int(bbox.x2)),
    )


def expand_spatial_weight_for_tensor(
    weight: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Expand a [1, 1, H, W] weight to match BCHW or BCTHW tensors."""
    validate_spatial_tensor(tensor, name="tensor")

    if weight.ndim < 2:
        raise ValueError(
            f"weight must contain spatial axes; received "
            f"shape {tuple(weight.shape)}"
        )

    while weight.ndim < tensor.ndim:
        weight = weight.unsqueeze(-3)

    if weight.ndim != tensor.ndim:
        raise ValueError(
            "weight rank could not be matched to tensor rank: "
            f"weight={tuple(weight.shape)}, tensor={tuple(tensor.shape)}"
        )

    if (
        int(weight.shape[-2]) != int(tensor.shape[-2])
        or int(weight.shape[-1]) != int(tensor.shape[-1])
    ):
        raise ValueError(
            "weight and tensor spatial dimensions do not match: "
            f"weight={tuple(weight.shape)}, tensor={tuple(tensor.shape)}"
        )

    return weight


def is_spatial_tensor_candidate(
    value: Any,
) -> bool:
    """Return True for tensors that can carry H/W in their final axes."""
    return (
        isinstance(value, torch.Tensor)
        and value.ndim >= 4
        and int(value.shape[-2]) > 0
        and int(value.shape[-1]) > 0
    )


def slice_spatial_tensor_for_region_batch(
    tensor: torch.Tensor,
    batch: "RuntimeRegionBatch",
    region_plan: RegionPlan,
    *,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor:
    validate_spatial_tensor(
        tensor,
        name="spatial tensor",
    )

    if tensor.shape[-2] == latent_height and tensor.shape[-1] == latent_width:
        bboxes = batch.context_bboxes
    else:
        scaled_bboxes = build_scaled_bboxes_for_plan(
            region_plan,
            tensor_height=tensor.shape[-2],
            tensor_width=tensor.shape[-1],
        )
        if len(scaled_bboxes) != region_plan.region_count:
            raise ValueError("scaled bbox count mismatch")
        bboxes = [scaled_bboxes[i] for i in batch.region_indices]

    slices = [
        tensor[spatial_slice_for_bbox(bbox)]
        for bbox in bboxes
    ]
    if not slices:
        raise ValueError("bboxes must be non-empty")
    return torch.cat(slices, dim=0)


def latent_dict_to_tensor(latent: dict[str, Any]) -> torch.Tensor:
    samples = latent.get("samples", None)
    if not isinstance(samples, torch.Tensor):
        raise ValueError("LATENT input must contain tensor under key 'samples'")
    validate_spatial_tensor(
        samples,
        name="LATENT samples tensor",
    )
    return samples


def blank_image_bhwc(width: int, height: int, *, dtype=torch.float32, device=None) -> torch.Tensor:
    return torch.zeros((1, int(height), int(width), 3), dtype=dtype, device=device)


def fit_image_to_exact_canvas(
    image: torch.Tensor,
    target_width: int,
    target_height: int,
    alignment_mode: str,
) -> tuple[torch.Tensor, dict[str, int | str]]:
    """Fit an IMAGE tensor to an exact target canvas when a LATENT defines the canvas size.

    Important behavior:
    - If the aspect ratio already matches the target canvas, resize exactly to the target.
      This keeps IMAGE and LATENT synchronized without introducing mirrored reflect-padding.
    - If the aspect ratio does not match, fail loudly instead of silently distorting or
      creating reflected borders. The user can then resize/crop upstream in a controlled way.

    We intentionally report zero padding here because this path is an exact canvas fit,
    not a padding-based alignment step.
    """
    _, h, w, _ = image.shape
    normalized_mode = "floor_crop" if alignment_mode == "floor_crop" else "exact_canvas_fit"

    if h <= 0 or w <= 0 or target_height <= 0 or target_width <= 0:
        raise ValueError("invalid image or target canvas dimensions")

    src_ratio = float(w) / float(h)
    dst_ratio = float(target_width) / float(target_height)

    # Tolerate up to 1% relative aspect-ratio difference to absorb rounding
    # errors that accumulate in upstream resize pipelines (e.g. one pixel off
    # after a bilinear resize).  Stricter than that risks rejecting canvases
    # that are visually equivalent while still catching genuine mismatches.
    ratio_tolerance = max(src_ratio, dst_ratio) * 0.01
    if abs(src_ratio - dst_ratio) > ratio_tolerance:
        raise ValueError(
            "IMAGE and LATENT aspect ratios do not match. "
            "Please resize/crop the IMAGE upstream to match the LATENT canvas exactly."
        )

    fitted = image
    if w != target_width or h != target_height:
        fitted = resize_bhwc(image, target_width, target_height)

    meta: dict[str, int | str] = {
        "original_width": target_width,
        "original_height": target_height,
        "alignment_mode": normalized_mode,
        "pad_left": 0,
        "pad_top": 0,
        "pad_right": 0,
        "pad_bottom": 0,
    }
    return fitted, meta


@dataclass(frozen=True)
class PackedSpatialTensorAdapter:
    entries: dict[str, torch.Tensor]

    def build_payload(self, latent, timestep, cond_dict) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, tensor in self.entries.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"spatial adapter entry '{name}' is not a tensor")
            validate_spatial_tensor(
                tensor,
                name=f"spatial adapter entry '{name}'",
            )
            out[name] = tensor.to(device=latent.device)
        return out


@dataclass(frozen=True)
class StaticObjectAdapter:
    entries: dict[str, Any]

    def build_payload(self, latent, timestep, cond_dict) -> dict[str, Any]:
        return dict(self.entries)


def build_packed_spatial_tensor_adapter(
    *,
    image_a: torch.Tensor | None = None,
    image_b: torch.Tensor | None = None,
    latent_a: dict | None = None,
    latent_b: dict | None = None,
    name_image_a: str = "spatial_image_a",
    name_image_b: str = "spatial_image_b",
    name_latent_a: str = "spatial_latent_a",
    name_latent_b: str = "spatial_latent_b",
) -> PackedSpatialTensorAdapter:
    entries: dict[str, torch.Tensor] = {}

    if image_a is not None:
        entries[name_image_a] = image_bhwc_to_bchw(image_a)

    if image_b is not None:
        entries[name_image_b] = image_bhwc_to_bchw(image_b)

    if latent_a is not None:
        entries[name_latent_a] = latent_dict_to_tensor(latent_a)

    if latent_b is not None:
        entries[name_latent_b] = latent_dict_to_tensor(latent_b)

    if not entries:
        raise ValueError("at least one IMAGE or LATENT input must be provided")

    return PackedSpatialTensorAdapter(entries=entries)


def build_static_object_adapter(**entries: Any) -> StaticObjectAdapter:
    clean = {k: v for k, v in entries.items() if v is not None}
    if not clean:
        raise ValueError("static object adapter needs at least one entry")
    return StaticObjectAdapter(entries=clean)


# ============================================================
# runtime protocols
# ============================================================

@dataclass(frozen=True)
class RegionalConditioningBatch:
    positive_crossattn: torch.Tensor
    negative_crossattn: torch.Tensor
    region_indices: list[int]
    sequence_length: int
    # Per-region pooled outputs for SDXL y-conditioning (None when not available)
    positive_pooled: list[torch.Tensor | None]
    negative_pooled: list[torch.Tensor | None]


@dataclass(frozen=True)
class RuntimeRegionBatch:
    region_indices: list[int]
    context_bboxes: list[RegionBBox]
    core_bboxes: list[RegionBBox]
    write_bboxes: list[RegionBBox]


@dataclass
class RuntimeCanvasState:
    latent_height: int
    latent_width: int
    output_accumulator: torch.Tensor
    weight_accumulator: torch.Tensor



@dataclass(frozen=True)
class AdaptiveRuntimeInputs:
    region_plan: RegionPlan
    positive_entries: list[dict[str, Any]]
    negative_entries: list[dict[str, Any]]
    runtime_payload_adapter: RuntimePayloadAdapter | None = None
    debug_runtime: bool = False


# ============================================================
# adaptive diffusion
# ============================================================


def build_region_batches(region_plan: RegionPlan, max_batch_size: int) -> list[RuntimeRegionBatch]:
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    batches: list[RuntimeRegionBatch] = []
    total = len(region_plan.bboxes_latent)
    current_indices: list[int] = []
    current_size: tuple[int, int] | None = None

    def _flush() -> None:
        nonlocal current_indices, current_size
        if not current_indices:
            return
        indices = list(current_indices)
        batches.append(
            RuntimeRegionBatch(
                region_indices=indices,
                context_bboxes=[region_plan.bboxes_latent[i] for i in indices],
                core_bboxes=[region_plan.core_bboxes_latent[i] for i in indices],
                write_bboxes=[region_plan.write_bboxes_latent[i] for i in indices],
            )
        )
        current_indices = []
        current_size = None

    for idx in range(total):
        bbox = region_plan.bboxes_latent[idx]
        size = (int(bbox.h), int(bbox.w))
        if current_size is None:
            current_size = size
        if size != current_size or len(current_indices) >= max_batch_size:
            _flush()
            current_size = size
        current_indices.append(idx)

    _flush()
    return batches


def build_region_batches_length_aware(
    region_plan: RegionPlan,
    positive_entries: list[dict],
    max_batch_size: int,
) -> list[RuntimeRegionBatch]:
    """Like build_region_batches but sorts regions by prompt length within each
    same-bbox-size group before dividing into batches.

    When users provide prompts of wildly different lengths (e.g. 30 tokens vs
    300 tokens), the default row-major ordering may place short and long prompts
    in the same batch, forcing all slots to pad to max length.  Grouping by
    length first means each batch sees prompts of similar length, which
    minimises the zero-padding overhead and reduces VRAM usage at high region
    counts (6K+ upscaling).

    The bbox-size constraint is always respected: regions with different context
    crop sizes cannot share a batch.
    """
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")

    # Collect (bbox_size, prompt_length, region_index) triples.
    triplets: list[tuple[tuple[int, int], int, int]] = []
    for idx in range(len(region_plan.bboxes_latent)):
        bbox = region_plan.bboxes_latent[idx]
        size = (int(bbox.h), int(bbox.w))
        if idx < len(positive_entries):
            cross = positive_entries[idx].get("c_crossattn", None)
            plen = int(cross.shape[1]) if cross is not None else 0
        else:
            plen = 0
        triplets.append((size, plen, idx))

    # Primary sort key: bbox size (ensures same-size groups stay together).
    # Secondary sort key: prompt length descending (longest first so that
    # padding decreases as we fill batches within a group).
    triplets.sort(key=lambda t: (t[0], -t[1]))

    batches: list[RuntimeRegionBatch] = []
    current_indices: list[int] = []
    current_size: tuple[int, int] | None = None

    def _flush_la() -> None:
        nonlocal current_indices, current_size
        if not current_indices:
            return
        indices = list(current_indices)
        batches.append(
            RuntimeRegionBatch(
                region_indices=indices,
                context_bboxes=[region_plan.bboxes_latent[i] for i in indices],
                core_bboxes=[region_plan.core_bboxes_latent[i] for i in indices],
                write_bboxes=[region_plan.write_bboxes_latent[i] for i in indices],
            )
        )
        current_indices = []
        current_size = None

    for size, _plen, idx in triplets:
        if current_size is None:
            current_size = size
        if size != current_size or len(current_indices) >= max_batch_size:
            _flush_la()
            current_size = size
        current_indices.append(idx)

    _flush_la()
    return batches


def slice_latent_regions(
    latent: torch.Tensor,
    bboxes: Sequence[RegionBBox],
) -> torch.Tensor:
    validate_spatial_tensor(
        latent,
        name="latent",
    )

    regions = [
        latent[spatial_slice_for_bbox(bbox)]
        for bbox in bboxes
    ]

    if not regions:
        raise ValueError("bboxes must be non-empty")

    return torch.cat(regions, dim=0)


def initialize_canvas_state(
    latent: torch.Tensor,
) -> RuntimeCanvasState:
    validate_spatial_tensor(
        latent,
        name="latent",
    )

    h = int(latent.shape[-2])
    w = int(latent.shape[-1])

    # One singleton dimension for every non-spatial latent axis.
    # BCHW  -> [1, 1, H, W]
    # BCTHW -> [1, 1, 1, H, W]
    weight_shape = (
        (1,) * (latent.ndim - 2)
        + (h, w)
    )

    return RuntimeCanvasState(
        latent_height=h,
        latent_width=w,
        output_accumulator=torch.zeros_like(latent),
        weight_accumulator=torch.zeros(
            weight_shape,
            device=latent.device,
            dtype=torch.float32,
        ),
    )



def _build_axis_write_weight(length: int, inner_start: int, inner_end: int, outer_start: int, outer_end: int, *, device) -> torch.Tensor:
    w = torch.zeros(length, device=device, dtype=torch.float32)
    if inner_end > inner_start:
        w[inner_start:inner_end] = 1.0

    if inner_start > outer_start:
        left_len = inner_start - outer_start
        if left_len == 1:
            w[outer_start:inner_start] = 0.5
        else:
            w[outer_start:inner_start] = torch.linspace(
                0.0,
                1.0,
                left_len + 2,
                device=device,
                dtype=torch.float32,
            )[1:-1]

    if outer_end > inner_end:
        right_len = outer_end - inner_end
        if right_len == 1:
            w[inner_end:outer_end] = 0.5
        else:
            w[inner_end:outer_end] = torch.linspace(
                1.0,
                0.0,
                right_len + 2,
                device=device,
                dtype=torch.float32,
            )[1:-1]

    return w.clamp_(0.0, 1.0)


def build_bbox_blend_weight(
    region_plan: RegionPlan,
    context_bbox: RegionBBox,
    core_bbox: RegionBBox,
    write_bbox: RegionBBox,
    *,
    device,
    dtype=torch.float32,
) -> torch.Tensor:
    h = context_bbox.h
    w = context_bbox.w

    core_x1 = max(0, core_bbox.x - context_bbox.x)
    core_y1 = max(0, core_bbox.y - context_bbox.y)
    core_x2 = min(w, core_x1 + core_bbox.w)
    core_y2 = min(h, core_y1 + core_bbox.h)

    write_x1 = max(0, write_bbox.x - context_bbox.x)
    write_y1 = max(0, write_bbox.y - context_bbox.y)
    write_x2 = min(w, write_x1 + write_bbox.w)
    write_y2 = min(h, write_y1 + write_bbox.h)

    xw = _build_axis_write_weight(w, core_x1, core_x2, write_x1, write_x2, device=device)
    yw = _build_axis_write_weight(h, core_y1, core_y2, write_y1, write_y2, device=device)
    weight = torch.outer(yw, xw).unsqueeze(0).unsqueeze(0)
    return weight.to(dtype=dtype)


def build_region_weight_map(region_plan: RegionPlan, *, device) -> list[torch.Tensor]:
    return [
        build_bbox_blend_weight(
            region_plan,
            context_bbox,
            core_bbox,
            write_bbox,
            device=device,
            dtype=torch.float32,
        )
        for context_bbox, core_bbox, write_bbox in zip(
            region_plan.bboxes_latent,
            region_plan.core_bboxes_latent,
            region_plan.write_bboxes_latent,
        )
    ]


def build_regional_conditioning_batch(
    runtime_inputs: AdaptiveRuntimeInputs,
    region_indices: Sequence[int],
    *,
    device=None,
    dtype=None,
) -> RegionalConditioningBatch:
    pos_cross = [runtime_inputs.positive_entries[i]["c_crossattn"] for i in region_indices]
    neg_cross = [runtime_inputs.negative_entries[i]["c_crossattn"] for i in region_indices]

    # Use max-length + zero-padding instead of LCM + semantic repeat.
    # LCM can explode (e.g. lcm(154,231)=462) with mixed prompt lengths,
    # causing severe OOM and garbled outputs.  Padding with zeros does not
    # alter the meaning of existing tokens and stays within normal bounds.
    all_lengths = [int(t.shape[1]) for t in pos_cross + neg_cross]
    target_len = max(all_lengths) if all_lengths else 77

    pos_batch = torch.cat(
        [pad_crossattn_to_length(t, target_len) for t in pos_cross],
        dim=0,
    )
    neg_batch = torch.cat(
        [pad_crossattn_to_length(t, target_len) for t in neg_cross],
        dim=0,
    )

    if device is not None or dtype is not None:
        kw: dict[str, Any] = {}
        if device is not None:
            kw["device"] = device
        if dtype is not None:
            kw["dtype"] = dtype
        pos_batch = pos_batch.to(**kw)
        neg_batch = neg_batch.to(**kw)

    # Collect per-region pooled outputs for SDXL y-conditioning.
    def _get_pooled(entries: list[dict[str, Any]], idx: int) -> torch.Tensor | None:
        extra = entries[idx].get("extra", {})
        p = extra.get("pooled_output", None)
        if p is None or not isinstance(p, torch.Tensor):
            return None
        p = p.detach()
        if device is not None:
            p = p.to(device=device)
        if dtype is not None:
            p = p.to(dtype=dtype)
        return p

    pos_pooled = [_get_pooled(runtime_inputs.positive_entries, i) for i in region_indices]
    neg_pooled = [_get_pooled(runtime_inputs.negative_entries, i) for i in region_indices]

    return RegionalConditioningBatch(
        positive_crossattn=pos_batch,
        negative_crossattn=neg_batch,
        region_indices=list(region_indices),
        sequence_length=target_len,
        positive_pooled=pos_pooled,
        negative_pooled=neg_pooled,
    )


def build_model_crossattn_batch(
    regional_batch: RegionalConditioningBatch,
    cond_or_uncond: Sequence[int],
    *,
    base_batch: int,
    device,
    dtype,
) -> torch.Tensor:
    """Build the cross-attention batch for the UNet call.

    For each region, produces one tensor per cond_or_uncond flag (0=positive,
    non-zero=negative), expanded to base_batch along dim 0.  The per-region y
    (pooled_output) is handled separately by build_regional_y_batch.
    """
    assembled: list[torch.Tensor] = []
    region_count = len(regional_batch.region_indices)

    for region_offset in range(region_count):
        pos_cross = regional_batch.positive_crossattn[region_offset:region_offset + 1]
        neg_cross = regional_batch.negative_crossattn[region_offset:region_offset + 1]

        for cond_flag in cond_or_uncond:
            chosen = pos_cross if int(cond_flag) == 0 else neg_cross
            chosen = chosen.to(device=device, dtype=dtype)
            chosen = repeat_to_batch_size(chosen, base_batch, dim=0)
            assembled.append(chosen)

    return torch.cat(assembled, dim=0)


def build_regional_y_batch(
    regional_batch: RegionalConditioningBatch,
    global_y: torch.Tensor,
    cond_or_uncond: Sequence[int],
    *,
    base_batch: int,
    device,
    dtype,
) -> torch.Tensor | None:
    """Build a per-region y tensor for SDXL by injecting per-region pooled_output.

    In SDXL, global_y has shape [B, 2816] = [pooled_1280 || size_embeds_1536].
    We replace the first 1280 dimensions with each region's own pooled_output
    while keeping the shared size/aesthetic embeddings intact.

    Returns None if no pooled_output is available for any region (e.g. SD1.5).
    Falls back to global_y for regions missing pooled_output.
    """
    has_any_pooled = any(
        p is not None
        for p in regional_batch.positive_pooled + regional_batch.negative_pooled
    )
    if not has_any_pooled:
        return None

    # SDXL pooled dimension is always 1280 (CLIP-G hidden size).
    # If global_y is smaller, we can't safely inject – skip.
    pooled_dim = 1280
    if global_y.shape[-1] < pooled_dim:
        return None

    # Global y for a single (cond or uncond) item: shape [base_batch, y_dim]
    # We need one global reference row to borrow the size-embedding tail from.
    # Use the first row (all rows share the same aesthetic params).
    y_ref = global_y[:1].to(device=device, dtype=dtype)  # [1, y_dim]

    assembled = []
    region_count = len(regional_batch.region_indices)

    for region_offset in range(region_count):
        pos_p = regional_batch.positive_pooled[region_offset]
        neg_p = regional_batch.negative_pooled[region_offset]

        for cond_flag in cond_or_uncond:
            chosen_p = pos_p if int(cond_flag) == 0 else neg_p
            y_row = y_ref.clone()

            if chosen_p is not None:
                p = chosen_p.to(device=device, dtype=dtype)
                if p.dim() == 1:
                    p = p.unsqueeze(0)  # [pooled_dim] → [1, pooled_dim]

                # Strict shape validation before injection.
                # Only replace when the pooled tensor has exactly the expected
                # dimension.  Any other shape means an architecture whose y
                # layout we don't know; fall back to global y_ref for this slot.
                if p.dim() == 2 and p.shape[0] >= 1 and p.shape[-1] == pooled_dim:
                    y_row[:, :pooled_dim] = p[:1, :pooled_dim]

            y_row = repeat_to_batch_size(y_row, base_batch, dim=0)
            assembled.append(y_row)

    return torch.cat(assembled, dim=0)


def accumulate_region_outputs(
    canvas_state: RuntimeCanvasState,
    region_outputs: torch.Tensor,
    region_batch: RuntimeRegionBatch,
    *,
    region_weights: list[torch.Tensor],
) -> None:
    validate_spatial_tensor(
        region_outputs,
        name="region_outputs",
    )

    region_count = len(region_batch.context_bboxes)

    if region_count <= 0:
        raise ValueError(
            "region_batch.context_bboxes must be non-empty"
        )

    if len(region_weights) != region_count:
        raise ValueError(
            "region weight count does not match region count: "
            f"{len(region_weights)} != {region_count}"
        )

    if region_outputs.shape[0] % region_count != 0:
        raise ValueError(
            "region output batch cannot be divided evenly among "
            f"{region_count} regions; output shape is "
            f"{tuple(region_outputs.shape)}"
        )

    per_region_batch = (
        region_outputs.shape[0] // region_count
    )

    for i, bbox in enumerate(
        region_batch.context_bboxes
    ):
        start = i * per_region_batch
        end = (i + 1) * per_region_batch
        current_output = region_outputs[start:end]

        weight = region_weights[i].to(
            device=current_output.device,
            dtype=current_output.dtype,
        )
        weight = expand_spatial_weight_for_tensor(
            weight,
            current_output,
        )

        weighted = current_output * weight
        spatial_slice = spatial_slice_for_bbox(bbox)

        canvas_target = (
            canvas_state.output_accumulator[
                spatial_slice
            ]
        )

        if canvas_target.shape != weighted.shape:
            raise ValueError(
                "regional output shape does not match canvas slice: "
                f"output={tuple(weighted.shape)}, "
                f"canvas={tuple(canvas_target.shape)}, "
                f"bbox={bbox.box}"
            )

        canvas_state.output_accumulator[
            spatial_slice
        ] += weighted

        canvas_weight = weight.to(
            device=canvas_state.weight_accumulator.device,
            dtype=canvas_state.weight_accumulator.dtype,
        )

        canvas_state.weight_accumulator[
            spatial_slice
        ] += canvas_weight


def finalize_canvas_output(canvas_state: RuntimeCanvasState, eps: float = 1e-8) -> torch.Tensor:
    weights = canvas_state.weight_accumulator.clamp(min=eps).to(
        device=canvas_state.output_accumulator.device,
        dtype=canvas_state.output_accumulator.dtype,
    )
    return canvas_state.output_accumulator / weights




def _slice_control_value_for_batch(
    value: Any,
    batch: RuntimeRegionBatch,
    latent: torch.Tensor,
    region_plan: RegionPlan,
) -> Any:
    if is_spatial_tensor_candidate(value):
        return slice_spatial_tensor_for_region_batch(
            value,
            batch,
            region_plan,
            latent_height=latent.shape[-2],
            latent_width=latent.shape[-1],
        )

    if isinstance(value, list):
        return [_slice_control_value_for_batch(item, batch, latent, region_plan) for item in value]

    if isinstance(value, tuple):
        return tuple(_slice_control_value_for_batch(item, batch, latent, region_plan) for item in value)

    if isinstance(value, dict):
        return {k: _slice_control_value_for_batch(v, batch, latent, region_plan) for k, v in value.items()}

    return value


def slice_control_dict_for_batch(
    control_dict: dict[str, Any],
    batch: RuntimeRegionBatch,
    latent: torch.Tensor,
    region_plan: RegionPlan,
) -> dict[str, Any]:
    return {k: _slice_control_value_for_batch(v, batch, latent, region_plan) for k, v in control_dict.items()}


def _debug_print_control_dict_shapes(control_dict: dict[str, Any], prefix: str = "[Egregora-ARMD]") -> None:
    try:
        keys = sorted(control_dict.keys())
    except Exception:
        keys = list(control_dict.keys())
    print(f"{prefix} control dict keys: {keys}")

    for key in keys:
        value = control_dict[key]
        if isinstance(value, list):
            print(f"{prefix} control[{key!r}] list len={len(value)}")
            for i, item in enumerate(value[:4]):
                if isinstance(item, torch.Tensor):
                    print(f"{prefix} control[{key!r}][{i}] shape={tuple(item.shape)} dtype={item.dtype}")
                else:
                    print(f"{prefix} control[{key!r}][{i}] type={type(item)}")
            if len(value) > 4:
                print(f"{prefix} control[{key!r}] ... {len(value) - 4} more items")
        elif isinstance(value, torch.Tensor):
            print(f"{prefix} control[{key!r}] shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"{prefix} control[{key!r}] type={type(value)}")


def clone_control_chain_for_batch(
    control: Any,
    batch: RuntimeRegionBatch,
    latent: torch.Tensor,
    region_plan: RegionPlan,
) -> Any:
    # Advanced-ControlNet in this workflow passes a dict of precomputed residuals
    # under c["control"]. Slice those residuals per region batch and hand the
    # regionalized dict directly to the UNet wrapper.
    if isinstance(control, dict):
        return slice_control_dict_for_batch(control, batch, latent, region_plan)
    return control


class EgregoraAdaptiveRegionalMixer:
    def __init__(self, runtime_inputs: AdaptiveRuntimeInputs, region_batch_size: int = 4):
        self.runtime_inputs = runtime_inputs
        self.region_batch_size = region_batch_size
        self._cached_shape: tuple[int, int] | None = None
        self._tile_weights: list[torch.Tensor] | None = None
        self._runtime_tile_weights: list[torch.Tensor] | None = None
        self._runtime_tile_weights_key: tuple[str, int | None, str, int, int] | None = None
        self._debug_logged = False

    def build_batches(self) -> list[RuntimeRegionBatch]:
        # Use length-aware batching when conditioning entries are available.
        # This sorts regions by prompt length within each same-bbox-size group
        # so that batches contain prompts of similar length, minimising
        # zero-padding overhead — especially important at high region counts
        # (6K+ upscaling) where users supply prompts of mixed lengths.
        entries = self.runtime_inputs.positive_entries
        if entries:
            return build_region_batches_length_aware(
                self.runtime_inputs.region_plan,
                entries,
                self.region_batch_size,
            )
        return build_region_batches(self.runtime_inputs.region_plan, self.region_batch_size)

    def _ensure_weight_cache(self, latent: torch.Tensor) -> None:
        h, w = latent.shape[-2], latent.shape[-1]
        if self._cached_shape == (h, w) and self._tile_weights is not None:
            return

        plan = self.runtime_inputs.region_plan
        if h != plan.latent_height or w != plan.latent_width:
            raise ValueError(
                f"latent shape mismatch: got {w}x{h}, expected "
                f"{plan.latent_width}x{plan.latent_height}"
            )

        self._tile_weights = build_region_weight_map(plan, device=latent.device)
        self._cached_shape = (h, w)
        self._runtime_tile_weights = None
        self._runtime_tile_weights_key = None

    def _ensure_runtime_tile_weights(self, latent: torch.Tensor) -> list[torch.Tensor]:
        device = latent.device
        device_index = getattr(device, "index", None)
        key = (device.type, device_index, str(latent.dtype), latent.shape[-2], latent.shape[-1])
        if self._runtime_tile_weights is None or self._runtime_tile_weights_key != key:
            self._runtime_tile_weights = [
                w.to(device=latent.device, dtype=latent.dtype) for w in self._tile_weights
            ]
            self._runtime_tile_weights_key = key
        return self._runtime_tile_weights

    def _slice_condition_value(self, value: Any, batch: RuntimeRegionBatch, latent: torch.Tensor) -> Any:
        if is_spatial_tensor_candidate(value):
            return slice_spatial_tensor_for_region_batch(
                value,
                batch,
                self.runtime_inputs.region_plan,
                latent_height=latent.shape[-2],
                latent_width=latent.shape[-1],
            )
        return value

    def _build_adapter_payload(self, latent: torch.Tensor, timestep: torch.Tensor, cond_dict: dict) -> dict[str, Any]:
        adapter = self.runtime_inputs.runtime_payload_adapter
        if adapter is None:
            return {}
        if not hasattr(adapter, "build_payload"):
            raise ValueError("runtime_payload_adapter must expose build_payload(latent, timestep, cond_dict)")
        built = adapter.build_payload(latent, timestep, cond_dict)
        if not isinstance(built, dict):
            raise ValueError("runtime_payload_adapter.build_payload must return a dict")
        return built

    def _maybe_debug_log(self, latent: torch.Tensor, timestep: torch.Tensor, cond_dict: dict, cond_or_uncond: list[int], full_conditions: dict[str, Any]) -> None:
        if not self.runtime_inputs.debug_runtime or self._debug_logged:
            return
        layout = (
            "BCHW"
            if latent.ndim == 4
            else "BCTHW-compatible"
            if latent.ndim == 5
            else f"{latent.ndim}D-spatial"
        )
        print(
            "[Egregora-ARMD] latent:",
            tuple(latent.shape),
            latent.dtype,
            latent.device,
            f"layout={layout}",
        )
        print("[Egregora-ARMD] timestep:", tuple(timestep.shape) if hasattr(timestep, "shape") else type(timestep))
        print("[Egregora-ARMD] cond_or_uncond:", cond_or_uncond)
        print("[Egregora-ARMD] input c keys:", sorted(cond_dict.keys()))
        print("[Egregora-ARMD] merged condition keys:", sorted(full_conditions.keys()))
        if "control" in full_conditions:
            print("[Egregora-ARMD] control present in conditioning")
        # Note: region processing order may differ from spatial row-major order.
        # build_region_batches_length_aware groups same-bbox-size regions by
        # prompt length (longest first) to minimise cross-attention padding.
        # This does not affect accumulation correctness — each region writes
        # to its correct canvas position regardless of processing order.
        batches = self.build_batches()
        print(f"[Egregora-ARMD] region_count={self.runtime_inputs.region_plan.region_count} "
              f"batch_count={len(batches)} "
              f"region_indices_per_batch={[b.region_indices for b in batches]}")
        self._debug_logged = True

    def __call__(self, model_function, args):
        latent: torch.Tensor = args["input"]
        validate_spatial_tensor(
            latent,
            name="incoming model latent",
        )

        timestep: torch.Tensor = args["timestep"]
        cond_dict: dict = args["c"]
        cond_or_uncond = list(args.get("cond_or_uncond", [0, 1]))

        self._ensure_weight_cache(latent)
        runtime_tile_weights = self._ensure_runtime_tile_weights(latent)
        canvas = initialize_canvas_state(latent)

        adapter_payload = self._build_adapter_payload(latent, timestep, cond_dict)
        full_conditions = dict(cond_dict)
        for key, value in adapter_payload.items():
            if key in full_conditions:
                raise ValueError(f"runtime payload key collision: {key}")
            full_conditions[key] = value

        self._maybe_debug_log(latent, timestep, cond_dict, cond_or_uncond, full_conditions)

        group_count = max(1, len(cond_or_uncond))
        if latent.shape[0] % group_count != 0:
            raise ValueError("incoming latent batch is incompatible with cond_or_uncond groups")
        base_batch = latent.shape[0] // group_count

        # Detect global y tensor for SDXL per-region pooled injection
        global_y: torch.Tensor | None = None
        if "y" in full_conditions and isinstance(full_conditions["y"], torch.Tensor):
            global_y = full_conditions["y"]

        for batch in self.build_batches():
            x_regions = slice_latent_regions(latent, batch.context_bboxes)
            t_regions = repeat_to_batch_size(timestep, x_regions.shape[0], dim=0)

            c_regions: dict[str, Any] = {}
            control_obj = None

            for key, value in full_conditions.items():
                if key in ("c_crossattn", "y"):
                    continue
                if key == "control":
                    control_obj = clone_control_chain_for_batch(
                        value, batch, latent, self.runtime_inputs.region_plan
                    )
                    if self.runtime_inputs.debug_runtime and \
                            not getattr(self, "_control_debug_logged", False):
                        print("[Egregora-ARMD] control object type:", type(value))
                        if isinstance(value, dict):
                            print("[Egregora-ARMD] original control dict:")
                            _debug_print_control_dict_shapes(value)
                        if isinstance(control_obj, dict):
                            print("[Egregora-ARMD] sliced control dict:")
                            _debug_print_control_dict_shapes(control_obj)
                        self._control_debug_logged = True
                    continue
                value = self._slice_condition_value(value, batch, latent)
                if isinstance(value, torch.Tensor) and value.shape[0] != x_regions.shape[0]:
                    value = repeat_to_batch_size(value, x_regions.shape[0], dim=0)
                c_regions[key] = value

            regional_batch = build_regional_conditioning_batch(
                self.runtime_inputs,
                batch.region_indices,
                device=x_regions.device,
                dtype=x_regions.dtype,
            )
            c_regions["c_crossattn"] = build_model_crossattn_batch(
                regional_batch,
                cond_or_uncond,
                base_batch=base_batch,
                device=x_regions.device,
                dtype=x_regions.dtype,
            )

            # Per-region y injection for SDXL pooled_output.
            # Each region gets its own pooled embedding instead of the global
            # placeholder pooled.  Falls back to broadcast global_y when
            # pooled_output is unavailable or architecture doesn't use y.
            if global_y is not None:
                regional_y = build_regional_y_batch(
                    regional_batch,
                    global_y,
                    cond_or_uncond,
                    base_batch=base_batch,
                    device=x_regions.device,
                    dtype=x_regions.dtype,
                )
                if regional_y is not None:
                    c_regions["y"] = regional_y
                else:
                    gy = global_y.to(device=x_regions.device, dtype=x_regions.dtype)
                    if gy.shape[0] != x_regions.shape[0]:
                        gy = repeat_to_batch_size(gy, x_regions.shape[0], dim=0)
                    c_regions["y"] = gy

            if control_obj is not None:
                if self.runtime_inputs.debug_runtime and \
                        not getattr(self, "_control_call_debug_logged", False):
                    print("[Egregora-ARMD] x_regions:", tuple(x_regions.shape))
                    print("[Egregora-ARMD] context bboxes:",
                          [bbox.box for bbox in batch.context_bboxes])
                    print("[Egregora-ARMD] core bboxes:",
                          [bbox.box for bbox in batch.core_bboxes])
                    print("[Egregora-ARMD] write bboxes:",
                          [bbox.box for bbox in batch.write_bboxes])
                    self._control_call_debug_logged = True
                c_regions["control"] = control_obj

            region_outputs = model_function(x_regions, t_regions, **c_regions)
            batch_weights = [runtime_tile_weights[i] for i in batch.region_indices]
            accumulate_region_outputs(
                canvas,
                region_outputs,
                batch,
                region_weights=batch_weights,
            )

        return finalize_canvas_output(canvas)


# ============================================================
# comfy nodes
# ============================================================

def build_empty_placeholder(clip) -> Any:
    from nodes import CLIPTextEncode
    return CLIPTextEncode().encode(clip, "")[0]


class EgregoraRegionPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "region_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "region_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "region_overlap": ("INT", {"default": 384, "min": 0, "max": 2048, "step": 8}),
                "blend_feather": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
                "compression": ("INT", {"default": 8, "min": 1, "max": 16, "step": 1}),
                "alignment_mode": (
                    [
                        "pad_reflect",
                        "floor_crop",
                    ],
                    {"default": "pad_reflect"},
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "EGREGORA_REGION_PLAN", "INT", "STRING")
    RETURN_NAMES = ("aligned_image", "regions_batch", "regions_list", "region_plan", "region_count", "region_order_text")
    OUTPUT_IS_LIST = (False, False, True, False, False, False)
    FUNCTION = "plan"
    CATEGORY = "Egregora-ARMD"

    def plan(self, region_width, region_height, region_overlap, blend_feather, compression, alignment_mode, image=None, latent=None):
        if image is None and latent is None:
            raise ValueError("Egregora Region Plan requires either an IMAGE or a LATENT input")

        aligned: torch.Tensor
        alignment_meta: dict[str, int | str]
        original_w: int
        original_h: int

        if latent is not None:
            latent_samples = latent_dict_to_tensor(latent)
            target_h = int(latent_samples.shape[-2]) * int(compression)
            target_w = int(latent_samples.shape[-1]) * int(compression)
            original_w = target_w
            original_h = target_h
            if image is None:
                aligned = blank_image_bhwc(target_w, target_h)
                alignment_meta = {
                    "original_width": target_w,
                    "original_height": target_h,
                    "alignment_mode": "pad_reflect" if alignment_mode != "floor_crop" else "floor_crop",
                    "pad_left": 0,
                    "pad_top": 0,
                    "pad_right": 0,
                    "pad_bottom": 0,
                }
            else:
                aligned, alignment_meta = fit_image_to_exact_canvas(image, target_w, target_h, alignment_mode)
        else:
            aligned, alignment_meta = align_image_to_compression(image, compression, alignment_mode)
            _, original_h, original_w, _ = image.shape

        _, h, w, _ = aligned.shape
        plan = build_region_plan(
            aligned_width=w,
            aligned_height=h,
            region_width=region_width,
            region_height=region_height,
            region_overlap=region_overlap,
            compression=compression,
            blend_feather=blend_feather,
            original_width=original_w,
            original_height=original_h,
            alignment_mode=str(alignment_meta["alignment_mode"]),
            pad_left=int(alignment_meta["pad_left"]),
            pad_top=int(alignment_meta["pad_top"]),
            pad_right=int(alignment_meta["pad_right"]),
            pad_bottom=int(alignment_meta["pad_bottom"]),
        )
        context_regions = extract_region_images(aligned, plan, use_core=False)
        core_regions = extract_region_images(aligned, plan, use_core=True)
        regions_batch = stack_region_images_with_padding(context_regions, pad_mode="edge") if context_regions else aligned[:0]
        return (
            aligned,
            regions_batch,
            core_regions,
            plan,
            plan.region_count,
            region_order_text(plan),
        )


class EgregoraRegionSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "region_plan": ("EGREGORA_REGION_PLAN",),
                "regions_list": ("IMAGE",),
                "region_index": ("INT", {"default": 1, "min": 1, "max": 8192, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("region",)
    FUNCTION = "select"
    CATEGORY = "Egregora-ARMD"

    def select(self, region_plan, regions_list, region_index):
        idx = max(1, min(region_index, len(regions_list))) - 1
        return (regions_list[idx],)


class EgregoraRegionalConditioning:
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "region_plan": ("EGREGORA_REGION_PLAN",),
                "positive_prompts": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "negative_prompts": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = (
        "EGREGORA_REGIONAL_CONDITIONING",
        "EGREGORA_REGIONAL_CONDITIONING",
        "CONDITIONING",
        "CONDITIONING",
        "INT",
    )
    RETURN_NAMES = (
        "regional_positive",
        "regional_negative",
        "placeholder_positive",
        "placeholder_negative",
        "prompt_count",
    )
    FUNCTION = "encode"
    CATEGORY = "Egregora-ARMD"

    def encode(self, clip, region_plan, positive_prompts, negative_prompts=None):
        from nodes import CLIPTextEncode

        clip = clip[0]
        region_plan = region_plan[0]

        positives = normalize_prompt_input(positive_prompts)
        negatives = normalize_prompt_input(negative_prompts) if negative_prompts is not None else []
        expected = region_plan.region_count

        if len(positives) != expected:
            raise ValueError(f"expected {expected} positive prompts, got {len(positives)}")

        if len(negatives) == 0:
            negatives = [""] * expected
        elif len(negatives) == 1:
            negatives = negatives * expected
        elif len(negatives) != expected:
            raise ValueError(f"expected 1 or {expected} negative prompts, got {len(negatives)}")

        reg_pos = []
        reg_neg = []

        for p in positives:
            cond = CLIPTextEncode().encode(clip, p)[0]
            reg_pos.append(conditioning_to_local_entry(cond))

        for n in negatives:
            cond = CLIPTextEncode().encode(clip, n)[0]
            reg_neg.append(conditioning_to_local_entry(cond))

        placeholder_positive = build_empty_placeholder(clip)
        placeholder_negative = build_empty_placeholder(clip)

        return (
            {"entries": reg_pos, "region_count": expected},
            {"entries": reg_neg, "region_count": expected},
            placeholder_positive,
            placeholder_negative,
            expected,
        )


class EgregoraSpatialTensorPack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name_image_a": ("STRING", {"default": "spatial_image_a"}),
                "name_image_b": ("STRING", {"default": "spatial_image_b"}),
                "name_latent_a": ("STRING", {"default": "spatial_latent_a"}),
                "name_latent_b": ("STRING", {"default": "spatial_latent_b"}),
            },
            "optional": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "latent_a": ("LATENT",),
                "latent_b": ("LATENT",),
            },
        }

    RETURN_TYPES = ("EGREGORA_RUNTIME_PAYLOAD_ADAPTER",)
    RETURN_NAMES = ("runtime_payload_adapter",)
    FUNCTION = "pack"
    CATEGORY = "Egregora-ARMD"

    def pack(
        self,
        name_image_a,
        name_image_b,
        name_latent_a,
        name_latent_b,
        image_a=None,
        image_b=None,
        latent_a=None,
        latent_b=None,
    ):
        adapter = build_packed_spatial_tensor_adapter(
            image_a=image_a,
            image_b=image_b,
            latent_a=latent_a,
            latent_b=latent_b,
            name_image_a=name_image_a,
            name_image_b=name_image_b,
            name_latent_a=name_latent_a,
            name_latent_b=name_latent_b,
        )
        return (adapter,)


class EgregoraStaticPayloadPack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name_value_a": ("STRING", {"default": "runtime_flag_a"}),
                "name_value_b": ("STRING", {"default": "runtime_flag_b"}),
                "value_a": ("STRING", {"default": ""}),
                "value_b": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("EGREGORA_RUNTIME_PAYLOAD_ADAPTER",)
    RETURN_NAMES = ("runtime_payload_adapter",)
    FUNCTION = "pack"
    CATEGORY = "Egregora-ARMD"

    def pack(self, name_value_a, name_value_b, value_a, value_b):
        payload = {}
        if value_a != "":
            payload[name_value_a] = value_a
        if value_b != "":
            payload[name_value_b] = value_b
        adapter = build_static_object_adapter(**payload)
        return (adapter,)


class EgregoraRuntimeAdapterMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "adapter_a": ("EGREGORA_RUNTIME_PAYLOAD_ADAPTER",),
                "adapter_b": ("EGREGORA_RUNTIME_PAYLOAD_ADAPTER",),
                "adapter_c": ("EGREGORA_RUNTIME_PAYLOAD_ADAPTER",),
            }
        }

    RETURN_TYPES = ("EGREGORA_RUNTIME_PAYLOAD_ADAPTER",)
    RETURN_NAMES = ("runtime_payload_adapter",)
    FUNCTION = "merge"
    CATEGORY = "Egregora-ARMD"

    def merge(self, adapter_a=None, adapter_b=None, adapter_c=None):
        merged = merge_runtime_adapters(adapter_a, adapter_b, adapter_c)
        return (merged,)


class EgregoraAdaptiveDiffusionApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "region_plan": ("EGREGORA_REGION_PLAN",),
                "regional_positive": ("EGREGORA_REGIONAL_CONDITIONING",),
                "regional_negative": ("EGREGORA_REGIONAL_CONDITIONING",),
                "region_batch_size": ("INT", {"default": 4, "min": 1, "max": 128, "step": 1}),
                "debug_runtime": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "runtime_payload_adapter": ("EGREGORA_RUNTIME_PAYLOAD_ADAPTER",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "Egregora-ARMD"

    def apply(
        self,
        model: ModelPatcher,
        region_plan,
        regional_positive,
        regional_negative,
        region_batch_size,
        debug_runtime,
        runtime_payload_adapter=None,
    ):
        if regional_positive["region_count"] != region_plan.region_count:
            raise ValueError("regional_positive region count mismatch")
        if regional_negative["region_count"] != region_plan.region_count:
            raise ValueError("regional_negative region count mismatch")

        runtime_inputs = AdaptiveRuntimeInputs(
            region_plan=region_plan,
            positive_entries=regional_positive["entries"],
            negative_entries=regional_negative["entries"],
            runtime_payload_adapter=runtime_payload_adapter,
            debug_runtime=debug_runtime,
        )
        mixer = EgregoraAdaptiveRegionalMixer(runtime_inputs, region_batch_size=region_batch_size)

        patched = model.clone()
        patched.set_model_unet_function_wrapper(mixer)
        patched.model_options["egregora_armd"] = True
        return (patched,)


class EgregoraRestoreOriginalSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "region_plan": ("EGREGORA_REGION_PLAN",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_image",)
    FUNCTION = "restore"
    CATEGORY = "Egregora-ARMD"

    def restore(self, image, region_plan: RegionPlan):
        _, h, w, _ = image.shape
        aligned_h = int(region_plan.aligned_height)
        aligned_w = int(region_plan.aligned_width)

        if aligned_h <= 0 or aligned_w <= 0:
            return (image,)

        has_padding = any(
            int(v) > 0
            for v in (
                region_plan.pad_left,
                region_plan.pad_top,
                region_plan.pad_right,
                region_plan.pad_bottom,
            )
        )

        # If there was no padding during planning, there is nothing meaningful to crop back.
        # In that case the safest behavior is to preserve the current output resolution.
        if not has_padding:
            return (image,)

        scale_h = h / float(aligned_h)
        scale_w = w / float(aligned_w)

        target_h = max(1, int(round(region_plan.original_height * scale_h)))
        target_w = max(1, int(round(region_plan.original_width * scale_w)))

        y1 = int(round(region_plan.pad_top * scale_h))
        x1 = int(round(region_plan.pad_left * scale_w))
        y2 = y1 + target_h
        x2 = x1 + target_w

        # Clamp crop bounds defensively.
        y1 = max(0, min(y1, h))
        x1 = max(0, min(x1, w))
        y2 = max(y1, min(y2, h))
        x2 = max(x1, min(x2, w))

        cropped = image[:, y1:y2, x1:x2, :]

        # If rounding during upscale caused a 1-2 px mismatch, normalize to the expected
        # scaled target size while preserving the recovered framing.
        ch = int(cropped.shape[1])
        cw = int(cropped.shape[2])
        if ch != target_h or cw != target_w:
            cropped = resize_bhwc(cropped, target_w, target_h)

        return (cropped,)


NODE_CLASS_MAPPINGS = {
    "EgregoraRegionPlan": EgregoraRegionPlan,
    "EgregoraRegionalConditioning": EgregoraRegionalConditioning,
    "EgregoraAdaptiveDiffusionApply": EgregoraAdaptiveDiffusionApply,
    "EgregoraRegionSelect": EgregoraRegionSelect,
    "EgregoraSpatialTensorPack": EgregoraSpatialTensorPack,
    "EgregoraStaticPayloadPack": EgregoraStaticPayloadPack,
    "EgregoraRuntimeAdapterMerge": EgregoraRuntimeAdapterMerge,
    "EgregoraRestoreOriginalSize": EgregoraRestoreOriginalSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EgregoraRegionPlan": "🧭 Egregora Region Plan",
    "EgregoraRegionalConditioning": "🧠 Egregora Regional Conditioning",
    "EgregoraAdaptiveDiffusionApply": "🌐 Egregora Adaptive Diffusion Apply",
    "EgregoraRegionSelect": "🔎 Egregora Region Select",
    "EgregoraSpatialTensorPack": "🧩 Egregora Spatial Tensor Pack",
    "EgregoraStaticPayloadPack": "📦 Egregora Static Payload Pack",
    "EgregoraRuntimeAdapterMerge": "🔗 Egregora Runtime Adapter Merge",
    "EgregoraRestoreOriginalSize": "📐 Egregora Restore Original Size",
}
