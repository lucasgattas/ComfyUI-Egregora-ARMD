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

__version__ = "0.1.4"


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


def lcm_for_lengths(lengths: Sequence[int]) -> int:
    if not lengths:
        return 1
    acc = int(lengths[0])
    for length in lengths[1:]:
        acc = math.lcm(acc, int(length))
    return acc


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
    rows: int
    cols: int
    bboxes_latent: list[RegionBBox]
    bboxes_pixel: list[RegionBBox]

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

    latent_width = aligned_width // compression
    latent_height = aligned_height // compression

    rw_l_raw = region_width // compression
    rh_l_raw = region_height // compression
    ro_l_raw = region_overlap // compression

    rw_l = min(rw_l_raw, latent_width)
    rh_l = min(rh_l_raw, latent_height)
    ro_l = min(ro_l_raw, max(0, rw_l - 1), max(0, rh_l - 1))

    effective_region_width = rw_l * compression
    effective_region_height = rh_l * compression
    effective_region_overlap = ro_l * compression

    xs = axis_positions(latent_width, rw_l, ro_l)
    ys = axis_positions(latent_height, rh_l, ro_l)

    bboxes_latent: list[RegionBBox] = []
    bboxes_pixel: list[RegionBBox] = []

    for y in ys:
        for x in xs:
            lat = RegionBBox(x=x, y=y, w=rw_l, h=rh_l)
            pix = RegionBBox(
                x=x * compression,
                y=y * compression,
                w=effective_region_width,
                h=effective_region_height,
            )
            bboxes_latent.append(lat)
            bboxes_pixel.append(pix)

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
        region_width_latent=rw_l,
        region_height_latent=rh_l,
        region_overlap_latent=ro_l,
        rows=len(ys),
        cols=len(xs),
        bboxes_latent=bboxes_latent,
        bboxes_pixel=bboxes_pixel,
    )


def region_order_text(plan: RegionPlan) -> str:
    lines = []
    for idx, (pb, lb) in enumerate(zip(plan.bboxes_pixel, plan.bboxes_latent), start=1):
        row = (idx - 1) // plan.cols
        col = (idx - 1) % plan.cols
        lines.append(
            f"region {idx}: row={row}, col={col}, pixel_bbox={pb.box}, latent_bbox={lb.box}"
        )
    return "\n".join(lines)


def extract_region_images(image: torch.Tensor, plan: RegionPlan) -> list[torch.Tensor]:
    out = []
    for bbox in plan.bboxes_pixel:
        out.append(image[:, bbox.y:bbox.y2, bbox.x:bbox.x2, :])
    return out


# ============================================================
# regional conditioning
# ============================================================

def collapse_prompt_text(s: str) -> str:
    return " ".join(str(s).split()).strip()


def normalize_prompt_input(raw: Any) -> list[str]:
    if raw is None:
        return []

    if isinstance(raw, str):
        return [line.strip() for line in raw.splitlines() if line.strip()]

    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for item in raw:
            if item is None:
                continue
            t = collapse_prompt_text(str(item))
            if t:
                out.append(t)
        return out

    t = collapse_prompt_text(str(raw))
    return [t] if t else []


def repeat_crossattn_to_length(t: torch.Tensor, target_len: int) -> torch.Tensor:
    if t.shape[1] == target_len:
        return t
    if target_len % t.shape[1] != 0:
        raise ValueError(
            f"cannot expand c_crossattn length {t.shape[1]} to target {target_len} cleanly"
        )
    return t.repeat(1, target_len // t.shape[1], 1)


def conditioning_to_local_entry(conditioning: Any) -> dict[str, torch.Tensor]:
    if not isinstance(conditioning, list) or len(conditioning) == 0:
        raise ValueError("unexpected conditioning format")
    base = conditioning[0]
    if not isinstance(base, (list, tuple)) or len(base) < 2:
        raise ValueError("unexpected conditioning entry format")
    cross = base[0]
    if not isinstance(cross, torch.Tensor):
        raise ValueError("conditioning c_crossattn tensor not found")
    return {"c_crossattn": cross.detach().cpu()}


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


def build_scaled_bboxes_for_plan(
    region_plan: RegionPlan,
    *,
    tensor_height: int,
    tensor_width: int,
) -> list[RegionBBox]:
    tile_w = scale_region_extent_to_tensor(
        full_latent_extent=region_plan.latent_width,
        tensor_extent=tensor_width,
        region_extent=region_plan.region_width_latent,
    )
    tile_h = scale_region_extent_to_tensor(
        full_latent_extent=region_plan.latent_height,
        tensor_extent=tensor_height,
        region_extent=region_plan.region_height_latent,
    )

    xs = fixed_count_axis_positions(tensor_width, tile_w, region_plan.cols)
    ys = fixed_count_axis_positions(tensor_height, tile_h, region_plan.rows)

    bboxes: list[RegionBBox] = []
    for y in ys:
        for x in xs:
            bboxes.append(RegionBBox(x=x, y=y, w=tile_w, h=tile_h))
    return bboxes


def slice_spatial_tensor_for_region_batch(
    tensor: torch.Tensor,
    batch: "RuntimeRegionBatch",
    region_plan: RegionPlan,
    *,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError("spatial tensor must be 4D")

    if tensor.shape[-2] == latent_height and tensor.shape[-1] == latent_width:
        bboxes = batch.latent_bboxes
    else:
        scaled_bboxes = build_scaled_bboxes_for_plan(
            region_plan,
            tensor_height=tensor.shape[-2],
            tensor_width=tensor.shape[-1],
        )
        if len(scaled_bboxes) != region_plan.region_count:
            raise ValueError("scaled bbox count mismatch")
        bboxes = [scaled_bboxes[i] for i in batch.region_indices]

    slices = [tensor[:, :, bbox.y:bbox.y2, bbox.x:bbox.x2] for bbox in bboxes]
    if not slices:
        raise ValueError("bboxes must be non-empty")
    return torch.cat(slices, dim=0)


def latent_dict_to_tensor(latent: dict[str, Any]) -> torch.Tensor:
    samples = latent.get("samples", None)
    if not isinstance(samples, torch.Tensor):
        raise ValueError("LATENT input must contain tensor under key 'samples'")
    if samples.ndim != 4:
        raise ValueError("LATENT samples tensor must be 4D")
    return samples


@dataclass(frozen=True)
class PackedSpatialTensorAdapter:
    entries: dict[str, torch.Tensor]

    def build_payload(self, latent, timestep, cond_dict) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, tensor in self.entries.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"spatial adapter entry '{name}' is not a tensor")
            if tensor.ndim != 4:
                raise ValueError(f"spatial adapter entry '{name}' must be 4D")
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


@dataclass(frozen=True)
class RuntimeRegionBatch:
    region_indices: list[int]
    latent_bboxes: list[RegionBBox]


@dataclass
class RuntimeCanvasState:
    latent_height: int
    latent_width: int
    output_accumulator: torch.Tensor
    weight_accumulator: torch.Tensor


@dataclass(frozen=True)
class AdaptiveRuntimeInputs:
    region_plan: RegionPlan
    positive_entries: list[dict[str, torch.Tensor]]
    negative_entries: list[dict[str, torch.Tensor]]
    runtime_payload_adapter: RuntimePayloadAdapter | None = None
    debug_runtime: bool = False


# ============================================================
# adaptive diffusion
# ============================================================

def build_region_batches(region_plan: RegionPlan, max_batch_size: int) -> list[RuntimeRegionBatch]:
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    batches = []
    total = len(region_plan.bboxes_latent)
    for start in range(0, total, max_batch_size):
        end = min(start + max_batch_size, total)
        indices = list(range(start, end))
        bboxes = [region_plan.bboxes_latent[i] for i in indices]
        batches.append(RuntimeRegionBatch(region_indices=indices, latent_bboxes=bboxes))
    return batches


def slice_latent_regions(latent: torch.Tensor, bboxes: Sequence[RegionBBox]) -> torch.Tensor:
    regions = [latent[:, :, bbox.y:bbox.y2, bbox.x:bbox.x2] for bbox in bboxes]
    if not regions:
        raise ValueError("bboxes must be non-empty")
    return torch.cat(regions, dim=0)


def initialize_canvas_state(latent: torch.Tensor) -> RuntimeCanvasState:
    n, c, h, w = latent.shape
    return RuntimeCanvasState(
        latent_height=h,
        latent_width=w,
        output_accumulator=torch.zeros((n, c, h, w), device=latent.device, dtype=latent.dtype),
        weight_accumulator=torch.zeros((1, 1, h, w), device=latent.device, dtype=torch.float32),
    )


def build_bbox_blend_weight(
    region_plan: RegionPlan,
    bbox: RegionBBox,
    *,
    device,
    dtype=torch.float32,
    min_overlap_weight: float = 1e-3,
) -> torch.Tensor:
    h = bbox.h
    w = bbox.w
    ov = int(region_plan.region_overlap_latent)

    xw = torch.ones(w, device=device, dtype=torch.float32)
    yw = torch.ones(h, device=device, dtype=torch.float32)

    if ov > 0:
        left_ramp = torch.linspace(min_overlap_weight, 1.0, ov, device=device, dtype=torch.float32)
        right_ramp = torch.linspace(1.0, min_overlap_weight, ov, device=device, dtype=torch.float32)

        if bbox.x > 0:
            xw[:ov] = torch.minimum(xw[:ov], left_ramp)

        if bbox.x2 < region_plan.latent_width:
            xw[-ov:] = torch.minimum(xw[-ov:], right_ramp)

        if bbox.y > 0:
            yw[:ov] = torch.minimum(yw[:ov], left_ramp)

        if bbox.y2 < region_plan.latent_height:
            yw[-ov:] = torch.minimum(yw[-ov:], right_ramp)

    return torch.outer(yw, xw).unsqueeze(0).unsqueeze(0).to(dtype=dtype)


def build_region_weight_map(region_plan: RegionPlan, *, device) -> list[torch.Tensor]:
    return [
        build_bbox_blend_weight(region_plan, bbox, device=device, dtype=torch.float32)
        for bbox in region_plan.bboxes_latent
    ]


def build_regional_conditioning_batch(
    runtime_inputs: AdaptiveRuntimeInputs,
    region_indices: Sequence[int],
    *,
    device=None,
    dtype=None,
) -> RegionalConditioningBatch:
    pos = [runtime_inputs.positive_entries[i]["c_crossattn"] for i in region_indices]
    neg = [runtime_inputs.negative_entries[i]["c_crossattn"] for i in region_indices]

    all_lengths = [int(t.shape[1]) for t in pos] + [int(t.shape[1]) for t in neg]
    target_len = lcm_for_lengths(all_lengths)

    pos_batch = torch.cat(
        [repeat_crossattn_to_length(t, target_len) for t in pos],
        dim=0,
    )
    neg_batch = torch.cat(
        [repeat_crossattn_to_length(t, target_len) for t in neg],
        dim=0,
    )

    if device is not None or dtype is not None:
        pos_batch = pos_batch.to(
            device=device if device is not None else pos_batch.device,
            dtype=dtype if dtype is not None else pos_batch.dtype,
        )
        neg_batch = neg_batch.to(
            device=device if device is not None else neg_batch.device,
            dtype=dtype if dtype is not None else neg_batch.dtype,
        )

    return RegionalConditioningBatch(
        positive_crossattn=pos_batch,
        negative_crossattn=neg_batch,
        region_indices=list(region_indices),
        sequence_length=target_len,
    )


def build_model_crossattn_batch(
    regional_batch: RegionalConditioningBatch,
    cond_or_uncond: Sequence[int],
    *,
    base_batch: int,
    device,
    dtype,
) -> torch.Tensor:
    assembled = []
    region_count = len(regional_batch.region_indices)

    for region_offset in range(region_count):
        pos = regional_batch.positive_crossattn[region_offset:region_offset + 1]
        neg = regional_batch.negative_crossattn[region_offset:region_offset + 1]

        for cond_flag in cond_or_uncond:
            chosen = pos if int(cond_flag) == 0 else neg
            chosen = chosen.to(device=device, dtype=dtype)
            chosen = repeat_to_batch_size(chosen, base_batch, dim=0)
            assembled.append(chosen)

    return torch.cat(assembled, dim=0)


def accumulate_region_outputs(
    canvas_state: RuntimeCanvasState,
    region_outputs: torch.Tensor,
    region_batch: RuntimeRegionBatch,
    *,
    region_weights: list[torch.Tensor],
) -> None:
    per_region_batch = region_outputs.shape[0] // len(region_batch.latent_bboxes)

    for i, bbox in enumerate(region_batch.latent_bboxes):
        weight = region_weights[i].to(
            device=canvas_state.output_accumulator.device,
            dtype=canvas_state.output_accumulator.dtype,
        )
        weighted = region_outputs[i * per_region_batch:(i + 1) * per_region_batch] * weight
        canvas_state.output_accumulator[:, :, bbox.y:bbox.y2, bbox.x:bbox.x2] += weighted
        canvas_state.weight_accumulator[:, :, bbox.y:bbox.y2, bbox.x:bbox.x2] += region_weights[i].to(
            device=canvas_state.weight_accumulator.device,
            dtype=canvas_state.weight_accumulator.dtype,
        )


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
    if isinstance(value, torch.Tensor) and value.ndim == 4:
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
        if isinstance(value, torch.Tensor) and value.ndim == 4:
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
        print("[Egregora-ARMD] latent:", tuple(latent.shape), latent.dtype, latent.device)
        print("[Egregora-ARMD] timestep:", tuple(timestep.shape) if hasattr(timestep, "shape") else type(timestep))
        print("[Egregora-ARMD] cond_or_uncond:", cond_or_uncond)
        print("[Egregora-ARMD] input c keys:", sorted(cond_dict.keys()))
        print("[Egregora-ARMD] merged condition keys:", sorted(full_conditions.keys()))
        if "control" in full_conditions:
            print("[Egregora-ARMD] control present in conditioning")
        self._debug_logged = True

    def __call__(self, model_function, args):
        latent: torch.Tensor = args["input"]
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

        for batch in self.build_batches():
            x_regions = slice_latent_regions(latent, batch.latent_bboxes)
            t_regions = repeat_to_batch_size(timestep, x_regions.shape[0], dim=0)

            c_regions = {}
            control_obj = None

            for key, value in full_conditions.items():
                if key == "c_crossattn":
                    continue
                if key == "control":
                    control_obj = clone_control_chain_for_batch(value, batch, latent, self.runtime_inputs.region_plan)
                    if self.runtime_inputs.debug_runtime and not getattr(self, "_control_debug_logged", False):
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

            if control_obj is not None:
                if self.runtime_inputs.debug_runtime and not getattr(self, "_control_call_debug_logged", False):
                    print("[Egregora-ARMD] x_regions:", tuple(x_regions.shape))
                    print("[Egregora-ARMD] control dict sliced for current regional batch")
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
                "image": ("IMAGE",),
                "region_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "region_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "region_overlap": ("INT", {"default": 64, "min": 0, "max": 2048, "step": 8}),
                "compression": ("INT", {"default": 8, "min": 1, "max": 16, "step": 1}),
                "alignment_mode": (
                    [
                        "pad_reflect",
                        "floor_crop",
                    ],
                    {"default": "pad_reflect"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "EGREGORA_REGION_PLAN", "INT", "STRING")
    RETURN_NAMES = ("aligned_image", "regions_batch", "regions_list", "region_plan", "region_count", "region_order_text")
    OUTPUT_IS_LIST = (False, False, True, False, False, False)
    FUNCTION = "plan"
    CATEGORY = "Egregora-ARMD"

    def plan(self, image, region_width, region_height, region_overlap, compression, alignment_mode):
        aligned, alignment_meta = align_image_to_compression(image, compression, alignment_mode)
        _, h, w, _ = aligned.shape
        _, original_h, original_w, _ = image.shape
        plan = build_region_plan(
            aligned_width=w,
            aligned_height=h,
            region_width=region_width,
            region_height=region_height,
            region_overlap=region_overlap,
            compression=compression,
            original_width=original_w,
            original_height=original_h,
            alignment_mode=str(alignment_meta["alignment_mode"]),
            pad_left=int(alignment_meta["pad_left"]),
            pad_top=int(alignment_meta["pad_top"]),
            pad_right=int(alignment_meta["pad_right"]),
            pad_bottom=int(alignment_meta["pad_bottom"]),
        )
        regions = extract_region_images(aligned, plan)
        regions_batch = torch.cat(regions, dim=0) if regions else aligned[:0]
        return (
            aligned,
            regions_batch,
            regions,
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
        target_h = int(region_plan.original_height)
        target_w = int(region_plan.original_width)

        if h == target_h and w == target_w:
            return (image,)

        if any((region_plan.pad_left, region_plan.pad_top, region_plan.pad_right, region_plan.pad_bottom)):
            y1 = int(region_plan.pad_top)
            x1 = int(region_plan.pad_left)
            y2 = y1 + target_h
            x2 = x1 + target_w
            if y2 <= h and x2 <= w:
                return (image[:, y1:y2, x1:x2, :],)

        restored = resize_bhwc(image, target_w, target_h)
        return (restored,)


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
