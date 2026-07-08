from typing import Any, Optional

import torch

from .triton.attn_qk_int8_per_block import forward as attn_noncausal
from .triton.quant_per_block import per_block_int8 as per_block_int8_triton


def _unsupported(name: str):
    raise NotImplementedError(
        f"{name} is not included in the HIP57 core wheel. "
        "This build only supports fixed-length non-causal "
        "qk-int8/pv-fp16 Triton attention."
    )


def sageattn_varlen(*args: Any, **kwargs: Any):
    _unsupported("sageattn_varlen")


def sageattn_qk_int8_pv_fp16_cuda(*args: Any, **kwargs: Any):
    _unsupported("sageattn_qk_int8_pv_fp16_cuda")


def sageattn_qk_int8_pv_fp8_cuda(*args: Any, **kwargs: Any):
    _unsupported("sageattn_qk_int8_pv_fp8_cuda")


def sageattn_qk_int8_pv_fp8_cuda_sm90(*args: Any, **kwargs: Any):
    _unsupported("sageattn_qk_int8_pv_fp8_cuda_sm90")


def should_force_triton_backend(device_index: int) -> bool:
    if getattr(torch.version, "hip", None):
        return True
    if not torch.cuda.is_available():
        return False
    try:
        name = torch.cuda.get_device_name(device_index).lower()
    except Exception:
        return False
    return "zluda" in name or "amd" in name or "radeon" in name


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    **kwargs: Any,
):
    return sageattn_qk_int8_pv_fp16_triton(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        sm_scale=sm_scale,
        return_lse=return_lse,
        **kwargs,
    )


def sageattn_qk_int8_pv_fp16_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    quantization_backend: str = "triton",
    is_causal: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    if is_causal:
        _unsupported("causal attention")
    if attn_mask is not None:
        _unsupported("attn_mask")
    if return_lse:
        _unsupported("return_lse")
    if quantization_backend != "triton":
        _unsupported(f"quantization_backend={quantization_backend!r}")

    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], (
        "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    )
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    torch.cuda.set_device(v.device)

    head_dim_og = q.size(-1)
    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, (
        "Last dim of qkv must be contiguous."
    )

    seq_dim = 1 if tensor_layout == "NHD" else 2
    km = k.mean(dim=seq_dim, keepdim=True) if smooth_k else None

    if dtype == torch.bfloat16:
        v = v.to(torch.float16)

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og ** 0.5)

    q_int8, q_scale, k_int8, k_scale = per_block_int8_triton(
        q,
        k,
        km=km,
        sm_scale=sm_scale,
        tensor_layout=tensor_layout,
    )
    o, _ = attn_noncausal(
        q_int8,
        k_int8,
        v,
        q_scale,
        k_scale,
        tensor_layout=tensor_layout,
        output_dtype=dtype,
        attn_mask=None,
        return_lse=False,
    )
    return o[..., :head_dim_og]
