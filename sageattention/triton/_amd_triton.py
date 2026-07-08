import os

import torch
import triton
import triton.language as tl


BLKQ = 32
BLKK = 16


def use_amd_triton_compat(tensor):
    if os.getenv("SAGEATTN_DISABLE_AMD_TRITON_COMPAT", "0").upper() in {"1", "TRUE", "YES"}:
        return False
    if getattr(torch.version, "hip", None):
        return True
    if not getattr(tensor, "is_cuda", False):
        return False
    try:
        index = tensor.device.index
        if index is None:
            index = torch.cuda.current_device()
        name = torch.cuda.get_device_name(index).lower()
    except Exception:
        return False
    return "amd" in name or "radeon" in name or "zluda" in name


@triton.jit
def _fixed_dot_kernel(
    Q,
    K,
    V,
    Q_scale,
    K_scale,
    Out,
    stride_qz,
    stride_qh,
    stride_qn,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_on,
    QO_LEN,
    KV_LEN,
    H: tl.constexpr,
    num_kv_groups: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_m_block = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    off_h_kv = off_h // num_kv_groups

    offs_m = off_m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qn + offs_d[None, :]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < QO_LEN, other=0)
    q_scale = tl.load(
        Q_scale
        + (off_z * H + off_h) * tl.cdiv(QO_LEN, BLOCK_Q)
        + (off_m_block * BLOCK_M) // BLOCK_Q
    )

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    k_scale_base = (off_z * (H // num_kv_groups) + off_h_kv) * tl.cdiv(KV_LEN, BLOCK_K)

    for start_n in range(0, KV_LEN, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n = start_n + offs_n
        k_scale = tl.load(K_scale + k_scale_base + start_n // BLOCK_K)
        k_ptrs = K + off_z * stride_kz + off_h_kv * stride_kh + offs_d[:, None] + n[None, :] * stride_kn
        k = tl.load(k_ptrs, mask=n[None, :] < KV_LEN, other=0)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale
        qk = tl.where((offs_m[:, None] < QO_LEN) & (n[None, :] < KV_LEN), qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = V + off_z * stride_vz + off_h_kv * stride_vh + n[:, None] * stride_vn + offs_d[None, :]
        v = tl.load(v_ptrs, mask=n[:, None] < KV_LEN, other=0)
        acc += tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)
        m_i = m_new

    o_ptrs = Out + off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_on + offs_d[None, :]
    tl.store(o_ptrs, acc / l_i[:, None], mask=offs_m[:, None] < QO_LEN)


def fixed_forward(
    q,
    k,
    v,
    q_scale,
    k_scale,
    tensor_layout="HND",
    is_causal=False,
    output_dtype=torch.float16,
    return_lse=False,
):
    if is_causal:
        raise NotImplementedError("causal attention is not included in the HIP57 core wheel.")
    if return_lse:
        raise NotImplementedError("return_lse is not included in the HIP57 core wheel.")

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")

    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)
    if tensor_layout == "HND":
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    else:
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)

    block_m = 16
    block_n = 16
    grid = (triton.cdiv(qo_len, block_m), h_qo, b)
    _fixed_dot_kernel[grid](
        q,
        k,
        v,
        q_scale,
        k_scale,
        o,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        stride_bz_v,
        stride_h_v,
        stride_seq_v,
        stride_bz_o,
        stride_h_o,
        stride_seq_o,
        qo_len,
        kv_len,
        h_qo,
        h_qo // h_kv,
        HEAD_DIM=head_dim,
        BLOCK_Q=BLKQ,
        BLOCK_K=BLKK,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=2,
        num_stages=1,
    )
    lse = torch.empty([0], dtype=torch.float32, device="cpu")
    return o, lse
