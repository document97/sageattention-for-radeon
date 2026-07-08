import torch

from ._amd_triton import fixed_forward


def forward(
    q,
    k,
    v,
    q_scale,
    k_scale,
    tensor_layout="HND",
    attn_mask=None,
    output_dtype=torch.float16,
    return_lse=False,
):
    if attn_mask is not None:
        raise NotImplementedError("attn_mask is not included in the HIP57 core wheel.")
    if return_lse:
        raise NotImplementedError("return_lse is not included in the HIP57 core wheel.")
    return fixed_forward(
        q,
        k,
        v,
        q_scale,
        k_scale,
        tensor_layout=tensor_layout,
        output_dtype=output_dtype,
    )
