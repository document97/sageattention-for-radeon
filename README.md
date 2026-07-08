# SageAttention2 HIP57 Core

Minimal SageAttention2 package for AMD/ZLUDA + HIP SDK 5.7 style Triton
runtime, prepared for ComfyUI video inference.

This is not a full SageAttention distribution. It keeps only the fixed-length,
non-causal QK-int8 / PV-fp16 Triton path that was validated on an AMD Radeon
780M environment exposed to PyTorch as a CUDA device through ZLUDA/native
compatibility.

## What Is Included

- `sageattention.sageattn`
- `sageattention.sageattn_qk_int8_pv_fp16_triton`
- Triton `per_block_int8` quantization for Q/K
- Triton non-causal `attn_qk_int8_per_block`
- AMD-compatible small-block `tl.dot` attention kernel
- HND and NHD tensor layouts
- GQA mapping where query heads are divisible by key/value heads
- Head dimensions up to 128, with padding for dimensions below 64

## What Is Not Included

- CUDA/C++ fused extensions
- NVIDIA SM80/SM89/SM90 wrapper modules
- FP8 PV paths
- Causal attention
- Variable-length attention
- `attn_mask`
- `return_lse=True`
- per-thread/per-warp quantization paths
- Benchmarks, examples, assets, and local development micro-tests

Unsupported entry points are left as importable stubs where useful, but they
raise `NotImplementedError`.

## Target Environment

Validated environment:

- Windows
- Python 3.12
- PyTorch `2.7.1+cu118`
- Triton `3.3.0`
- AMD Radeon 780M Graphics exposed as `cuda:0`
- HIP SDK 5.7 style Clang toolchain available to Triton
- ComfyUI / Aki launcher Python environment

Other AMD GPUs or ROCm/HIP versions may work, but are not guaranteed.

## Install

Install the wheel without dependencies so your existing PyTorch/Triton stack is
not modified:

```powershell
D:\ComfyUI-aki-v2\python\python.exe -m pip install --force-reinstall --no-deps .\sageattention-2.2.0+hip57core-py3-none-any.whl
```

## Smoke Test

```powershell
$env:TRITON_CACHE_DIR = "C:\temp\sageattention-triton-cache"
D:\ComfyUI-aki-v2\python\python.exe -c "import torch; from sageattention import sageattn; q=torch.randn((1,1,16,64),device='cuda',dtype=torch.float16); k=torch.randn((1,1,16,64),device='cuda',dtype=torch.float16); v=torch.randn((1,1,16,64),device='cuda',dtype=torch.float16); o=sageattn(q,k,v,tensor_layout='HND',is_causal=False); torch.cuda.synchronize(); print(o.shape, o.dtype)"
```

Expected shape:

```text
torch.Size([1, 1, 16, 64]) torch.float16
```

The first run for a new kernel shape can take a long time because Triton invokes
HIP SDK Clang to JIT compile GPU code. Later runs in the same process should be
much faster.

## Build The Core Wheel

From this repository root:

```powershell
D:\ComfyUI-aki-v2\python\python.exe setup.py bdist_wheel
```

The wheel will be written to `dist/`.

## Validation Performed

The core wheel was tested from a clean temporary install target, not only from
the source tree:

- `sageattention` import
- `sageattn()` fixed-length non-causal HND smoke test
- GPU output shape and dtype validation

The broader local development tree also validated:

- HND and NHD layouts
- `head_dim=64` and `head_dim=128`
- GQA shape such as `Hq=4, Hkv=2`
- non-block-aligned tails such as `QO=17, KV=19`
- long sequence smoke test such as `NHD (1, 10000, 1, 64)`

## Acknowledgements

This project is based on [thu-ml/sageattention](https://github.com/thu-ml/sageattention).
Thanks to the SageAttention authors for the original SageAttention2
implementation and Apache 2.0 release.

This repository only carries a minimal AMD/HIP57 Triton compatibility path
adapted for the Radeon 780M environment described above.

## License

This project is released under the Apache License 2.0, following the upstream
SageAttention license. See `LICENSE`.
