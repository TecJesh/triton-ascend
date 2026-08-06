import torch
import torch_npu
import pytest
import triton
import triton.language as tl

device = "npu"


@triton.jit
def cross_if_kernel(
    x_ptr,
    y_ptr,
    flag_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N
    x = tl.load(x_ptr + offset, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offset, mask=mask).to(tl.float32)
    flag = tl.load(flag_ptr + pid).to(tl.int1)

    # Conditional branches: cross-if fusion target
    if flag:
        tmp = x * 2.0 + y
    else:
        tmp = x + y * 2.0

    result = tl.math.exp(tmp * 0.5) + tl.math.log(tmp + 1.0)
    tl.store(out_ptr + offset, result, mask=mask)


def torch_reference(x, y, flags, N):
    x = x.float()
    y = y.float()
    result = torch.zeros_like(x)
    for i in range(flags.shape[0]):
        start = i * (N // flags.shape[0])
        end = start + (N // flags.shape[0])
        if end > N:
            end = N
        if flags[i]:
            tmp = x[start:end] * 2.0 + y[start:end]
        else:
            tmp = x[start:end] + y[start:end] * 2.0
        result[start:end] = torch.exp(tmp * 0.5) + torch.log(tmp + 1.0)
    return result


@pytest.mark.parametrize("N,BLOCK", [(1024, 256), (4096, 512)])
def test_cross_if_fusion(N, BLOCK):
    grid = triton.cdiv(N, BLOCK)
    flags = torch.randint(0, 2, (grid, ), dtype=torch.int32, device=device)

    x = torch.randn(N, dtype=torch.float16, device=device)
    y = torch.randn(N, dtype=torch.float16, device=device)
    out = torch.zeros(N, dtype=torch.float16, device=device)

    cross_if_kernel[grid](
        x,
        y,
        flags,
        out,
        N=N,
        BLOCK=BLOCK,
        enable_cross_if_fusion=True,
    )
    torch_npu.npu.synchronize()

    ref = torch_reference(x.cpu(), y.cpu(), flags.cpu(), N)
    max_diff = (out.cpu().float() - ref).abs().max().item()
    assert max_diff < 1e-2, f"max_diff={max_diff:.2e} exceeds threshold"
