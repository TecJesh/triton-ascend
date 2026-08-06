import torch
import torch_npu
import pytest
import triton
import triton.language as tl

device = "npu"


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K)
        mask_b = (offs_k[:, None] < K - k * BLOCK_K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a, b, input_precision=INPUT_PRECISION)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


@pytest.mark.parametrize("input_precision", ["tf32"])
def test_dot_input_precision(input_precision):
    M, N, K = 128, 128, 128
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32

    a = torch.randn(M, K, dtype=torch.float16, device=device)
    b = torch.randn(K, N, dtype=torch.float16, device=device)
    c = torch.zeros(M, N, dtype=torch.float32, device=device)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        a,
        b,
        c,
        M=M,
        N=N,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        INPUT_PRECISION=input_precision,
    )
    torch_npu.npu.synchronize()

    ref = torch.mm(a.float(), b.float())
    max_diff = (c.cpu() - ref).abs().max().item()
    assert max_diff < 1e-2, f"[{input_precision}] max_diff={max_diff:.2e} exceeds threshold"
