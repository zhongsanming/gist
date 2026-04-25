"""
Benchmark for the extracted lerp_tensor_ implementation.
Measures latency vs. PyTorch native torch.lerp_ for shape (1024, 1024, 1024), float16.
"""

import math
import torch
import triton
import triton.language as tl
import triton.testing


# ── scalar function ──────────────────────────────────────────────────────────
@triton.jit
def _lerp_scalar(input, end, weight):
    return tl.where(
        tl.abs(weight) < 0.5,
        input + weight * (end - input),
        end - (end - input) * (1 - weight),
    )


# ── Triton kernel ────────────────────────────────────────────────────────────
@triton.jit
def _lerp_tensor_kernel_rank1(
    in0_ptr: tl.tensor,
    in1_ptr: tl.tensor,
    in2_ptr: tl.tensor,
    out0_ptr: tl.tensor,
    in0_stride0: int,
    in0_stride_order0: tl.constexpr,
    in1_stride0: int,
    in1_stride_order0: tl.constexpr,
    in2_stride0: int,
    in2_stride_order0: tl.constexpr,
    out0_stride0: int,
    out0_stride_order0: tl.constexpr,
    s0,
    num_tasks,
    tiles_per_cta: int,
    tile_size0: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
):
    pid = tl.program_id(0)

    if one_tile_per_cta:
        tile_id0 = pid
        offset0 = (tile_id0 * tile_size0).to(tl.int32)

        in0_bptr = tl.make_block_ptr(
            in0_ptr,
            (s0,),
            (in0_stride0,),
            (offset0,),
            (tile_size0,),
            order=(in0_stride_order0,),
        )
        in0 = tl.load(in0_bptr, boundary_check=(in0_stride_order0,)).to(
            in0_ptr.type.element_ty
        )
        in1_bptr = tl.make_block_ptr(
            in1_ptr,
            (s0,),
            (in1_stride0,),
            (offset0,),
            (tile_size0,),
            order=(in1_stride_order0,),
        )
        in1 = tl.load(in1_bptr, boundary_check=(in1_stride_order0,)).to(
            in1_ptr.type.element_ty
        )
        in2_bptr = tl.make_block_ptr(
            in2_ptr,
            (s0,),
            (in2_stride0,),
            (offset0,),
            (tile_size0,),
            order=(in2_stride_order0,),
        )
        in2 = tl.load(in2_bptr, boundary_check=(in2_stride_order0,)).to(
            in2_ptr.type.element_ty
        )

        out0 = _lerp_scalar(in0, in1, in2)

        out0_bptr = tl.make_block_ptr(
            out0_ptr,
            (s0,),
            (out0_stride0,),
            (offset0,),
            (tile_size0,),
            order=(out0_stride_order0,),
        )
        tl.store(
            out0_bptr,
            out0.to(out0_bptr.type.element_ty),
            boundary_check=(out0_stride_order0,),
        )

    else:  # grid-stride-loop (our case: tiles_per_cta=32)
        num_ctas = tl.num_programs(0)
        for j in range(0, tiles_per_cta):
            tile_id = pid + j * num_ctas
            offset0 = (tile_id * tile_size0).to(tl.int32)

            in0_bptr = tl.make_block_ptr(
                in0_ptr,
                (s0,),
                (in0_stride0,),
                (offset0,),
                (tile_size0,),
                order=(in0_stride_order0,),
            )
            in0 = tl.load(in0_bptr, boundary_check=(in0_stride_order0,)).to(
                in0_ptr.type.element_ty
            )
            in1_bptr = tl.make_block_ptr(
                in1_ptr,
                (s0,),
                (in1_stride0,),
                (offset0,),
                (tile_size0,),
                order=(in1_stride_order0,),
            )
            in1 = tl.load(in1_bptr, boundary_check=(in1_stride_order0,)).to(
                in1_ptr.type.element_ty
            )
            in2_bptr = tl.make_block_ptr(
                in2_ptr,
                (s0,),
                (in2_stride0,),
                (offset0,),
                (tile_size0,),
                order=(in2_stride_order0,),
            )
            in2 = tl.load(in2_bptr, boundary_check=(in2_stride_order0,)).to(
                in2_ptr.type.element_ty
            )

            out0 = _lerp_scalar(in0, in1, in2)

            out0_bptr = tl.make_block_ptr(
                out0_ptr,
                (s0,),
                (out0_stride0,),
                (offset0,),
                (tile_size0,),
                order=(out0_stride_order0,),
            )
            tl.store(
                out0_bptr,
                out0.to(out0_bptr.type.element_ty),
                boundary_check=(out0_stride_order0,),
            )


# ── wrapper ──────────────────────────────────────────────────────────────────
def _lerp_tensor_wrapper_rank1(in0, in1, in2, *, out0):
    shape = out0.shape
    num_tasks = out0.numel()
    if num_tasks == 0:
        return out0

    tile_sizes = (min(512, triton.next_power_of_2(shape[0])),)
    tile_size = math.prod(tile_sizes)
    num_tiles = math.prod(triton.cdiv(s, ts) for s, ts in zip(shape, tile_sizes))
    num_ctas = min(65536, num_tiles)
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    num_warps = 4 if tile_size < 2048 else (8 if tile_size < 4096 else 16)
    one_tile_per_cta = tiles_per_cta == 1
    grid = (num_ctas, 1, 1)

    in0_strides = in0.stride()
    in1_strides = in1.stride()
    in2_strides = in2.stride()
    out0_strides = out0.stride()

    with torch.cuda.device(in0.device.index):
        _lerp_tensor_kernel_rank1[grid](
            in0,
            in1,
            in2,
            out0,
            in0_strides[0],
            0,
            in1_strides[0],
            0,
            in2_strides[0],
            0,
            out0_strides[0],
            0,
            shape[0],
            num_tasks,
            tiles_per_cta=tiles_per_cta,
            tile_size0=tile_sizes[0],
            one_tile_per_cta=one_tile_per_cta,
            num_warps=num_warps,
        )
    return out0


def lerp_tensor_(input, end, weight):
    flat_input = input.view(-1)
    flat_end = end.view(-1)
    flat_weight = weight.view(-1)
    _lerp_tensor_wrapper_rank1(flat_input, flat_end, flat_weight, out0=flat_input)
    return input


# ── benchmark ────────────────────────────────────────────────────────────────
def main():
    SHAPE = (1024, 1024, 1024)
    DTYPE = torch.float16
    DEVICE = "cuda"

    # allocate inputs once; reuse across benchmark iterations
    inp = torch.rand(*SHAPE, dtype=DTYPE, device=DEVICE)
    end = torch.rand(*SHAPE, dtype=DTYPE, device=DEVICE) + 1.0
    weight = torch.rand(*SHAPE, dtype=DTYPE, device=DEVICE)

    # warm-up: compile the kernel before timing
    lerp_tensor_(inp, end, weight)
    torch.cuda.synchronize()

    # --- FlagGems extracted kernel ---
    ms_flaggems = triton.testing.do_bench(
        lambda: lerp_tensor_(inp, end, weight),
        warmup=25,  # additional warm-up iterations inside do_bench
        rep=100,  # measurement repetitions
        return_mode="median",
    )

    # --- bandwidth calculation ---
    # 3 reads (input, end, weight) + 1 write (input in-place)
    bytes_accessed = 4 * inp.numel() * inp.element_size()  # 4 tensors × 2 bytes
    gb = bytes_accessed / 1e9

    print(f"Shape : {SHAPE}, dtype: {DTYPE}")
    print(f"Data  : {gb:.2f} GB accessed (3R + 1W)")
    print()
    print(f"{'Impl':<25} {'Latency (ms)':>14} {'Bandwidth (GB/s)':>18}")
    print("-" * 60)
    print(
        f"{'FlagGems extracted':<25} {ms_flaggems:>14.3f} {gb / (ms_flaggems * 1e-3):>18.1f}"
    )


if __name__ == "__main__":
    main()
