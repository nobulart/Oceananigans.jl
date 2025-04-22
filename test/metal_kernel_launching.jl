using Test
using Metal
using KernelAbstractions
using Oceananigans.Architectures
using Oceananigans.Utils: launch!

# Metal Shading Language kernel source
kernel_source = """
#include <metal_stdlib>
using namespace metal;
kernel void add_one(device float *A [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    A[id] += 1.0f;
}
"""

# Initialize MetalGPU architecture
arch = MetalGPU()
N = 16
A = Metal.MtlArray(zeros(Float32, N))

# Launch the kernel
launch!(arch, kernel_source, "add_one", A; N=N)

# Copy result back to CPU and test
A_host = Array(A)
@assert all(A_host .== 1.0f0)

println("Metal kernel launch test passed.")
