using Test
using Metal
using .MPSFFT

# Create a Metal array with random Float32 data
A = Metal.MtlArray(rand(Float32, 16))

# Try to run the forward FFT (this will only work if the Objective-C bindings are implemented)
B = MPSFFT.metal_fft_forward!(A)

println("Result type: ", typeof(B))
@test size(B) == size(A)
