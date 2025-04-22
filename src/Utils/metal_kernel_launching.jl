# Minimal Metal kernel launcher utility
using Metal

"""
    metal_launch!(arch::MetalGPU, kernel_source::String, fn_name::String, args...; N)

Compile and launch a Metal kernel from source on the given MetalGPU.
- kernel_source: Metal Shading Language (MSL) kernel as a string.
- fn_name: Name of the kernel function in the MSL source.
- args: Metal.MtlArray arguments.
- N: number of elements to launch over.
"""
function metal_launch!(arch::MetalGPU, kernel_source::String, fn_name::String, args...; N)
    device = arch.device
    program = Metal.Program(device, kernel_source)
    kernel = Metal.Kernel(program, fn_name)
    queue = Metal.CmdQueue(device)
    event = Metal.launch(queue, kernel, (N,), args...)
    Metal.synchronize(event)
    return nothing
end
