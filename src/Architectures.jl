module Architectures

export AbstractArchitecture, AbstractSerialArchitecture
export CPU, GPU, MetalGPU, ReactantState
export device, architecture, unified_array, device_copy_to!
export array_type, on_architecture, arch_array

using CUDA
using KernelAbstractions
using Adapt
using OffsetArrays
using Metal  # Import Metal.jl for Apple Metal support

"""
    AbstractArchitecture

Abstract supertype for architectures supported by Oceananigans.
"""
abstract type AbstractArchitecture end

"""
    AbstractSerialArchitecture

Abstract supertype for serial architectures supported by Oceananigans.
"""
abstract type AbstractSerialArchitecture <: AbstractArchitecture end

"""
    CPU <: AbstractArchitecture

Run Oceananigans on one CPU node. Uses multiple threads if the environment
variable `JULIA_NUM_THREADS` is set.
"""
struct CPU <: AbstractSerialArchitecture end

"""
    GPU(device)

Return a GPU architecture using `device`.
`device` defaults to CUDA.CUDABackend(always_inline=true)
"""
struct GPU{D} <: AbstractSerialArchitecture 
    device :: D
end

const CUDAGPU = GPU{<:CUDA.CUDABackend}
CUDAGPU() = GPU(CUDA.CUDABackend(always_inline=true))
Base.summary(::CUDAGPU) = "CUDAGPU"

function GPU()
    if CUDA.has_cuda_gpu()
        return CUDAGPU()
    else
        msg = """We cannot make a GPU with the CUDA backend:
                 a CUDA GPU was not found!"""
        throw(ArgumentError(msg))
    end
end

"""
    MetalGPU(device::Metal.MTLDevice)

Run Oceananigans on an Apple Metal GPU.
"""
struct MetalGPU <: AbstractSerialArchitecture
    device :: Metal.MTLDevice
end

# Constructor for MetalGPU with error handling
function MetalGPU()
    devs = Metal.devices()
    if isempty(devs)
        throw(ArgumentError("No Apple Metal devices found!"))
    end
    return MetalGPU(devs[1])
end

Base.summary(::MetalGPU) = "MetalGPU"

device(a::MetalGPU) = a.device

# Architecture dispatch for Metal arrays
architecture(::Metal.MtlArray) = MetalGPU()

# Array type for MetalGPU
array_type(::MetalGPU) = Metal.MtlArray

# on_architecture for MetalGPU
on_architecture(::MetalGPU, a::Array) = Metal.MtlArray(a)
on_architecture(::MetalGPU, a::Metal.MtlArray) = a
on_architecture(::MetalGPU, a::BitArray) = Metal.MtlArray(a)
on_architecture(::MetalGPU, a::SubArray{<:Any, <:Any, <:Array}) = Metal.MtlArray(a)

# unified_array for MetalGPU
unified_array(::MetalGPU, a) = a

# device_copy_to! for MetalGPU
@inline device_copy_to!(dst::Metal.MtlArray, src::Metal.MtlArray; kw...) = Metal.copy!(dst, src)

# unsafe_free! for MetalGPU
@inline unsafe_free!(a::Metal.MtlArray) = Metal.unsafe_free!(a)

# convert_to_device for MetalGPU
@inline convert_to_device(::MetalGPU, args) = args
@inline convert_to_device(::MetalGPU, args::Tuple) = map(identity, args)

"""
    ReactantState <: AbstractArchitecture

Run Oceananigans on Reactant.
"""
struct ReactantState <: AbstractSerialArchitecture end

#####
##### These methods are extended in DistributedComputations.jl
#####

device(a::CPU) = KernelAbstractions.CPU()
device(a::GPU) = a.device

architecture() = nothing
architecture(::Number) = nothing
architecture(::Array) = CPU()
architecture(::CuArray) = CUDAGPU()
architecture(a::SubArray) = architecture(parent(a))
architecture(a::OffsetArray) = architecture(parent(a))

"""
    child_architecture(arch)

Return `arch`itecture of child processes.
On single-process, non-distributed systems, return `arch`.
"""
child_architecture(arch::AbstractSerialArchitecture) = arch

array_type(::CPU) = Array
array_type(::GPU) = CuArray

# Fallback 
on_architecture(arch, a) = a

# Tupled implementation
on_architecture(arch::AbstractSerialArchitecture, t::Tuple) = Tuple(on_architecture(arch, elem) for elem in t)
on_architecture(arch::AbstractSerialArchitecture, nt::NamedTuple) = NamedTuple{keys(nt)}(on_architecture(arch, Tuple(nt)))

# On architecture for array types
on_architecture(::CPU, a::Array) = a
on_architecture(::CPU, a::BitArray) = a
on_architecture(::CPU, a::CuArray) = Array(a)
on_architecture(::CPU, a::SubArray{<:Any, <:Any, <:CuArray}) = Array(a)
on_architecture(::CPU, a::SubArray{<:Any, <:Any, <:Array}) = a
on_architecture(::CPU, a::StepRangeLen) = a

on_architecture(::CUDAGPU, a::Array) = CuArray(a)
on_architecture(::CUDAGPU, a::CuArray) = a
on_architecture(::CUDAGPU, a::BitArray) = CuArray(a)
on_architecture(::CUDAGPU, a::SubArray{<:Any, <:Any, <:CuArray}) = a
on_architecture(::CUDAGPU, a::SubArray{<:Any, <:Any, <:Array}) = CuArray(a)
on_architecture(::CUDAGPU, a::StepRangeLen) = a

on_architecture(arch::AbstractSerialArchitecture, a::OffsetArray) =
    OffsetArray(on_architecture(arch, a.parent), a.offsets...)

cpu_architecture(::CPU) = CPU()
cpu_architecture(::GPU) = CPU()
cpu_architecture(::ReactantState) = CPU()

unified_array(::CPU, a) = a
unified_array(::GPU, a) = a

# cu alters the type of `a`, so we convert it back to the correct type
unified_array(::GPU, a::AbstractArray) = map(eltype(a), cu(a; unified = true))

## GPU to GPU copy of contiguous data
@inline function device_copy_to!(dst::CuArray, src::CuArray; async::Bool = false) 
    n = length(src)
    context!(context(src)) do
        GC.@preserve src dst begin
            unsafe_copyto!(pointer(dst, 1), pointer(src, 1), n; async)
        end
    end
    return dst
end
 
@inline device_copy_to!(dst::Array, src::Array; kw...) = Base.copyto!(dst, src)

@inline unsafe_free!(a::CuArray) = CUDA.unsafe_free!(a)

# Convert arguments to GPU-compatible types
@inline convert_to_device(arch, args)  = args
@inline convert_to_device(::CPU, args) = args
@inline convert_to_device(::CUDAGPU, args) = CUDA.cudaconvert(args)
@inline convert_to_device(::CUDAGPU, args::Tuple) = map(CUDA.cudaconvert, args)

# Deprecated functions
function arch_array(arch, arr) 
    @warn "`arch_array` is deprecated. Use `on_architecture` instead."
    return on_architecture(arch, arr)
end

@inline unsafe_free!(a) = nothing

import KernelAbstractions: isgpu
isgpu(dev::Metal.MTL.MTLDeviceInstance) = true

end # module