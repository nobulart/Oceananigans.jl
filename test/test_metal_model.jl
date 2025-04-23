# Metal end-to-end model test for Oceananigans
using Test
using Oceananigans
using Oceananigans.Architectures
using Metal

println("[DEBUG] Metal.devices(): ", Metal.devices())

arch = nothing
try
    global arch
    arch = MetalGPU()
catch e
    @info "No Metal GPU available, skipping Metal model test."
    @info "[DEBUG] MetalGPU() constructor error: $e"
    @info "[DEBUG] Exception type: $(typeof(e))"
    @info "[DEBUG] Exception stacktrace: $(catch_backtrace())"
end

if arch === nothing
    @info "No Metal GPU available, skipping Metal model test."
    exit(0)
end

println("[DEBUG] arch: ", arch)

grid = RectilinearGrid(Float32, size=(8, 1, 1), extent=(1.0, 1.0, 1.0))

println("[DEBUG] grid eltype: ", eltype(grid))
println("[DEBUG] grid Lx type: ", typeof(grid.Lx))
println("[DEBUG] grid xᶠᵃᵃ type: ", typeof(grid.xᶠᵃᵃ))
println("[DEBUG] grid xᶠᵃᵃ eltype: ", eltype(grid.xᶠᵃᵃ))

# Minimal Metal array allocation test
A = Metal.MtlArray(zeros(Float32, 8, 1, 1))
println("[DEBUG] Allocated Metal.MtlArray with Float32: ", typeof(A), ", eltype: ", eltype(A))

model = NonhydrostaticModel(grid=grid, architecture=arch)

set!(model.velocities.u, 1.0)
set!(model.velocities.v, 0.0)
set!(model.velocities.w, 0.0)

sim = Simulation(model, Δt=0.01, stop_iteration=5)
run!(sim)

@test isfinite(sum(Array(model.velocities.u)))
println("Metal model simulation test passed.")
