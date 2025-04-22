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

grid = RectilinearGrid(size=(8, 1, 1), extent=(1.0, 1.0, 1.0))
model = NonhydrostaticModel(grid=grid, architecture=arch)

set!(model.velocities.u, 1.0)
set!(model.velocities.v, 0.0)
set!(model.velocities.w, 0.0)

sim = Simulation(model, Δt=0.01, stop_iteration=5)
run!(sim)

@test isfinite(sum(Array(model.velocities.u)))
println("Metal model simulation test passed.")
