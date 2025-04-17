module MetalGPUTests

using Test
using Metal
using Oceananigans
using Oceananigans.Architectures: GPU, array_type
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel, SplitExplicitFreeSurface
using Oceananigans.TurbulenceClosures: WENO
using Oceananigans.Coriolis: FPlane
using Oceananigans.BuoyancyFormulations: BuoyancyTracer
using Oceananigans.Grids: RectilinearGrid, architecture
using Oceananigans.Simulations: Simulation
using Oceananigans.Units: minute, minutes
using Oceananigans.Utils: time

Oceananigans.defaults.FloatType = Float32

@testset "MetalGPU extension" begin
    metal = Metal.MetalBackend()
    arch = GPU(metal)
    grid = RectilinearGrid(arch, size=(4, 8, 16), x=[0, 1, 2, 3, 4], y=(0, 1), z=(0, 16))

    @test parent(grid.xᶠᵃᵃ) isa MtlArray
    @test parent(grid.xᶜᵃᵃ) isa MtlArray
    @test eltype(grid) == Float32
    @test architecture(grid) isa GPU

    model = HydrostaticFreeSurfaceModel(; grid,
                                        coriolis = FPlane(latitude=45),
                                        buoyancy = BuoyancyTracer(),
                                        tracers = :b,
                                        momentum_advection = WENO(order=5),
                                        tracer_advection = WENO(order=5),
                                        free_surface = SplitExplicitFreeSurface(grid; substeps=60))

    @test parent(model.velocities.u) isa MtlArray
    @test parent(model.velocities.v) isa MtlArray
    @test parent(model.velocities.w) isa MtlArray
    @test parent(model.tracers.b) isa MtlArray

    simulation = Simulation(model, Δt=1minute, stop_iteration=3)
    run!(simulation)

    @test iteration(simulation) == 3
    @test time(simulation) == 3minutes

    # Additional test to confirm Metal device availability
    @test length(Metal.devices()) > 0
end

end # module