using Oceananigans.Grids: topology

"""
    adapt_advection_order(advection, grid::AbstractGrid)

Adapts the advection operator `advection` based on the grid `grid` by adjusting the order of advection in each direction.
For example, if the grid has only one point in the x-direction, the advection operator in the x-direction is set to first order
upwind or 2nd order centered scheme, depending on the original user-specified advection scheme. A high order advection sheme 
is reduced to a lower order advection scheme if the grid has fewer points in that direction.

# Arguments
- `advection`: The original advection scheme.
- `grid::AbstractGrid`: The grid on which the advection scheme is applied.

If the order of advection is changed in at least one direction, the adapted advection scheme with adjusted advection order returned 
by this function is a `FluxFormAdvection`.
"""
function adapt_advection_order(advection, grid::AbstractGrid)
    advection_x = x_advection(advection)
    advection_y = y_advection(advection)
    advection_z = z_advection(advection)

    tx = topology(grid, 1)()
    ty = topology(grid, 2)()
    tz = topology(grid, 3)()

    # Note: no adaptation in Flat directions.
    new_advection_x = adapt_advection_order(tx, advection_x, size(grid, 1), grid)
    new_advection_y = adapt_advection_order(ty, advection_y, size(grid, 2), grid)
    new_advection_z = adapt_advection_order(tz, advection_z, size(grid, 3), grid)

    # Check that we indeed changed the advection operator
    changed_x = new_advection_x != advection_x
    changed_y = new_advection_y != advection_y
    changed_z = new_advection_z != advection_z

    new_advection = FluxFormAdvection(new_advection_x, new_advection_y, new_advection_z)
    changed_advection = any((changed_x, changed_y, changed_z))

    if changed_x
        @info "Using the advection scheme $(summary(new_advection.x)) in the x-direction because size(grid, 1) = $(size(grid, 1))"
    end
    if changed_y
        @info "Using the advection scheme $(summary(new_advection.y)) in the y-direction because size(grid, 2) = $(size(grid, 2))"
    end
    if changed_z
        @info "Using the advection scheme $(summary(new_advection.z)) in the z-direction because size(grid, 3) = $(size(grid, 3))"
    end

    return ifelse(changed_advection, new_advection, advection)
end

x_advection(flux_form::FluxFormAdvection) = flux_form.x
y_advection(flux_form::FluxFormAdvection) = flux_form.y
z_advection(flux_form::FluxFormAdvection) = flux_form.z

x_advection(advection) = advection
y_advection(advection) = advection
z_advection(advection) = advection

# Don't adapt advection in Flat directions
adapt_advection_order(topo, advection, N, grid) = adapt_advection_order(advection, N, grid)
adapt_advection_order(::Flat, advection, N, grid) = advection

# For the moment, we do not adapt the advection order for the VectorInvariant advection scheme
adapt_advection_order(advection::VectorInvariant, grid::AbstractGrid) = advection
adapt_advection_order(advection::Nothing, grid::AbstractGrid) = nothing
adapt_advection_order(advection::Nothing, N::Int, grid::AbstractGrid) = nothing
adapt_advection_order(advection, N, grid) = advection

#####
##### Directional adapt advection order
#####

function adapt_advection_order(advection::Centered{B}, N::Int, grid::AbstractGrid) where B
    if N >= B
        return advection
    else
        return Centered(; order=2N)
    end
end

function adapt_advection_order(advection::UpwindBiased{B}, N::Int, grid::AbstractGrid) where B
    if N >= B
        return advection
    else
        return UpwindBiased(; order=2N-1)
    end
end
function adapt_advection_order(advection::WENO{B}, N::Int, grid::AbstractGrid) where B
    if N >= B
        return advection
    else
        return WENO(order=2N-1)
    end
end