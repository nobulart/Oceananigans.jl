"""
    poisson_eigenvalues(FT, N, L, dim, ::Periodic)

Return the eigenvalues satisfying the discrete form of Poisson's equation
with periodic boundary conditions along the dimension `dim` with `N` grid
points and domain extent `L`.
"""
function poisson_eigenvalues(FT, N, L, dim, ::Periodic)
    inds = reshape(FT.(1:N), reshaped_size(N, dim)...)
    return @. (2sin((inds - FT(1)) * FT(π) / FT(N)) / (FT(L) / FT(N)))^2
end

"""
    poisson_eigenvalues(FT, N, L, dim, ::Bounded)

Return the eigenvalues satisfying the discrete form of Poisson's equation
with staggered Neumann boundary conditions along the dimension `dim` with
`N` grid points and domain extent `L`.
"""
function poisson_eigenvalues(FT, N, L, dim, ::Bounded)
    inds = reshape(FT.(1:N), reshaped_size(N, dim)...)
    return @. (2sin((inds - FT(1)) * FT(π) / (FT(2)*FT(N))) / (FT(L) / FT(N)))^2
end

"""
    poisson_eigenvalues(FT, N, L, dim, ::Flat)

Return N-element array of `0.0` reshaped to three-dimensions.
This is also the first `poisson_eigenvalue` for `Bounded` and `Periodic` directions.
"""
poisson_eigenvalues(FT, N, L, dim, ::Flat) = reshape(zeros(FT, N), reshaped_size(N, dim)...)

"""
    poisson_eigenvalues(N, L, dim, topo)

Fallback for backwards compatibility. Assumes Float64 type.
"""
poisson_eigenvalues(N, L, dim, topo) = poisson_eigenvalues(Float64, N, L, dim, topo)

