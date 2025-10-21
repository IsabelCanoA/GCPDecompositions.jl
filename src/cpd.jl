## CP decomposition type

"""
    CPD

Tensor decomposition type for the canonical polyadic decompositions (CPD)
of a tensor (i.e., a multi-dimensional array) `A`.
This is the return type of `gcp(_)`,
the corresponding tensor decomposition function.

If `M::CPD` is the decomposition object,
the weights `λ` and the factor matrices `U = (U[1],...,U[N])`
can be obtained via `M.λ` and `M.U`,
such that `A = Σ_j λ[j] U[1][:,j] ∘ ⋯ ∘ U[N][:,j]`.
"""
struct CPD{T,N,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}}
    λ::Tλ
    U::NTuple{N,TU}
    function CPD{T,N,Tλ,TU}(λ, U) where {T,N,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}}
        Base.require_one_based_indexing(λ, U...)
        for k in Base.OneTo(N)
            size(U[k], 2) == length(λ) || throw(
                DimensionMismatch(
                    "U[$k] has dimensions $(size(U[k])) but λ has length $(length(λ))",
                ),
            )
        end
        return new{T,N,Tλ,TU}(λ, U)
    end
end
CPD(λ::Tλ, U::NTuple{N,TU}) where {T,N,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}} =
    CPD{T,N,Tλ,TU}(λ, U)

# destructuring
Base.iterate(M::CPD) = (M.λ, Val(:U))
Base.iterate(M::CPD, ::Val{:U}) = (M.U, Val(:done))
Base.iterate(::CPD, ::Val{:done}) = nothing

"""
    ncomps(M::CPD)

Return the number of components in `M`.

See also: `ndims`, `size`.
"""
ncomps(M::CPD) = length(M.λ)
ndims(::CPD{T,N}) where {T,N} = N

size(M::CPD{T,N}, dim::Integer) where {T,N} = dim <= N ? size(M.U[dim], 1) : 1
size(M::CPD{T,N}) where {T,N} = ntuple(d -> size(M, d), N)

function show(io::IO, mime::MIME{Symbol("text/plain")}, M::CPD{T,N}) where {T,N}
    # Compute displaysize for showing fields
    LINES, COLUMNS = displaysize(io)
    LINES_FIELD = max(LINES - 2 - N, 0) ÷ (1 + N)
    io_field = IOContext(io, :displaysize => (LINES_FIELD, COLUMNS))

    # Show summary and fields
    summary(io, M)
    println(io)
    println(io, "λ weights:")
    show(io_field, mime, M.λ)
    for k in Base.OneTo(N)
        println(io, "\nU[$k] factor matrix:")
        show(io_field, mime, M.U[k])
    end
end

function summary(io::IO, M::CPD)
    dimstring =
        ndims(M) == 0 ? "0-dimensional" :
        ndims(M) == 1 ? "$(size(M,1))-element" : join(map(string, size(M)), '×')
    _ncomps = ncomps(M)
    return print(
        io,
        dimstring,
        " ",
        typeof(M),
        " with ",
        _ncomps,
        _ncomps == 1 ? " component" : " components",
    )
end

function getindex(M::CPD{T,N}, I::Vararg{Int,N}) where {T,N}
    @boundscheck Base.checkbounds_indices(Bool, axes(M), I) || Base.throw_boundserror(M, I)
    val = zero(eltype(T))
    for j in Base.OneTo(ncomps(M))
        val += M.λ[j] * prod(M.U[k][I[k], j] for k in Base.OneTo(ndims(M)))
    end
    return val
end
getindex(M::CPD{T,N}, I::CartesianIndex{N}) where {T,N} = getindex(M, Tuple(I)...)

function AbstractArray(A::CPD{T,N}) where {T,N}
    out_type = promote_type(eltype.(A.U)..., eltype(A.λ))
    Y = Array{out_type}(undef, size(A))
    return copy!(Y, A; buffers = create_copy_buffers(Y, A))
end
Array(A::CPD) = Array(AbstractArray(A))

function find_split_point(sz, Ndim)
    k_opt = 1
    M_k = sz[1]
    N_k = prod(sz[k_opt+1:end])
    min_cost = M_k + N_k
    for k in 2:(Ndim-1)
        M_k *= sz[k]
        N_k = div(N_k, sz[k])
        cost = M_k + N_k
        if cost < min_cost
            min_cost = cost
            k_opt = k
        end
    end
    return k_opt
end

function create_copy_buffers(Y, A::CPD{T,N}) where {T,N}
    sz = size(A)
    R = size(A.U[1], 2)
    k_opt = find_split_point(sz, ndims(A))

    rows_L = prod(sz[1:k_opt])
    rows_R = prod(sz[k_opt+1:N])

    L_buffer = Array{eltype(Y)}(undef, rows_L, R)
    R_buffer = Array{eltype(Y)}(undef, rows_R, R)

    return (L = L_buffer, R = R_buffer)
end

function copy!(
    Y::AbstractArray,
    A::CPD{T,N};
    buffers = create_copy_buffers(Y, A),
) where {T,N}
    U, λ, sz, R = A.U, A.λ, size(A), size(A.U[1], 2)
    Ndim = ndims(A)

    if Ndim == 1
        mul!(Y, U[1], λ)
        return Y
    end

    # Absorb λ into the smallest factor matrix     
    min_dim = argmin(sz)
    U = ntuple(Val(N)) do k
        if k == min_dim
            return U[k] * Diagonal(λ)
        else
            return U[k]
        end
    end

    k_opt = find_split_point(sz, Ndim)
    L, R_mat = buffers.L, buffers.R

    # Compute "Left" Matrix L 
    TensorKernels.khatrirao!(L, reverse(U[1:k_opt])...)
    TensorKernels.khatrirao!(R_mat, reverse(U[k_opt+1:N])...)

    Y_matrix = reshape(Y, (size(L, 1), size(R_mat, 1)))
    mul!(Y_matrix, L, R_mat')

    return Y
end

norm(M::CPD, p::Real = 2) =
    p == 2 ? norm2(M) : norm((M[I] for I in CartesianIndices(size(M))), p)
function norm2(M::CPD{T,N}) where {T,N}
    V = reduce(.*, M.U[i]'M.U[i] for i in 1:N)
    return sqrt(abs(M.λ' * V * M.λ))
end

"""
    permutedims(M::CPD, perm)

Permute the dimensions (axes) of `M`.
`perm` is a vector or a tuple of length `ndims(M)` specifying the permutation.

The permuted `CPD` object returned by this function is formed without copying
(the output shares storage with the input `M`).
"""
function permutedims(M::CPD, perm)
    (length(perm) == ndims(M) && isperm(perm)) ||
        throw(ArgumentError("no valid permutation of dimensions"))
    return CPD(M.λ, ntuple(k -> M.U[perm[k]], ndims(M)))
end

"""
    normalizecomps(M::CPD, p::Real = 2)

Normalize the components of `M` so that the columns of all its factor matrices
all have `p`-norm equal to unity, i.e., `norm(M.U[k][:, j], p) == 1` for all
`k ∈ 1:ndims(M)` and `j ∈ 1:ncomps(M)`. The excess weight is absorbed into `M.λ`.
Norms equal to zero are ignored (i.e., treated as though they were equal to one).

The following keyword arguments can be used to modify this behavior:
+ `dims` specifies what to normalize (default: `[:λ; 1:ndims(M)]`)
+ `distribute_to` specifies where to distribute the excess weight (default: `:λ`)
Valid options for these arguments are the symbol `:λ`, an integer in `1:ndims(M)`,
or a collection of these.

See also: `normalizecomps!`, `norm`.
"""
normalizecomps(M::CPD, p::Real = 2; dims = [:λ; 1:ndims(M)], distribute_to = :λ) =
    normalizecomps!(deepcopy(M), p; dims, distribute_to)

"""
    normalizecomps!(M::CPD, p::Real = 2)

Normalize the components of `M` in-place so that the columns of all its factor matrices
all have `p`-norm equal to unity, i.e., `norm(M.U[k][:, j], p) == 1` for all
`k ∈ 1:ndims(M)` and `j ∈ 1:ncomps(M)`. The excess weight is absorbed into `M.λ`.
Norms equal to zero are ignored (i.e., treated as though they were equal to one).

The following keyword arguments can be used to modify this behavior:
+ `dims` specifies what to normalize (default: `[:λ; 1:ndims(M)]`)
+ `distribute_to` specifies where to distribute the excess weight (default: `:λ`)
Valid options for these arguments are the symbol `:λ`, an integer in `1:ndims(M)`,
or a collection of these.

See also: `normalizecomps`, `norm`.
"""
function normalizecomps!(
    M::CPD{T,N},
    p::Real = 2;
    dims = [:λ; 1:N],
    distribute_to = :λ,
) where {T,N}
    # Check dims and put into standard (mask) form
    dims_iterable = dims isa Symbol ? (dims,) : dims
    all(d -> d === :λ || (d isa Integer && d in 1:N), dims_iterable) || throw(
        ArgumentError(
            "`dims` must be `:λ`, an integer specifying a mode, or a collection, got $dims",
        ),
    )
    dims_λ = :λ in dims_iterable
    dims_U = ntuple(in(dims_iterable), N)

    # Check distribute_to and put into standard (mask) form
    dist_iterable = distribute_to isa Symbol ? (distribute_to,) : distribute_to
    all(d -> d === :λ || (d isa Integer && d in 1:N), dist_iterable) || throw(
        ArgumentError("`distribute_to` must be `:λ`, an integer specifying a mode, \
                       or a collection, got $distribute_to"),
    )
    dist_λ = :λ in dist_iterable
    dist_U = ntuple(in(dist_iterable), N)

    # Call inner function
    return _normalizecomps!(M, p, dims_λ, dims_U, dist_λ, dist_U)
end

function _normalizecomps!(
    M::CPD{T,N},
    p::Real,
    dims_λ::Bool,
    dims_U::NTuple{N,Bool},
    dist_λ::Bool,
    dist_U::NTuple{N,Bool},
) where {T,N}
    # Utility function to handle zero weights and norms
    zero_to_one(x) = iszero(x) ? oneunit(x) : x

    # Normalize components and collect excess weight
    excess = ones(T, 1, ncomps(M))
    if dims_λ
        norms = map(zero_to_one ∘ abs, M.λ)
        M.λ ./= norms
        excess .*= reshape(norms, 1, ncomps(M))
    end
    for k in Base.OneTo(N)
        if dims_U[k]
            norms = mapslices(zero_to_one ∘ Base.Fix2(norm, p), M.U[k]; dims = 1)
            M.U[k] ./= norms
            excess .*= norms
        end
    end

    # Distribute excess weight (uniformly across specified parts)
    excess .= excess .^ (1 / count((dist_λ, dist_U...)))
    if dist_λ
        M.λ .*= dropdims(excess; dims = 1)
    end
    for k in Base.OneTo(N)
        if dist_U[k]
            M.U[k] .*= excess
        end
    end

    # Return normalized CPD
    return M
end

"""
    permutecomps(M::CPD, perm)

Permute the components of `M`.
`perm` is a vector or a tuple of length `ncomps(M)` specifying the permutation.

See also: `permutecomps!`, `sortcomps`, `sortcomps!`.
"""
permutecomps(M::CPD, perm) = permutecomps!(deepcopy(M), perm)

"""
    permutecomps!(M::CPD, perm)

Permute the components of `M` in-place.
`perm` is a vector or a tuple of length `ncomps(M)` specifying the permutation.

See also: `permutecomps`, `sortcomps`, `sortcomps!`.
"""
permutecomps!(M::CPD, perm) = permutecomps!(M, collect(perm))
function permutecomps!(M::CPD, perm::Vector)
    # Check that perm is a valid permutation
    (length(perm) == ncomps(M) && isperm(perm)) ||
        throw(ArgumentError("`perm` is not a valid permutation of the components"))

    # Permute weights and factor matrices
    M.λ .= M.λ[perm]
    for k in Base.OneTo(ndims(M))
        M.U[k] .= M.U[k][:, perm]
    end

    # Return CPD with permuted components
    return M
end

"""
    sortcomps(M::CPD; dims=:λ, alg::Algorithm=DEFAULT_UNSTABLE, lt=isless, \
              by=identity, rev::Bool=false, order::Ordering=Reverse)

Sort the components of `M`. `dims` specifies what part to sort by;
it must be the symbol `:λ`, an integer in `1:ndims(M)`, or a collection of these.

For the remaining keyword arguments, see the documentation of `sort!`.

See also: `permutecomps`, `permutecomps!`, `sortcomps!`, `sort`, `sort!`.
"""
sortcomps(M::CPD; dims = :λ, order::Ordering = Reverse, kwargs...) =
    sortcomps!(deepcopy(M); dims, order, kwargs...)

"""
    sortcomps!(M::CPD; dims=:λ, alg::Algorithm=DEFAULT_UNSTABLE, lt=isless, \
               by=identity, rev::Bool=false, order::Ordering=Reverse)

Sort the components of `M` in-place. `dims` specifies what part to sort by;
it must be the symbol `:λ`, an integer in `1:ndims(M)`, or a collection of these.

For the remaining keyword arguments, see the documentation of `sort!`.

See also: `permutecomps`, `permutecomps!`, `sortcomps`, `sort`, `sort!`.
"""
sortcomps!(M::CPD; dims = :λ, order::Ordering = Reverse, kwargs...) =
    permutecomps!(M, sortperm(_sortvals(M, dims); order, kwargs...))

function _sortvals(M::CPD, dims)
    # Check dims
    dims_iterable = dims isa Symbol ? (dims,) : dims
    all(d -> d === :λ || (d isa Integer && d in 1:ndims(M)), dims_iterable) || throw(
        ArgumentError(
            "`dims` must be `:λ`, an integer specifying a mode, or a collection, got $dims",
        ),
    )

    # Return vector of values to sort by
    return dims === :λ ? M.λ :
           [map(d -> d === :λ ? M.λ[j] : view(M.U[d], :, j), dims) for j in 1:ncomps(M)]
end
