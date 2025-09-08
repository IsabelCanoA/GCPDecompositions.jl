## SparseTensorCOO: coordinate format

"""
    SparseTensorCOO{Tv,Ti<:Integer,N} <: AbstractSparseTensor{Tv,Ti,N}

`N`-dimensional sparse tensor stored in the **COO**rdinate format.
Elements are stored as a vector of indices (using type `Ti`)
and a vector of values (of type `Tv`).

Fields:
+ `dims::Dims{N}`              : tuple of dimensions
+ `inds::Vector{NTuple{N,Ti}}` : vector of indices
+ `vals::Vector{Tv}`           : vector of values
"""
struct SparseTensorCOO{Tv,Ti<:Integer,N} <: AbstractSparseTensor{Tv,Ti,N}
    dims::Dims{N}                   # Dimensions
    inds::Vector{NTuple{N,Ti}}      # Stored indices
    vals::Vector{Tv}                # Stored values

    function SparseTensorCOO{Tv,Ti,N}(dims::Dims{N}, inds::Vector{NTuple{N,Ti}},
                            vals::Vector{Tv}) where {Tv,Ti<:Integer,N}
        check_Ti(dims, Ti)
        check_coo_buffers(inds, vals)
        check_coo_inds(dims, inds)
        return new(dims, inds, vals)
    end
end
function SparseTensorCOO(dims::Dims{N}, inds::Vector{NTuple{N,Ti}},
                        vals::Vector{Tv}) where {Tv,Ti<:Integer,N}
    if issorted(inds; by = reverse)
        _inds = inds
        _vals = vals
    else
        perm = sortperm(inds; by = reverse)
        _inds = inds[perm]
        _vals = vals[perm]
    end
    SparseTensorCOO{Tv,Ti,N}(dims, _inds, _vals)
end

"""
    SparseTensorCOO{Tv,Ti<:Integer}(undef, dims)
    SparseTensorCOO{Tv,Ti<:Integer,N}(undef, dims)

Construct an uninitialized `N`-dimensional `SparseTensorCOO`
with indices using type `Ti` and elements of type `Tv`.
Here uninitialized means it has no stored entries.

Here `undef` is the `UndefInitializer`. If `N` is supplied,
then it must match the length of `dims`.

# Examples
```julia-repl
julia> A = SparseTensorCOO{Float64, Int8, 3}(undef, (2, 3, 4)) # N given explicitly
2×3×4 SparseTensorCOO{Float64, Int8, 3} with 0 stored entries

julia> B = SparseTensorCOO{Float64, Int8}(undef, (4,)) # N determined by the input
4-element SparseTensorCOO{Float64, Int8, 1} with 0 stored entries

julia> similar(B, 2, 4, 1) # use typeof(B), and the given size
2×4×1 SparseTensorCOO{Float64, Int8, 3} with 0 stored entries
```
"""
SparseTensorCOO{Tv,Ti,N}(::UndefInitializer, dims::Dims{N}) where {Tv,Ti<:Integer,N} =
    SparseTensorCOO(dims, Vector{NTuple{N,Ti}}(), Vector{Tv}())
SparseTensorCOO{Tv,Ti}(::UndefInitializer, dims::Dims{N}) where {Tv,Ti<:Integer,N} =
    SparseTensorCOO{Tv,Ti,N}(undef, dims)

"""
    SparseTensorCOO(Ti, A::AbstractArray)

Convert an AbstractArray `A` into a `SparseTensorCOO`
with indices using type `Ti`.

# Examples
```julia-repl
julia> A = SparseTensorCOO(Int8, Float16[1.1 0.0 0.0; 2.1 0.0 2.3])
2×3 SparseTensorCOO{Float16, Int8, 2} with 3 stored entries:
  [1, 1]  =  1.1
  [2, 1]  =  2.1
  [2, 3]  =  2.3

julia> B = SparseTensorCOO(Int16, Float16[1.1 0.0 0.0; 2.1 0.0 2.3])
2×3 SparseTensorCOO{Float16, Int16, 2} with 3 stored entries:
  [1, 1]  =  1.1
  [2, 1]  =  2.1
  [2, 3]  =  2.3
```
"""
function SparseTensorCOO(Ti::Type{<:Integer}, A::AbstractArray)
    Tv, N = eltype(A), ndims(A)
    dims = size(A)
    nzidx = findall(!iszero, A)
    inds = convert(Vector{NTuple{N,Ti}}, CartesianIndices(A)[nzidx])
    vals = convert(Vector{Tv}, A[nzidx])
    SparseTensorCOO{Tv,Ti,N}(dims, inds, vals)
end

## Minimal AbstractArray interface

size(A::SparseTensorCOO) = A.dims

function getindex(A::SparseTensorCOO{Tv,<:Integer,N}, I::Vararg{Int,N}) where {Tv,N}
    @boundscheck checkbounds(A, I...)
    ptr = searchsortedfirst(A.inds, I; by = reverse)
    return (ptr == length(A.inds) + 1 || A.inds[ptr] != I) ? zero(Tv) : A.vals[ptr]
end

function setindex!(A::SparseTensorCOO{Tv,Ti,N}, v, I::Vararg{Int,N}) where {Tv,Ti<:Integer,N}
    @boundscheck checkbounds(A, I...)
    ind, val = convert(NTuple{N,Ti}, I), convert(Tv, v)
    ptr = searchsortedfirst(A.inds, ind; by = reverse)
    if ptr == length(A.inds) + 1 || A.inds[ptr] != ind
        if !iszero(val)
            insert!(A.inds, ptr, ind)
            insert!(A.vals, ptr, val)
        end
    else
        A.vals[ptr] = val
    end
    return A
end

IndexStyle(::Type{<:SparseTensorCOO}) = IndexCartesian()

## Overloads for specializing outputs

similar(::SparseTensorCOO{<:Any,Ti}, ::Type{Tv}, dims::Dims{N}) where {Tv,Ti<:Integer,N} =
    SparseTensorCOO{Tv,Ti,N}(undef, dims)

## Overloads for improving efficiency

# technically specializes the output since the state is different
function iterate(A::SparseTensorCOO{Tv}, state=((eachindex(A),),1)) where {Tv}
    idxstate, nextptr = state
    y = iterate(idxstate...)
    y === nothing && return nothing
    if nextptr > length(A.inds) || A.inds[nextptr] != Tuple(y[1])
        val = zero(Tv)
    else
        val = A.vals[nextptr]
        nextptr += 1
    end
    val, ((idxstate[1], Base.tail(y)...), nextptr)
end

## AbstractSparseTensor interface

function dropstored!(f::Function, A::SparseTensorCOO)
    ptrs = findall(f, A.vals)
    deleteat!(A.inds, ptrs)
    deleteat!(A.vals, ptrs)
    return A
end

numstored(A::SparseTensorCOO) = length(A.vals)
storedindices(A::SparseTensorCOO) = A.inds
storedvalues(A::SparseTensorCOO) = A.vals
storedpairs(A::SparseTensorCOO) = Iterators.map(Pair, A.inds, A.vals)

## AbstractSparseTensor optional interface (internal)

findall_stored(f::Function, A::SparseTensorCOO) =
    [convert(keytype(A), CartesianIndex(ind)) for (ind, val) in storedpairs(A) if f(val)]

## Utilities

"""
    check_coo_buffers(inds, vals)

Check that the `inds` and `vals` buffers are valid:
+ their lengths match (`length(inds) == length(vals)`)
If not, throw an `ArgumentError`.
"""
function check_coo_buffers(inds::Vector{NTuple{N,Ti}}, vals::Vector{Tv}) where {Tv,Ti<:Integer,N}
    length(inds) == length(vals) ||
        throw(ArgumentError("the buffer lengths (length(inds) = $(length(inds)), length(vals) = $(length(vals))) do not match"))
    return nothing
end

"""
    check_coo_inds(dims, inds)

Check that the indices in `inds` are valid:
+ each index is in bounds (`1 ≤ inds[ptr][k] ≤ dims[k]`)
+ the indices are sorted (`issorted(inds; by=CartesianIndex)`)
+ the indices are all unique (`allunique(inds`)
If not, throw an `ArgumentError`.
"""
function check_coo_inds(dims::Dims{N}, inds::Vector{NTuple{N,Ti}}) where {Ti<:Integer,N}
    # Check all the conditions in a single pass over inds for efficiency
    itr = iterate(inds)
    itr === nothing && return nothing
    prevind, state = itr
    checkbounds_dims(dims, prevind...)
    itr = iterate(inds, state)
    while itr !== nothing
        thisind, state = itr
        if reverse(prevind) < reverse(thisind)
            checkbounds_dims(dims, thisind...)
        elseif reverse(prevind) > reverse(thisind)
            throw(ArgumentError("inds are not sorted"))
        else
            throw(ArgumentError("inds are not all unique"))
        end
        prevind = thisind
        itr = iterate(inds, state)
    end
    return nothing
end
