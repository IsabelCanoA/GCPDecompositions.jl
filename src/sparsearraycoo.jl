## Sparse array type

"""
    SparseArrayCOO{Tv,Ti<:Integer,N} <: AbstractSparseArray{Tv,Ti,N}

`N`-dimensional sparse array stored in the **COO**rdinate format.
Elements are stored as a vector of indices (using type `Ti`)
and a vector of values (of type `Tv`).

Fields:
+ `dims::Dims{N}`              : tuple of dimensions
+ `inds::Vector{NTuple{N,Ti}}` : vector of indices
+ `vals::Vector{Tv}`           : vector of values
"""
struct SparseArrayCOO{Tv,Ti<:Integer,N} <: AbstractSparseArray{Tv,Ti,N}
    dims::Dims{N}                   # Dimensions
    inds::Vector{NTuple{N,Ti}}      # Stored indices
    vals::Vector{Tv}                # Stored values

    function SparseArrayCOO{Tv,Ti,N}(dims::Dims{N}, inds::Vector{NTuple{N,Ti}},
                            vals::Vector{Tv}) where {Tv,Ti<:Integer,N}
        check_Ti(dims, Ti)
        check_coo_buffers(inds, vals)
        check_coo_inds(dims, inds)
        return new(dims, inds, vals)
    end
end
function SparseArrayCOO(dims::Dims{N}, inds::Vector{NTuple{N,Ti}},
                        vals::Vector{Tv}) where {Tv,Ti<:Integer,N}
    if issorted(inds; by = reverse)
        _inds = inds
        _vals = vals
    else
        perm = sortperm(inds; by = reverse)
        _inds = inds[perm]
        _vals = vals[perm]
    end
    SparseArrayCOO{Tv,Ti,N}(dims, _inds, _vals)
end

"""
    SparseArrayCOO{Tv,Ti<:Integer}(undef, dims)
    SparseArrayCOO{Tv,Ti<:Integer,N}(undef, dims)

Construct an uninitialized `N`-dimensional `SparseArrayCOO`
with indices using type `Ti` and elements of type `Tv`.
Here uninitialized means it has no stored entries.

Here `undef` is the `UndefInitializer`. If `N` is supplied,
then it must match the length of `dims`.

# Examples
```julia-repl
julia> A = SparseArrayCOO{Float64, Int8, 3}(undef, (2, 3, 4)) # N given explicitly
2×3×4 SparseArrayCOO{Float64, Int8, 3} with 0 stored entries

julia> B = SparseArrayCOO{Float64, Int8}(undef, (4,)) # N determined by the input
4-element SparseArrayCOO{Float64, Int8, 1} with 0 stored entries

julia> similar(B, 2, 4, 1) # use typeof(B), and the given size
2×4×1 SparseArrayCOO{Float64, Int8, 3} with 0 stored entries
```
"""
SparseArrayCOO{Tv,Ti,N}(::UndefInitializer, dims::Dims{N}) where {Tv,Ti<:Integer,N} =
    SparseArrayCOO(dims, Vector{NTuple{N,Ti}}(), Vector{Tv}())
SparseArrayCOO{Tv,Ti}(::UndefInitializer, dims::Dims{N}) where {Tv,Ti<:Integer,N} =
    SparseArrayCOO{Tv,Ti,N}(undef, dims)

"""
    SparseArrayCOO(Ti, A::AbstractArray)

Convert an AbstractArray `A` into a `SparseArrayCOO`
with indices using type `Ti`.

# Examples
```julia-repl
julia> A = SparseArrayCOO(Int8, Float16[1.1 0.0 0.0; 2.1 0.0 2.3])
2×3 SparseArrayCOO{Float16, Int8, 2} with 3 stored entries:
  [1, 1]  =  1.1
  [2, 1]  =  2.1
  [2, 3]  =  2.3

julia> B = SparseArrayCOO(Int16, Float16[1.1 0.0 0.0; 2.1 0.0 2.3])
2×3 SparseArrayCOO{Float16, Int16, 2} with 3 stored entries:
  [1, 1]  =  1.1
  [2, 1]  =  2.1
  [2, 3]  =  2.3
```
"""
function SparseArrayCOO(Ti::Type{<:Integer}, A::AbstractArray)
    Tv, N = eltype(A), ndims(A)
    dims = size(A)
    nzidx = findall(!iszero, A)
    inds = convert(Vector{NTuple{N,Ti}}, CartesianIndices(A)[nzidx])
    vals = convert(Vector{Tv}, A[nzidx])
    SparseArrayCOO{Tv,Ti,N}(dims, inds, vals)
end

## Minimal AbstractArray interface

size(A::SparseArrayCOO) = A.dims

function getindex(A::SparseArrayCOO{Tv,<:Integer,N}, I::Vararg{Int,N}) where {Tv,N}
    @boundscheck checkbounds(A, I...)
    ptr = searchsortedfirst(A.inds, I; by = reverse)
    return (ptr == length(A.inds) + 1 || A.inds[ptr] != I) ? zero(Tv) : A.vals[ptr]
end

function setindex!(A::SparseArrayCOO{Tv,Ti,N}, v, I::Vararg{Int,N}) where {Tv,Ti<:Integer,N}
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

IndexStyle(::Type{<:SparseArrayCOO}) = IndexCartesian()

## Overloads for specializing outputs

similar(::SparseArrayCOO{<:Any,Ti}, ::Type{Tv}, dims::Dims{N}) where {Tv,Ti<:Integer,N} =
    SparseArrayCOO{Tv,Ti,N}(undef, dims)

## Overloads for improving efficiency

# technically specializes the output since the state is different
function iterate(A::SparseArrayCOO{Tv}, state=((eachindex(A),),1)) where {Tv}
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

## AbstractSparseArray interface

function dropstored!(f::Function, A::SparseArrayCOO)
    ptrs = findall(f, A.vals)
    deleteat!(A.inds, ptrs)
    deleteat!(A.vals, ptrs)
    return A
end

numstored(A::SparseArrayCOO) = length(A.vals)
storedindices(A::SparseArrayCOO) = A.inds
storedvalues(A::SparseArrayCOO) = A.vals
storedpairs(A::SparseArrayCOO) = Iterators.map(Pair, A.inds, A.vals)

## AbstractSparseArray optional interface (internal)

findall_stored(f::Function, A::SparseArrayCOO) =
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

"""
    check_Ti(dims, Ti)

Check that the `dims` tuple and `Ti` index type are valid:
+ `dims` are nonnegative and fit in `Ti` (`0 ≤ dims[k] ≤ typemax(Ti)`)
+ corresponding length fits in `Int` (`prod(dims) ≤ typemax(Int)`)
If not, throw an `ArgumentError`.
"""
function check_Ti(dims::Dims{N}, Ti::Type) where {N}
    # Check that dims are nonnegative and fit in Ti
    maxTi = typemax(Ti)
    for k in 1:N
        dim = dims[k]
        dim >= 0 || throw(ArgumentError("the size along dimension $k (dims[$k] = $dim) is negative"))
        dim <= maxTi ||
            throw(ArgumentError("the size along dimension $k (dims[$k] = $dim) does not fit in Ti = $(Ti) (typemax($Ti) = $(typemax(Ti)))"))
    end

    # Check that corresponding length fits in Int
    len = reduce(widemul, dims)
    len <= typemax(Int) ||
        throw(ArgumentError("number of elements (length = $len) does not fit in Int (prevents linear indexing)"))
    # do not need to check that dims[k] <= typemax(Int) for CartesianIndex since eltype(dims) == Int

    return nothing
end

"""
    checkbounds_dims(Bool, dims, I...)

Return `true` if the specified indices `I` are in bounds for an array
with the given dimensions `dims`. Useful for checking the inputs to constructors.
"""
function checkbounds_dims(::Type{Bool}, dims::Dims{N}, I::Vararg{Integer,N}) where {N}
    for k in 1:N
        (1 <= I[k] <= dims[k]) || return false
    end
    return true
end

"""
    checkbounds_dims(dims, I...)

Throw an error if the specified indices `I` are not in bounds for an array
with the given dimensions `dims`. Useful for checking the inputs to constructors.
"""
function checkbounds_dims(dims::Dims{N}, I::Vararg{Integer,N}) where {N}
    checkbounds_dims(Bool, dims, I...) ||
        throw(ArgumentError("index (= $I) out of bounds (dims = $dims)"))
    return nothing
end
