## Algorithm types

"""
Algorithms for Generalized CP Decomposition.
"""
module GCPAlgorithms

using ..GCPDecompositions
using ..GCPLosses: value, deriv, domain
using ..TensorKernels: create_mttkrp_buffer, mttkrp!, mttkrps!
using ..TensorKernels: khatrirao!, khatrirao
using IntervalSets: Interval
using LinearAlgebra: lu!, mul!, norm, rdiv!, rmul!, Diagonal
using LBFGSB: lbfgsb

# Algorithms

"""
    AbstractAlgorithm

Abstract type for GCP algorithms.

Concrete types `ConcreteAlgorithm <: AbstractAlgorithm` should implement
`_gcp(X, r, loss, constraints, algorithm::ConcreteAlgorithm)`
that returns a `CPD`.
"""
abstract type AbstractAlgorithm end

"""
    _gcp(X, r, loss, constraints, algorithm)

Internal function to compute an approximate rank-`r` CP decomposition
of the tensor `X` with respect to the loss function `loss` and the
constraints `constraints` using the algorithm `algorithm`, returning
a `CPD` object.
"""
function _gcp end

include("gcp-algorithms/lbfgsb.jl")
include("gcp-algorithms/als.jl")
include("gcp-algorithms/fastals.jl")

# Objective function and gradients

"""
    objective(M::CPD, X::AbstractArray, loss)

Compute the GCP objective function for the model tensor `M`, data tensor `X`,
and loss function `loss`.
"""
function objective(M::CPD{T,N}, X::Array{TX,N}, loss) where {T,TX,N}
    return sum(value(loss, X[I], M[I]) for I in CartesianIndices(X) if !ismissing(X[I]))
end

"""
    grad_U!(GU, M::CPD, X::AbstractArray, loss)

Compute the GCP gradient with respect to the factor matrices `U = (U[1],...,U[N])`
for the model tensor `M`, data tensor `X`, and loss function `loss`, and store
the result in `GU = (GU[1],...,GU[N])`.
"""
function grad_U!(
    GU::NTuple{N,TGU},
    M::CPD{T,N},
    X::Array{TX,N},
    loss,
) where {T,TX,N,TGU<:AbstractMatrix{T}}
    Y = [
        ismissing(X[I]) ? zero(nonmissingtype(eltype(X))) : deriv(loss, X[I], M[I]) for
        I in CartesianIndices(X)
    ]
    mttkrps!(GU, Y, M.U)
    for k in 1:N
        rmul!(GU[k], Diagonal(M.λ))
    end
    return GU
end

end
