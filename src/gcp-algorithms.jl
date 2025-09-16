## GCP Algorithms

"""
Algorithms for Generalized CP Decomposition.
"""
module GCPAlgorithms

using ..GCPDecompositions
using ..GCPLosses: value, deriv, domain
using ..GCPConstraints: project!
using ..TensorKernels: create_mttkrp_buffer, mttkrp!, mttkrps!
using ..TensorKernels: khatrirao!, khatrirao
using IntervalSets: Interval
using LinearAlgebra: lu!, mul!, norm, rdiv!, rmul!, Diagonal
using LBFGSB: lbfgsb

# Abstract type and associated functions

"""
    AbstractAlgorithm

Abstract type for GCP algorithms.

Concrete types `ConcreteAlgorithm <: AbstractAlgorithm` should implement
`_gcp!(M, X, loss, constraints, algorithm::ConcreteAlgorithm)`
that modifies the initialization `M` and returns the modified version.
"""
abstract type AbstractAlgorithm end

"""
    _gcp!(M, X, loss, constraints, algorithm)

Internal function to compute an approximate rank-`r` CP decomposition
of the data tensor `X` with respect to the loss function `loss` and the
constraints `constraints` using the algorithm `algorithm`, modifying the
initialization `M` and returning the modified version.
"""
function _gcp! end

# Built-in algorithms

include("gcp-algorithms/lbfgsb.jl")
include("gcp-algorithms/als.jl")
include("gcp-algorithms/fastals.jl")

# Objective function

"""
    gcp_objective(M::CPD, X::AbstractArray, loss)

Compute the GCP objective function for the model tensor `M`, data tensor `X`,
and loss function `loss`.
"""
function gcp_objective(M::CPD{T,N}, X::Array{TX,N}, loss) where {T,TX,N}
    return sum(value(loss, X[I], M[I]) for I in CartesianIndices(X) if !ismissing(X[I]))
end

# Gradient function

"""
    gcp_grad_U!(GU, M::CPD, X::AbstractArray, loss)

Compute the GCP gradient with respect to the factor matrices `U = (U[1],...,U[N])`
for the model tensor `M`, data tensor `X`, and loss function `loss`, and store
the result in `GU = (GU[1],...,GU[N])`.
"""
function gcp_grad_U!(
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
