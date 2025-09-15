module LossFunctionsExt

using GCPDecompositions, LossFunctions
using IntervalSets

const SupportedLosses = Union{LossFunctions.DistanceLoss,LossFunctions.MarginLoss}

Base.convert(::Type{GCPLosses.AbstractLoss}, loss::SupportedLosses) =
    GCPLosses.Wrapped(loss, LossFunctions)

GCPLosses.value(loss::GCPLosses.Wrapped{<:SupportedLosses}, x, m) = loss.loss(m, x)
GCPLosses.deriv(loss::GCPLosses.Wrapped{<:SupportedLosses}, x, m) = LossFunctions.deriv(loss.loss, m, x)
GCPLosses.domain(::GCPLosses.Wrapped{<:SupportedLosses})          = Interval(-Inf, Inf)

end
