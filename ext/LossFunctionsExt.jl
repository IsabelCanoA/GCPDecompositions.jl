module LossFunctionsExt

using GCPDecompositions, LossFunctions
using IntervalSets

const SupportedLosses = Union{LossFunctions.DistanceLoss,LossFunctions.MarginLoss}

Base.convert(::Type{GCPLosses.AbstractLoss}, loss::SupportedLosses) =
    GCPLosses.Wrapped(loss, LossFunctions)

GCPLosses.value(loss::SupportedLosses, x, m) = loss(m, x)
GCPLosses.deriv(loss::SupportedLosses, x, m) = LossFunctions.deriv(loss, m, x)
GCPLosses.domain(::SupportedLosses)          = Interval(-Inf, Inf)

end
