using GCPDecompositions, LinearAlgebra
using GCPDecompositions.TensorKernels

function gmlm(X, Y, r; loss=GCPLosses.LeastSquares())
	# Extract dimensions
	n = only(unique([length(X), length(Y)]))
	M = only(unique(size.(Y)))
	N = only(unique(size.(X)))
	P, Q = length(M), length(N)

	# Initialization
	B0 = CPD(ones(r), rand.((N..., M...), r))
	vu0 = vcat(vec.(B0.U)...)

    # Setup vectorized objective function and gradient
    vec_cutoffs = (0, cumsum(r .* (N..., M...))...)
    vec_ranges = ntuple(k -> vec_cutoffs[k]+1:vec_cutoffs[k+1], Val(P+Q))
    function f(vu)
        VU = map(range -> reshape(view(vu, range), :, r), vec_ranges)
        return gmlm_objective(CPD(ones(r), VU), X, Y, loss)
    end
    function g!(gvu, vu)
        VU = map(range -> reshape(view(vu, range), :, r), vec_ranges)
        GVU = map(range -> reshape(view(gvu, range), :, r), vec_ranges)
        gmlm_grad!(GVU, CPD(ones(r), VU), X, Y, loss)
        return gvu
    end

	# Run LBFGSB
	algorithm = GCPAlgorithms.LBFGSB(iprint=0)
	lbfgsopts = (; (pn => getproperty(algorithm, pn) for pn in propertynames(algorithm))...)
    vu = GCPDecompositions.GCPAlgorithms.lbfgsb(f, g!, vu0; lbfgsopts...)[2]
    VU = map(range -> reshape(vu[range], :, r), vec_ranges)
	return CPD(ones(r), VU)
end

function contract(Xi, B)
	N, Q = size(Xi), ndims(Xi)
	M = size(B)[Q+1:end]
	map(CartesianIndices(M)) do j
		sum(CartesianIndices(N)) do i
			Xi[i]*B[CartesianIndex(i,j)]
		end
	end
end

function gmlm_objective(B, X, Y, loss)
	n = only(unique([length(X), length(Y)]))
	M = only(unique(size.(Y)))
	return sum(
		GCPLosses.value(loss, Y[i][j], contract(X[i], B)[j])
		for j in CartesianIndices(M), i in 1:n
	)
end

function gmlm_grad!(GVU, B, X, Y, loss)
	n = only(unique([length(X), length(Y)]))
	M = only(unique(size.(Y)))
	N = only(unique(size.(X)))
	P, Q = length(M), length(N)

	V, U = collect.(B.U[1:Q]), collect.(B.U[Q+1:end])
	GV, GU = GVU[1:Q], GVU[Q+1:end]

	# Compute the U gradients
	_GU = mapreduce(.+, 1:n) do i
		η = contract(X[i], B)
		Gi = [GCPLosses.deriv(loss, Y[i][j], η[j]) for j in CartesianIndices(M)]
		wi = khatrirao(reverse(V)...)'*vec(X[i])
		mttkrps(Gi, U) .* Ref(Diagonal(wi))
	end
	for k in 1:P
		GU[k] .= _GU[k]
	end

	# Compute the V gradients
	_GV = mapreduce(.+, 1:n) do i
		η = contract(X[i], B)
		Gi = [GCPLosses.deriv(loss, Y[i][j], η[j]) for j in CartesianIndices(M)]
		zi = khatrirao(reverse(U)...)' * vec(Gi)
		mttkrps(X[i], V) .* Ref(Diagonal(zi))
	end
	for k in 1:Q
		GV[k] .= _GV[k]
	end

	return GVU
end