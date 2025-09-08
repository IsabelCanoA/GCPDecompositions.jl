## Sparse array type

@testitem "constructor" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3,
        Ti in [Int, UInt8],
        Tv in [Float64, BigFloat, Int8]

        dims = (5, 3, 2)[1:N]
        inds = (Ti[2, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        inds = tuple.(inds...)
        vals = Tv[1, 0, 10]

        # SparseArrayCOO(dims, inds, vals)
        A = SparseArrayCOO(dims, inds, vals)
        @test typeof(A) === SparseArrayCOO{Tv,Ti,N}
        @test A.dims === dims
        @test A.inds == inds && A.vals == vals

        # check_Ti(dims, Ti)
        @test_throws ArgumentError SparseArrayCOO{Tv,Ti,N}((-1, 3, 2)[1:N], inds, vals)
        if Ti !== Int
            @test_throws ArgumentError SparseArrayCOO{Tv,Ti,N}(
                (Int(typemax(Ti)) + 1, 3, 2)[1:N],
                inds,
                vals,
            )
        end
        if N >= 2
            @test_throws ArgumentError SparseArrayCOO{Tv,Ti,N}(
                (typemax(Int) ÷ 2, 3, 2)[1:N],
                inds,
                vals,
            )
        end

        # check_coo_buffers(inds, vals)
        @test_throws ArgumentError SparseArrayCOO{Tv,Ti,N}(dims, inds, vals[1:end-1])

        # check_coo_inds(dims, inds) - index in bounds
        if Ti <: Signed
            badinds = (Ti[2, -1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
            badinds = tuple.(badinds...)
            @test_throws ArgumentError SparseArrayCOO{Tv,Ti,N}(dims, badinds, vals)
        end
        badinds = (Ti[2, 1, 6], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        badinds = badinds = tuple.(badinds...)
        @test_throws ArgumentError SparseArrayCOO{Tv,Ti,N}(dims, badinds, vals)
    end
end

@testitem "undef constructors" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3,
        Ti in [Int, UInt8],
        Tv in [Float64, BigFloat, Int8]

        dims = (5, 3, 2)[1:N]

        # SparseArrayCOO{Tv,Ti,N}(undef, dims)
        A = SparseArrayCOO{Tv,Ti,N}(undef, dims)
        @test typeof(A) === SparseArrayCOO{Tv,Ti,N}
        @test A.dims === dims
        @test A.inds == Vector{NTuple{N,Ti}}()
        @test A.vals == Vector{Tv}()

        # SparseArrayCOO{Tv,Ti}(undef, dims)
        A = SparseArrayCOO{Tv,Ti}(undef, dims)
        @test typeof(A) === SparseArrayCOO{Tv,Ti,N}
        @test A.dims === dims
        @test A.inds == Vector{NTuple{N,Ti}}()
        @test A.vals == Vector{Tv}()
    end
end

@testitem "AbstractArray constructor" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3,
        Ti in [Int, UInt8],
        Tv in [Float64, BigFloat, Int8]

        dims = (5, 3, 2)[1:N]
        inds = (Ti[2, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        inds = sort(tuple.(inds...); by = CartesianIndex)
        vals = Tv[1, 100, 10]

        # SparseArrayCOO(Ti, A::Array)
        A = zeros(Tv, dims)
        A[CartesianIndex.(inds)] = vals
        C = SparseArrayCOO(Ti, A)
        @test typeof(C) === SparseArrayCOO{Tv,Ti,N}
        @test C.dims === dims
        @test C.inds == inds && C.vals == vals
    end
end

## Minimal AbstractArray interface

@testitem "size" begin
    @testset "N=$N" for N in 1:3
        dims = (5, 3, 2)[1:N]
        inds = ([2, 1, 4], [1, 3, 2], [1, 2, 1])[1:N]
        vals = [1, 0, 10]
        A = SparseArrayCOO(dims, tuple.(inds...), vals)

        @test size(A) == dims
        for k in 1:N
            @test size(A, k) == dims[k]
        end
        @test size(A, N + 1) == 1
    end
end

@testitem "getindex" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3,
        Ti in [Int, UInt8],
        Tv in [Float64, BigFloat, Int8]

        dims = (6, 3, 2)[1:N]
        inds = (Ti[5, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        vals = Tv[1, 0, 10]
        A = SparseArrayCOO(dims, tuple.(inds...), vals)

        # in bounds
        ind_stored    = (4, 2, 1)[1:N]
        ind_notstored = (3, 2, 2)[1:N]
        @test typeof(A[ind_stored...]) === Tv && A[ind_stored...] == Tv(10)
        @test typeof(A[ind_notstored...]) === Tv && A[ind_notstored...] == zero(Tv)

        # out of bounds
        ind_out1 = (0, 1, 1)[1:N]
        ind_out2 = (7, 3, 2)[1:N]
        @test_throws BoundsError A[ind_out1...]
        @test_throws BoundsError A[ind_out2...]

        # duplicate indices
        A = SparseArrayCOO(dims, [tuple.(inds...); tuple.(inds...)], [vals; vals])
        for (ind, val) in zip(tuple.(inds...), vals)
            @test A[ind...] == Tv(2 * val)
        end
    end
end

@testitem "setindex!" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3,
        Ti in [Int, UInt8],
        Tv in [Float64, BigFloat, Int8]

        dims = (6, 3, 2)[1:N]
        inds = (Ti[5, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        inds = tuple.(inds...)
        vals = Tv[1, 0, 10]

        for val in [-4, 0]
            # store new value
            ind = (3, 2, 2)[1:N]
            A = SparseArrayCOO(dims, copy(inds), copy(vals))
            A[ind...] = val
            @test typeof(A) === SparseArrayCOO{Tv,Ti,N}
            @test A.dims === dims
            if iszero(val)
                @test A.inds == inds && A.vals == vals
            else
                @test A.inds == [inds; [ind]] && A.vals == [vals; [val]]
            end

            # overwrite existing value
            ind = (1, 3, 2)[1:N]
            A = SparseArrayCOO(dims, copy(inds), copy(vals))
            A[ind...] = val
            @test typeof(A) === SparseArrayCOO{Tv,Ti,N}
            @test A.dims === dims
            @test A.inds == inds && A.vals == [vals[1], val, vals[3]]
        end

        # out of bounds
        ind_out1 = (0, 1, 1)[1:N]
        ind_out2 = (7, 3, 2)[1:N]
        A = SparseArrayCOO(dims, inds, vals)
        @test_throws BoundsError A[ind_out1...] = 0
        @test_throws BoundsError A[ind_out2...] = 0
    end

    # properly handle error during value conversion
    dims = (5, 3, 2)
    inds = tuple.([2, 1, 4], [1, 3, 2], [1, 2, 1])
    vals = [1, 0, 10]
    A = SparseArrayCOO(dims, copy(inds), copy(vals))
    @test_throws InexactError A[1, 1, 1] = 1.2
    @test A.inds !== inds && A.vals !== vals
    @test A.inds == inds
    @test A.vals == vals
end

@testitem "IndexStyle" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3,
        Ti in [Int, UInt8],
        Tv in [Float64, BigFloat, Int8]

        @test IndexStyle(SparseArrayCOO{Tv,Ti,N}) === IndexCartesian()
    end
end

## Overloads for specializing outputs

@testitem "similar" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3,
        Ti in [Int, UInt8],
        Tv in [Float64, BigFloat, Int8]

        dims = (5, 3, 2)[1:N]
        inds = (Ti[2, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        vals = Tv[1, 0, 10]
        A = SparseArrayCOO(dims, tuple.(inds...), vals)

        # similar(A)
        S = similar(A)
        @test typeof(S) === SparseArrayCOO{Tv,Ti,N}
        @test S.dims === dims
        @test isempty(S.inds) && isempty(S.vals)

        # similar(A, ::Type{S})
        for TvNew in [UInt8]
            S = similar(A, TvNew)
            @test typeof(S) === SparseArrayCOO{TvNew,Ti,N}
            @test S.dims === dims
            @test isempty(S.inds) && isempty(S.vals)
        end

        # similar(A, dims::Dims)
        for dimsNew in [(2,), (2, 4), (2, 4, 3)]
            S = similar(A, dimsNew)
            @test typeof(S) === SparseArrayCOO{Tv,Ti,length(dimsNew)}
            @test S.dims === dimsNew
            @test isempty(S.inds) && isempty(S.vals)
        end

        # similar(A, ::Type{S}, dims::Dims)
        for TvNew in [UInt8], dimsNew in [(2,), (2, 4), (2, 4, 3)]
            S = similar(A, TvNew, dimsNew)
            @test typeof(S) === SparseArrayCOO{TvNew,Ti,length(dimsNew)}
            @test S.dims === dimsNew
            @test isempty(S.inds) && isempty(S.vals)
        end
    end
end

@testitem "show(io, A)" begin
    @testset "nstored=$nstored, N=$N, Ti=$Ti, Tv=$Tv" for nstored in 0:3,
        N in 1:3,
        Ti in [Int, UInt8],
        Tv in [Float64, BigFloat, UInt8]

        dims = (10, 3, 2)[1:N]
        inds = (Ti[2, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        inds = tuple.(inds...)
        vals = Tv[0, 1, 10]

        # Take subset of entries
        inds = inds[1:nstored]
        vals = vals[1:nstored]

        # SparseArrayCOO
        C = SparseArrayCOO(dims, inds, vals)
        @test sprint(show, C; context = :module => @__MODULE__) ==
              "SparseArrayCOO{$Tv, $Ti, $N}($dims, $inds, $vals)"
    end
end

@testitem "show(io, ::MIME\"text/plain\", A)" begin
    @testset "nstored=$nstored, N=$N, Ti=$Ti, Tv=$Tv" for nstored in 0:3,
        N in 1:3,
        Ti in [Int, UInt8],
        Tv in [Float64, BigFloat, UInt8]

        dims = (10, 3, 2)[1:N]
        inds = (Ti[2, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        inds = tuple.(inds...)
        vals = Tv[0, 1, 10]

        # Take subset of entries
        inds = inds[1:nstored]
        vals = vals[1:nstored]

        # SparseArrayCOO
        perm = sortperm(inds; by = CartesianIndex)
        sinds, svals = inds[perm], vals[perm]
        C = SparseArrayCOO(dims, inds, vals)
        entrystrs = map(sinds, svals) do ind, val
            indstr = join(lpad.(Int.(ind), ndigits.(dims)), ", ")
            return "  [$indstr]  =  $val"
        end
        showstr = length(vals) == 0 ? summary(C) : "$(summary(C)):\n$(join(entrystrs,'\n'))"
        @test sprint(show, MIME("text/plain"), C) == showstr
    end

    @testset "displayheight=$displayheight" for displayheight in 0:11
        iocontext =
            IOContext(IOBuffer(), :displaysize => (displayheight, 80), :limit => true)
        dims = (5, 10, 2)
        inds =
            tuple.(
                UInt8[4, 1, 2, 5, 5, 3, 2],
                UInt8[1, 2, 3, 4, 10, 7, 6],
                UInt8[1, 1, 2, 1, 2, 1, 2],
            )
        vals = Float32[0.0, 0.5, 0.4, 0.2, 0.3, 0.8, 0.9]

        # SparseArrayCOO
        showstr = Dict(
            0 => "5×10×2 SparseArrayCOO{Float32, UInt8, 3} with 7 stored entries: \u2026",
            1 => "5×10×2 SparseArrayCOO{Float32, UInt8, 3} with 7 stored entries: \u2026",
            2 => "5×10×2 SparseArrayCOO{Float32, UInt8, 3} with 7 stored entries: \u2026",
            3 => "5×10×2 SparseArrayCOO{Float32, UInt8, 3} with 7 stored entries: \u2026",
            4 => "5×10×2 SparseArrayCOO{Float32, UInt8, 3} with 7 stored entries: \u2026",
            5 => """
            5×10×2 SparseArrayCOO{Float32, UInt8, 3} with 7 stored entries:
             \u22ee""",
            6 => """
            5×10×2 SparseArrayCOO{Float32, UInt8, 3} with 7 stored entries:
              [4,  1, 1]  =  0.0
                          \u22ee""",
            7 => """
            5×10×2 SparseArrayCOO{Float32, UInt8, 3} with 7 stored entries:
              [4,  1, 1]  =  0.0
                          \u22ee
              [5, 10, 2]  =  0.3""",
            8 => """
            5×10×2 SparseArrayCOO{Float32, UInt8, 3} with 7 stored entries:
              [4,  1, 1]  =  0.0
              [1,  2, 1]  =  0.5
                          \u22ee
              [5, 10, 2]  =  0.3""",
            9 => """
            5×10×2 SparseArrayCOO{Float32, UInt8, 3} with 7 stored entries:
              [4,  1, 1]  =  0.0
              [1,  2, 1]  =  0.5
                          \u22ee
              [2,  6, 2]  =  0.9
              [5, 10, 2]  =  0.3""",
            10 => """
            5×10×2 SparseArrayCOO{Float32, UInt8, 3} with 7 stored entries:
              [4,  1, 1]  =  0.0
              [1,  2, 1]  =  0.5
              [5,  4, 1]  =  0.2
                          \u22ee
              [2,  6, 2]  =  0.9
              [5, 10, 2]  =  0.3""",
            11 => """
            5×10×2 SparseArrayCOO{Float32, UInt8, 3} with 7 stored entries:
              [4,  1, 1]  =  0.0
              [1,  2, 1]  =  0.5
              [5,  4, 1]  =  0.2
              [3,  7, 1]  =  0.8
              [2,  3, 2]  =  0.4
              [2,  6, 2]  =  0.9
              [5, 10, 2]  =  0.3""",
        )[displayheight]
        C = SparseArrayCOO(dims, inds, vals)
        @test sprint(
            show,
            MIME("text/plain"),
            C;
            context = IOContext(iocontext, :module => @__MODULE__),
        ) == showstr
    end
end

@testitem "summary(io, A)" begin
    @testset "nstored=$nstored, N=$N, Ti=$Ti, Tv=$Tv" for nstored in 0:3,
        N in 1:3,
        Ti in [Int, UInt8],
        Tv in [Float64, BigFloat, UInt8]

        dims = (10, 3, 2)[1:N]
        inds = (Ti[2, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        inds = tuple.(inds...)
        vals = Tv[0, 1, 10]

        # Take subset of entries
        inds = inds[1:nstored]
        vals = vals[1:nstored]

        # References
        dimstr = ["10-element", "10×3", "10×3×2"]
        valstr = "with $(length(vals)) stored " * (length(vals) == 1 ? "entry" : "entries")

        # SparseArrayCOO
        C = SparseArrayCOO(dims, inds, vals)
        @test sprint(summary, C; context = :module => @__MODULE__) ==
              "$(dimstr[N]) SparseArrayCOO{$Tv, $Ti, $N} $valstr"
    end
end
