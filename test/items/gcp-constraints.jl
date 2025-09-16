## Constraint types

@testitem "constraint constructors" begin
    # LowerBound constraint
    @test GCPConstraints.LowerBound(0) isa GCPConstraints.LowerBound
    @test GCPConstraints.LowerBound(-Inf) isa GCPConstraints.LowerBound
end

@testitem "satisfies/project! methods" begin
    using InteractiveUtils: subtypes
    using .GCPConstraints: satisfies, project!, AbstractConstraint
    @testset "type=$type" for type in subtypes(AbstractConstraint)
        @test hasmethod(satisfies, Tuple{CPD,type})
        @test hasmethod(project!, Tuple{CPD,type})
    end

    # LowerBound constraint
    M = CPD(ones(2), ([-ones(3, 1) ones(3, 1)], ones(4, 2), ones(5, 2)))
    @test !satisfies(M, GCPConstraints.LowerBound(0))
    project!(M, GCPConstraints.LowerBound(0))
    @test satisfies(M, GCPConstraints.LowerBound(0))
end
