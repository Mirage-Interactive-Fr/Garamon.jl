using Test
include("binary_rank_compact_cases.jl")
include("bounds_b4.jl")
using .BoundsB4Prototype

@testset "B4 execution environment" begin
    @test VERSION.major==1 && VERSION.minor==13
    @test Threads.nthreads()==1
    @test Base.JLOptions().check_bounds in (0,1)
end

@testset "B4 compact checked and annotated kernels preserve all selected coefficients" begin
    for n in (2,8,65,128),family in (:low_rank_high_grade,:higher_rank),
        signature in RANK_SIGNATURES
        fixture=compact_fixture(n,family,signature)
        for query in ((:all,0),(:grade,max(0,n-1)),(:coefficient,last(fixture.masks)))
            a,b=first(fixture.inputs)
            workspace=b4_workspace(a,b;query)
            @test size(workspace.inner.plan.output_index)==
                (length(workspace.inner.plan.left_masks),length(workspace.inner.plan.right_masks))
            checked=[b4_product!(workspace,pair...;variant=:checked) for pair in fixture.inputs]
            annotated=[b4_product!(workspace,pair...;variant=:inbounds) for pair in fixture.inputs]
            @test length(unique(objectid(output.values) for output in checked))==4
            @test length(unique(objectid(output.values) for output in annotated))==4
            for i in eachindex(fixture.inputs)
                expected=rank_oracle(fixture,fixture.inputs[i]...,query)
                @test checked[i].values==expected
                @test annotated[i].values==expected
            end
            saved=copy(annotated[end].values)
            empty!(annotated[1].values)
            @test annotated[end].values==saved
        end
    end
end

@testset "B4 memory and mutation contracts" begin
    fixture=compact_fixture(8,:low_rank_high_grade,:indefinite)
    a,b=first(fixture.inputs)
    @test_throws ArgumentError b4_workspace(a,b;max_bytes=1)
    workspace=b4_workspace(a,b)
    subset=multivector(a.algebra,Dict(first(fixture.masks)=>1.0);storage=:sparse)
    @test_throws ArgumentError b4_product!(workspace,subset,b;variant=:inbounds)
    @test b4_product!(workspace,subset,b;variant=:inbounds,allow_subsets=true).values==
        rank_oracle(fixture,subset,b,(:all,0))
    @test_throws ArgumentError b4_product!(workspace,a,b;variant=:unknown)
    original=workspace.inner.plan.output_index[1,1]
    workspace.inner.plan.output_index[1,1]=length(workspace.inner.plan.phi)+1
    @test_throws ArgumentError BoundsB4Prototype._check_b4_structure(workspace.inner.plan,workspace.inner)
    workspace.inner.plan.output_index[1,1]=original
    @test b4_product!(workspace,a,b;variant=:inbounds).values==
        rank_oracle(fixture,a,b,(:all,0))
    a.algebra.metric.diag[1]=-1.0
    @test_throws ArgumentError b4_product!(workspace,a,b;variant=:inbounds)
end
