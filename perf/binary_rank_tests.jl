using Test
include("binary_rank_cases.jl")

@testset "N01 ambient cocycle and reproducible fixtures" begin
    for (index,n) in enumerate(RANK_SCREEN_DIMS),family in RANK_FAMILIES
        signature=RANK_SIGNATURES[mod1(index,3)]
        fixture=rank_fixture(n,family,signature)
        a,b=first(fixture.inputs)
        for query in ((:all,0),(:grade,n-min(n,3)+1),(:coefficient,last(fixture.masks)))
            plan=prepare_rank(a,b;query)
            @test length(plan.basis)==fixture.d
            @test length(unique(plan.phi))==1<<fixture.d
            @test plan.ambient_grades==count_ones.(plan.phi)
            for method in RANK_METHODS
                artifact=method==:binary_rank ? plan : rank_artifact(fixture,method,query)
                @test rank_correct(fixture,method,artifact,query)
            end
        end
    end
end

@testset "N01 directly measured episodes own every output" begin
    fixture=rank_fixture(8,:low_rank_high_grade,:indefinite)
    for method in RANK_METHODS,horizon in (1,32,1024)
        result=rank_owned_episode(fixture,method,(:all,0),horizon)
        @test rank_owned_correct(fixture,result,(:all,0),horizon)
        if horizon>1
            saved=copy(result.outputs[2].values)
            result.outputs[1].values[UInt64(0)]=99.
            @test result.outputs[2].values==saved
        end
    end
end

@testset "N01 contracts and counterexamples to Cl(d,0)" begin
    ga=algebra(Diagonal([1.,1.,1.]))
    a=multivector(ga,Dict(UInt64(3)=>1.);storage=:sparse)
    plan=prepare_rank(a,a)
    @test length(plan.basis)==1
    @test plan.ambient_grades==[0,2]
    @test rank_product(plan,a,a).values==Dict(UInt64(0)=>-1.)
    zeroa=multivector(ga,Dict{UInt64,Float64}();storage=:sparse)
    @test isempty(rank_product(plan,zeroa,a;allow_subsets=true).values)
    @test_throws ArgumentError rank_product(plan,zeroa,a)
    @test_throws ArgumentError rank_product(plan,basisvector(ga,1;storage=:sparse),a)
    @test_throws ArgumentError prepare_rank(a,a;max_rank=0)
    @test_throws ArgumentError prepare_rank(a,a;max_bytes=1)
    @test_throws ArgumentError prepare_rank(a,a;max_pairs=0)
    ga.metric.diag[1]=-1.
    @test_throws ArgumentError rank_product(plan,a,a)
    nonorth=algebra([1. 0.5;0.5 1.])
    x=basisvector(nonorth,1;storage=:sparse)
    @test_throws ArgumentError prepare_rank(x,x)
    nullga=algebra(Diagonal([0.,-1.]))
    nullblade=multivector(nullga,Dict(UInt64(3)=>2.);storage=:sparse)
    @test isempty(rank_product(prepare_rank(nullblade,nullblade),nullblade,nullblade).values)
    # Beyond both native mask boundaries is supported without narrowing.
    fixture=rank_fixture(129,:low_rank_high_grade,:indefinite)
    @test rank_correct(fixture,:binary_rank,rank_artifact(fixture,:binary_rank,(:all,0)),(:all,0))
end
