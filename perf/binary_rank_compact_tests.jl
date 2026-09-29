using Test
include("binary_rank_compact_cases.jl")

@testset "K1 compact ambient products, signatures and queries" begin
    for n in COMPACT_SCREEN_DIMS,family in RANK_FAMILIES,signature in RANK_SIGNATURES
        fixture=compact_fixture(n,family,signature)
        a,b=first(fixture.inputs)
        for query in ((:all,0),(:grade,n-min(n,3)+1),(:coefficient,last(fixture.masks)))
            plan=prepare_compact_rank(a,b;query)
            @test length(plan.phi)<=min(big(1)<<fixture.d,length(plan.left_masks)*length(plan.right_masks))
            @test length(unique(plan.output_coordinates))==length(plan.phi)
            @test plan.ambient_grades==count_ones.(plan.phi)
            for (coordinate,mask) in zip(plan.output_coordinates,plan.phi)
                @test mask==foldl(xor,(plan.basis[j] for j in eachindex(plan.basis) if !iszero(coordinate & (1<<(j-1))));init=zero(mask))
            end
            for method in COMPACT_DECISION_METHODS
                artifact=method==:rank_compact ? plan : compact_artifact(fixture,method,query)
                @test compact_correct(fixture,method,artifact,query)
            end
        end
    end
end

@testset "K1 workspace ownership and structural subsets" begin
    for signature in RANK_SIGNATURES,query in ((:all,0),(:grade,2),(:coefficient,0))
        fixture=compact_fixture(8,:low_rank_high_grade,signature)
        a,b=first(fixture.inputs);plan=prepare_compact_rank(a,b;query)
        workspace=CompactRankWorkspace(plan,Float64)
        saved=compact_rank_product!(workspace,a,b);saved_values=copy(saved.values)
        a2,b2=fixture.inputs[2]
        compact_rank_product!(workspace,a2,b2)
        @test saved.values==saved_values
        @test saved.values!==compact_rank_product!(workspace,a,b).values
        subset=multivector(a.algebra,Dict(first(fixture.masks)=>1.);storage=:sparse)
        expected=rank_oracle(fixture,subset,b,query)
        @test compact_rank_product(plan,subset,b;allow_subsets=true).values==expected
        @test compact_rank_product!(workspace,subset,b;allow_subsets=true).values==expected
        @test_throws ArgumentError compact_rank_product!(workspace,subset,b)
        empty=multivector(a.algebra,Dict{UInt64,Float64}();storage=:sparse)
        @test isempty(compact_rank_product!(workspace,empty,b;allow_subsets=true).values)
        @test compact_rank_product!(workspace,a,b).values==rank_oracle(fixture,a,b,query)
    end
    fixture=compact_fixture(128,:higher_rank,:indefinite)
    for method in COMPACT_DECISION_METHODS,horizon in (1,32,1024)
        @test rank_owned_correct(fixture,compact_episode(fixture,method,(:all,0),horizon),(:all,0),horizon)
    end
end

@testset "K1 budgets, rank reduction and rejection contracts" begin
    fixture=compact_fixture(128,:higher_rank,:positive)
    a,b=first(fixture.inputs);plan=prepare_compact_rank(a,b)
    @test length(plan.phi)<(1<<fixture.d)
    @test_throws ArgumentError prepare_compact_rank(a,b;max_bytes=1)
    @test_throws ArgumentError prepare_compact_rank(a,b;max_rank=1)
    @test_throws ArgumentError prepare_compact_rank(a,b;max_pairs=1)
    @test_throws ArgumentError prepare_compact_rank(a,b;max_outputs=1)
    @test_throws ArgumentError CompactRankWorkspace(plan,Float64;max_bytes=1)
    @test_throws ArgumentError CompactRankWorkspace(plan,Float32)
    @test_throws ArgumentError prepare_compact_rank(a,b;query=(:unknown,0))
    workspace=CompactRankWorkspace(plan,Float64)
    outsider=multivector(a.algebra,Dict(UInt128(1)<<127=>1.);storage=:sparse)
    @test_throws ArgumentError compact_rank_product!(workspace,outsider,b;allow_subsets=true)
    a.algebra.metric.diag[1]=-1.
    @test_throws ArgumentError compact_rank_product!(workspace,a,b)
    nonorth=algebra([1. 0.5;0.5 1.]);x=basisvector(nonorth,1;storage=:sparse)
    @test_throws ArgumentError prepare_compact_rank(x,x)
    for n in (64,65,129)
        fixture=compact_fixture(n,:low_rank_high_grade,:degenerate)
        for method in COMPACT_DECISION_METHODS
            @test compact_correct(fixture,method,compact_artifact(fixture,method,(:all,0)),(:all,0))
        end
    end
end

@testset "K1 full replay partition is exhaustive and stable" begin
    seen=Set{String}();indices=Int[];shards=Dict(n=>0 for n in 2:128)
    for n in 2:128,family in RANK_FAMILIES,signature in RANK_SIGNATURES,horizon in RANK_HORIZONS,
        query in COMPACT_FULL_QUERIES,method in COMPACT_DECISION_METHODS
        case=(;n,family,signature,horizon,query,method,passage=1)
        push!(seen,compact_case_id(case));push!(indices,compact_global_index(case));shards[n]+=1
    end
    @test length(seen)==41_148
    @test indices==collect(1:41_148)
    @test all(==(COMPACT_SHARD_SIZE),values(shards))
    @test compact_shard_id(2)=="n002"
    @test compact_shard_id(128)=="n128"
end
