using Test, Garamon, LinearAlgebra

# Independent Clifford sign/metric computation, deliberately not product APIs.
function selector_oracle(a,b,c,requests)
    ga=a.algebra; n=dimension(ga)
    function factor(x,y)
        inversions=sum(((x>>(i-1))&1)*((y>>(j-1))&1) for i in 1:n for j in 1:i-1;init=0)
        value=isodd(inversions) ? -1//1 : 1//1
        for i in 1:n
            !iszero((x&y)&(one(x)<<(i-1))) && (value*=Rational{BigInt}(metric(ga)[i,i]))
        end
        value
    end
    [sum(Rational{BigInt}(av)*Rational{BigInt}(bv)*Rational{BigInt}(cv)*factor(am,bm)*factor(am⊻bm,cm)
         for (am,av) in Garamon._terms(a), (bm,bv) in Garamon._terms(b), (cm,cv) in Garamon._terms(c)
         if am⊻bm⊻cm==Garamon._mask(ga,indices);init=big(0)//1) for indices in requests]
end

@testset "Experimental exact triple selection" begin
    for n in (2,3,65), family in (:euclidean,:signed,:null,:scaled)
        diagonal=ones(n)
        family==:signed && (diagonal[end]=-1)
        family==:null && (diagonal[end]=0)
        family==:scaled && (diagonal[1]=0.5;diagonal[end]=2)
        ga=algebra(Matrix(Diagonal(diagonal)))
        a=scalar(ga,1.;storage=:sparse)+basisvector(ga,1;storage=:sparse)
        b=scalar(ga,2.;storage=:sparse)-basisvector(ga,n;storage=:sparse)
        c=scalar(ga,-1.;storage=:sparse)+basisblade(ga,[1,n];storage=:sparse)
        requests=[Int[],[1],[n],[1,n]]
        expected=Float64.(selector_oracle(a,b,c,requests))
        for strategy in (:full,:recursive,:join3,:prepared,:workspace,:auto)
            selector=prepare_triple_selector(a,b,c,requests;horizon=2,strategy)
            first=run_selected_triple!(selector,a,b,c)
            @test first==expected
            @test first isa Vector{Float64}
            @test run_selected_triple!(selector,2a,b,c)==2expected
            @test first==expected # owned output survives the second evaluation
            @test_throws TripleSelectionRefusal run_selected_triple!(selector,a,b,c)
        end
    end

    ga=algebra(Matrix{Float64}(I,2,2))
    a=scalar(ga,1.;storage=:sparse)+basisvector(ga,1;storage=:sparse)
    minus=scalar(ga,1.;storage=:sparse)-basisvector(ga,1;storage=:sparse)
    requests=[Int[],[1]]
    small_features=triple_strategy_features(a,a,a,requests;horizon=2)
    @test small_features.pair1 isa Int
    big_features=merge(small_features,NamedTuple{(:pair1,:pair2,:ab_bound,:join_work,:recursive_work)}(
        big.((small_features.pair1,small_features.pair2,small_features.ab_bound,
              small_features.join_work,small_features.recursive_work))))
    small_admission=triple_strategy_admission(small_features)
    big_admission=triple_strategy_admission(big_features)
    @test all(small_admission[s].admitted==big_admission[s].admitted &&
              big(small_admission[s].cost_proxy)==big_admission[s].cost_proxy
              for s in keys(small_admission))
    huge_horizon=triple_strategy_features(a,a,a,requests;horizon=typemax(Int))
    @test huge_horizon.pair1 isa BigInt
    @test all(!candidate.admitted for candidate in values(triple_strategy_admission(huge_horizon)))
    for strategy in (:prepared,:workspace)
        selector=prepare_triple_selector(a,minus,a,requests;horizon=3,strategy)
        @test run_selected_triple!(selector,a,minus,a)==[0.,0.]
        @test run_selected_triple!(selector,a,a,a)==[4.,4.]
        @test run_selected_triple!(selector,a,minus,a)==[0.,0.]
        @test selector.second_plan_rebuilds==(strategy==:workspace ? 2 : 0)
    end
    for strategy in (:full,:recursive,:join3,:prepared,:workspace,:auto)
        selector=prepare_triple_selector(a,a,a,requests;strategy)
        @test_throws TripleSelectionRefusal run_selected_triple!(selector,scalar(ga,1.;storage=:sparse),a,a)
        @test selector.remaining_calls==1
        @test_throws TripleSelectionRefusal prepare_triple_selector(a,a,a,requests;
            strategy,budget=TripleSelectorBudget(max_work=1))
        @test_throws TripleSelectionRefusal prepare_triple_selector(a,a,a,requests;
            strategy,budget=TripleSelectorBudget(max_bytes=1))
    end
    @test_throws TripleSelectionRefusal prepare_triple_selector(a,a,a,requests;
        strategy=:prepared,budget=TripleSelectorBudget(max_paths=1))
    @test_throws TripleSelectionRefusal prepare_triple_selector(a,a,a,requests;
        strategy=:recursive,budget=TripleSelectorBudget(max_support=1))
    @test_throws ArgumentError prepare_triple_selector(a,a,a,[Int[],Int[]])
    @test_throws ArgumentError prepare_triple_selector(a,a,a,requests;horizon=0)
    selector=prepare_triple_selector(a,a,a,requests)
    metric(ga)[1,1]=2
    @test_throws TripleSelectionRefusal run_selected_triple!(selector,a,a,a)

    ga=algebra([1. 1.;1. 2.])
    e1=basisvector(ga,1;storage=:sparse);e2=basisvector(ga,2;storage=:sparse)
    selector=prepare_triple_selector(e1,e2,e1,[[1],[2]])
    @test selector.decision.strategy==:full
    @test run_selected_triple!(selector,e1,e2,e1)==[2.,-1.]
    @test_throws TripleSelectionRefusal prepare_triple_selector(e1,e2,e1,[Int[]];strategy=:join3)
    ga=algebra(3,:ega)
    a=multivector(ga,Dict(0=>Inf);storage=:sparse)
    @test_throws TripleSelectionRefusal prepare_triple_selector(a,a,a,[Int[]])
    ga=algebra(129,:ega); a=scalar(ga,1.;storage=:sparse)
    @test_throws TripleSelectionRefusal prepare_triple_selector(a,a,a,[Int[]])
end
