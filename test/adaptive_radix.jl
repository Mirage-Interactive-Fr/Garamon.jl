using Random

@testset "adaptive radix sparse wedge preserves every exact path" begin
    rng=MersenneTwister(43)
    for n in (2,6,16,65,129)
        ga=algebra(n,:ega)
        K=n<=64 ? UInt64 : n<=128 ? UInt128 : BigInt
        function fixture(count)
            terms=Dict{K,Float64}()
            for _ in 1:count
                positions=rand(rng,1:n,rand(rng,0:3))
                mask=foldl(|,(one(K)<<(i-1) for i in positions);init=zero(K))
                terms[mask]=Float64(rand(rng,(-3,-2,-1,1,2,3)))
            end
            multivector(ga,terms;storage=:sparse)
        end
        for _ in 1:12
            left,right=fixture(12),fixture(12)
            index=prepare_adaptive_radix(right)
            @test radix_wedge(left,index)==wedge(left,right)
            @test radix_stats(index).leaves==length(right.values)
        end
    end
end

@testset "radix tiers, compression, snapshot, and explicit budgets" begin
    ga=algebra(8,:ega)
    left=multivector(ga,Dict(UInt64(0)=>1.0);storage=:sparse)
    for (support,tier) in ((4,:capacity4),(5,:capacity16),
                           (17,:capacity48),(49,:capacity256))
        right=multivector(ga,Dict(UInt64(i)=>1.0 for i in 0:support-1);
            storage=:sparse)
        index=prepare_adaptive_radix(right)
        @test getproperty(radix_stats(index),tier)==1
        @test radix_wedge(left,index)==right
        @test_throws ArgumentError prepare_adaptive_radix(right;
            max_nodes=2support-2)
    end
    high=algebra(129,:ega)
    a=multivector(high,Dict(BigInt(1)=>2.0);storage=:sparse)
    b=multivector(high,Dict((big(1)<<120)=>3.0,
        (big(1)<<121)=>-2.0);storage=:sparse)
    index=prepare_adaptive_radix(b)
    @test radix_stats(index).prefix_bytes>0
    @test radix_wedge(a,index)==wedge(a,b)
    snapshot=copy(b)
    set_coefficient!(b,[123],5.0)
    @test radix_wedge(a,index)==wedge(a,snapshot)
    @test radix_wedge(a,prepare_adaptive_radix(b))==wedge(a,b)
    @test_throws ArgumentError radix_wedge(a,index;max_terms=1)
    other_metric=zeros(Float64,129,129)
    for i in 1:129
        other_metric[i,i]=-1
    end
    other=algebra(other_metric)
    foreign=multivector(other,Dict(BigInt(1)=>1.0);storage=:sparse)
    @test_throws ArgumentError radix_wedge(foreign,index)
    empty=multivector(high,Dict{BigInt,Float64}();storage=:sparse)
    @test radix_stats(prepare_adaptive_radix(empty)).leaves==0
    @test iszero(radix_wedge(a,prepare_adaptive_radix(empty)))
end
