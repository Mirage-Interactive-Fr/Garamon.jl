using Random

function egraph_word_oracle(factors,diagonal)
    result=Dict{UInt64,BigInt}()
    choices=[collect(f.values) for f in factors]
    for selected in Iterators.product(choices...)
        mask=UInt64(0)
        coefficient=BigInt(1)
        for (right,value) in selected
            inversions=0
            for i in 1:length(diagonal), j in 1:i-1
                !iszero(mask & (UInt64(1)<<(i-1))) &&
                    !iszero(right & (UInt64(1)<<(j-1))) && (inversions+=1)
            end
            isodd(inversions) && (coefficient=-coefficient)
            overlap=mask & right
            for i in 1:length(diagonal)
                !iszero(overlap & (UInt64(1)<<(i-1))) &&
                    (coefficient*=diagonal[i])
            end
            coefficient*=value
            mask=xor(mask,right)
        end
        result[mask]=get(result,mask,BigInt(0))+coefficient
    end
    filter!(pair->!iszero(last(pair)),result)
    result
end

@testset "bounded associative e-graph exact extraction" begin
    rng=MersenneTwister(223)
    for n in (2,4,8,16,32,64), count in (3,4,5)
        diagonal=Int64[mod(i,7)==0 ? 0 : iseven(i) ? -1 : 1 for i in 1:n]
        gram=zeros(Int64,n,n)
        for i in 1:n
            gram[i,i]=diagonal[i]
        end
        ga=algebra(gram)
        factors=[multivector(ga,Dict{UInt64,BigInt}(
            UInt64(0)=>BigInt(rand(rng,(-2,-1,1,2))),
            UInt64(1)<<(rand(rng,1:n)-1)=>BigInt(rand(rng,(-2,-1,1,2))));
            storage=:sparse) for _ in 1:count]
        plan=prepare_product_egraph(factors;max_support=1<<12)
        result=run_product_egraph(plan,factors)
        @test result.values==egraph_word_oracle(factors,diagonal)
        @test result.values==reduce(geometric_product,factors).values
        @test product_egraph_stats(plan).candidates==
              sum((width-1)*(count-width+1) for width in 2:count)
        changed=[multivector(ga,Dict(mask=>BigInt(value*2)
                   for (mask,value) in f.values);storage=:sparse) for f in factors]
        @test run_product_egraph(plan,changed).values==
              egraph_word_oracle(changed,diagonal)
    end
end

@testset "e-graph refuses changed supports and budgets" begin
    ga=algebra(Int64[1 0 0;0 1 0;0 0 1])
    a=multivector(ga,Dict{UInt64,BigInt}(0=>1,1=>2);storage=:sparse)
    b=multivector(ga,Dict{UInt64,BigInt}(0=>1,2=>2);storage=:sparse)
    factors=[a,b,a]
    plan=prepare_product_egraph(factors)
    @test_throws ArgumentError run_product_egraph(plan,[a,
        multivector(ga,Dict{UInt64,BigInt}(0=>1,4=>2);storage=:sparse),a])
    @test_throws ArgumentError prepare_product_egraph(factors;max_candidates=1)
    @test_throws ArgumentError prepare_product_egraph(factors;max_support=1)
    @test_throws ArgumentError prepare_product_egraph(factors;max_terms=0)
    @test_throws ArgumentError prepare_product_egraph([a])
end
