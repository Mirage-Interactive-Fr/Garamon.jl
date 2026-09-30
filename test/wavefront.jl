using Random

function wavefront_word_oracle(factors,target,diagonal)
    K=typeof(target)
    result=Int64(0)
    choices=[collect(factor.values) for factor in factors]
    for picked in Iterators.product(choices...)
        mask=zero(K)
        value=Int64(1)
        for term in picked
            right=first(term)
            left_indices=[i for i in eachindex(diagonal)
                          if !iszero(mask & (one(K)<<(i-1)))]
            right_indices=[i for i in eachindex(diagonal)
                           if !iszero(right & (one(K)<<(i-1)))]
            inversions=sum((i>j for i in left_indices for j in right_indices);init=0)
            factor=isodd(inversions) ? Int64(-1) : Int64(1)
            for i in intersect(left_indices,right_indices)
                factor*=diagonal[i]
            end
            value*=factor*last(term)
            mask=xor(mask,right)
        end
        mask==target && (result+=value)
    end
    result
end

@testset "wavefront exact suffix certificate across dimensions" begin
    rng=MersenneTwister(42)
    for n in (2,6,16,65,129)
        diagonal=Int64[mod(i,5)==0 ? 0 : iseven(i) ? -1 : 1 for i in 1:n]
        gram=zeros(Int64,n,n)
        for i in 1:n
            gram[i,i]=diagonal[i]
        end
        ga=algebra(gram)
        K=n<=64 ? UInt64 : n<=128 ? UInt128 : BigInt
        for _ in 1:8
            factors=typeof(multivector(ga,Dict{K,Int64}();storage=:sparse))[]
            for _ in 1:3
                terms=Dict{K,Int64}(zero(K)=>rand(rng,(-2,-1,1,2)))
                for _ in 1:3
                    positions=rand(rng,1:n,rand(rng,1:2))
                    mask=foldl(|,(one(K)<<(i-1) for i in positions);init=zero(K))
                    terms[mask]=rand(rng,(-2,-1,1,2))
                end
                push!(factors,multivector(ga,terms;storage=:sparse))
            end
            masks=[first(first(factor.values)) for factor in factors]
            target=foldl(xor,masks;init=zero(K))
            plan=prepare_wavefront(factors,target)
            expected=wavefront_word_oracle(factors,target,diagonal)
            @test run_wavefront(plan,factors)==expected
            @test coefficient_mask(reduce(geometric_product,factors),target)==expected
            @test wavefront_stats(plan).max_suffix_width<=1<<14
            changed=[multivector(ga,Dict(mask=>value*2 for (mask,value) in f.values);
                                 storage=:sparse) for f in factors]
            @test run_wavefront(plan,changed)==
                wavefront_word_oracle(changed,target,diagonal)
        end
    end
end

@testset "wavefront refuses lost certificates and resource overflow" begin
    ga=algebra(4,:ega)
    K=UInt64
    a=multivector(ga,Dict{K,Int64}(0=>1,1=>2,2=>3);storage=:sparse)
    b=multivector(ga,Dict{K,Int64}(0=>1,4=>2);storage=:sparse)
    factors=[a,b,a]
    plan=prepare_wavefront(factors,K(0))
    @test run_wavefront(plan,factors)==coefficient_mask(reduce(geometric_product,factors),0)
    changed=copy(factors)
    changed[2]=multivector(ga,Dict{K,Int64}(0=>1,8=>2);storage=:sparse)
    @test_throws ArgumentError run_wavefront(plan,changed)
    @test_throws ArgumentError prepare_wavefront(factors,K(0);max_suffix=1)
    @test_throws ArgumentError run_wavefront(
        prepare_wavefront(factors,K(0);max_pairs=1),factors)
    @test_throws ArgumentError prepare_wavefront(factors,K(16))
    other=algebra([1 1; 1 1])
    c=multivector(other,Dict{UInt64,Int64}(0=>1);storage=:sparse)
    @test_throws ArgumentError prepare_wavefront([c],UInt64(0))
end
