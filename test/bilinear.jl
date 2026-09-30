using Random

function bilinear_word_oracle(left,right,diagonal)
    n=length(diagonal)
    result=zeros(BigInt,1<<n)
    for amask in 0:(1<<n)-1,bmask in 0:(1<<n)-1
        a=left[amask+1];b=right[bmask+1]
        iszero(a) || iszero(b) || begin
            value=a*b
            for i in 1:n
                !iszero(amask & (1<<(i-1))) || continue
                isodd(count_ones(bmask & ((1<<(i-1))-1))) &&
                    (value=-value)
                !iszero(bmask & (1<<(i-1))) && (value*=diagonal[i])
            end
            result[xor(amask,bmask)+1]+=value
        end
    end
    result
end

@testset "recursive bilinear synthesis and tensor verified selection" begin
    rng=MersenneTwister(4041)
    for n in (2,4,6,8,10)
        diagonal=Int64[isodd(i) ? 1 : -1 for i in 1:n]
        gram=zeros(Int64,n,n)
        for i in 1:n
            gram[i,i]=diagonal[i]
        end
        ga=algebra(gram)
        left=zeros(BigInt,1<<n);right=zeros(BigInt,1<<n)
        for _ in 1:min(8,1<<n)
            left[rand(rng,eachindex(left))]=BigInt(rand(rng,(-2,-1,1,2)))
            right[rand(rng,eachindex(right))]=BigInt(rand(rng,(-2,-1,1,2)))
        end
        a=DenseMultiVector(ga,left);b=DenseMultiVector(ga,right)
        expected=bilinear_word_oracle(left,right,diagonal)
        for cutoff in (1,2)
            plan=prepare_bilinear_split(a;cutoff)
            @test run_bilinear_split(plan,b).values==expected
            @test bilinear_split_stats(plan).cutoff==cutoff
        end
        fast=prepare_verified_bilinear(a;mul_weight=100,add_weight=1)
        cheap=prepare_verified_bilinear(a;mul_weight=1,add_weight=100)
        @test verified_bilinear_stats(fast).candidate==:strassen
        @test verified_bilinear_stats(cheap).candidate==:classical
        @test verified_bilinear_stats(fast).verified_pairs==32
        @test run_verified_bilinear(fast,b).values==expected
        @test run_verified_bilinear(cheap,b).values==expected
        @test geometric_product(a,b).values==expected
    end
end

@testset "bilinear bounded contract" begin
    ga=algebra(Int64[1 0;0 -1])
    a=DenseMultiVector(ga,BigInt[1,2,3,4])
    @test_throws ArgumentError prepare_bilinear_split(a;cutoff=3)
    @test_throws ArgumentError prepare_bilinear_split(a;max_bytes=1)
    @test_throws ArgumentError prepare_verified_bilinear(a;candidates=())
    @test_throws ArgumentError prepare_verified_bilinear(a;candidates=(:unknown,))
    @test_throws ArgumentError prepare_verified_bilinear(a;mul_weight=0)
    other=algebra(Int64[1 0;0 1])
    b=DenseMultiVector(other,BigInt[1,2,3,4])
    @test_throws ArgumentError run_bilinear_split(prepare_bilinear_split(a),b)
end
