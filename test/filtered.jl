using Test, Random, Garamon

@testset "Certified checked-integer sign with exact fallback" begin
    rng=MersenneTwister(4501)
    for n in (2,3,4,5,8,16,32,65,129)
        matrix=zeros(Int64,n,n)
        for i in 1:n
            matrix[i,i]=i%5==0 ? 0 : iseven(i) ? -2 : 3
        end
        ga=algebra(matrix)
        K=n<=64 ? UInt64 : n<=128 ? UInt128 : BigInt
        hi=one(K)<<(n-1)
        masks=K[0,one(K),hi,one(K)|hi]
        target=[1,n]
        a=multivector(ga,Dict(mask=>BigInt(rand(rng,1:100))
            for mask in masks);storage=:sparse)
        b=multivector(ga,Dict(mask=>BigInt(rand(rng,1:100))
            for mask in masks);storage=:sparse)
        plan=prepare_filtered_sign(a,b,target)
        for _ in 1:5
            changed_a=multivector(ga,Dict(mask=>BigInt(rand(rng,-100:-1))
                for mask in masks);storage=:sparse)
            changed_b=multivector(ga,Dict(mask=>BigInt(rand(rng,1:100))
                for mask in masks);storage=:sparse)
            expected=sign(coefficient_mask(geometric_product(changed_a,changed_b),
                one(K)|hi))
            result,info=run_filtered_sign(plan,changed_a,changed_b;
                diagnostics=true)
            @test result==expected
            @test !info.used_fallback
            @test filtered_product_sign(changed_a,changed_b,target)==expected
        end
        huge=multivector(ga,Dict(mask=>big(2)^100 for mask in masks);
            storage=:sparse)
        result,info=run_filtered_sign(plan,huge,huge;diagnostics=true)
        @test result==sign(coefficient_mask(geometric_product(huge,huge),
            one(K)|hi))
        @test info.used_fallback
    end
    ga=algebra(Int64[1 0;0 1])
    a=multivector(ga,Dict(UInt64(1)=>BigInt(2));storage=:sparse)
    b=multivector(ga,Dict(UInt64(2)=>BigInt(3));storage=:sparse)
    plan=prepare_filtered_sign(a,b,[1,2])
    @test_throws ArgumentError prepare_filtered_sign(a,b,[1,2];max_paths=0)
    @test_throws BoundsError prepare_filtered_sign(a,b,[3])
    @test_throws ArgumentError run_filtered_sign(plan,a,
        multivector(ga,Dict(UInt64(1)=>BigInt(3));storage=:sparse))
    fractional=multivector(ga,Dict(UInt64(2)=>3//2);storage=:sparse)
    @test_throws ArgumentError prepare_filtered_sign(a,fractional,[1,2])
    metric(ga)[1,1]=2
    @test_throws ArgumentError run_filtered_sign(plan,a,b)
end

