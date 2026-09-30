using Test, Random, Garamon

@testset "Certified multimodular integer product" begin
    rng=MersenneTwister(4401)
    for n in (2,3,5,8,65,129)
        metric_matrix=zeros(Int64,n,n)
        for i in 1:n
            metric_matrix[i,i]=i%5==0 ? 0 : iseven(i) ? -2 : 3
        end
        ga=algebra(metric_matrix)
        K=n<=64 ? UInt64 : n<=128 ? UInt128 : BigInt
        masks=K[0,one(K),one(K) << (n-1)]
        n>=3 && push!(masks,(one(K) << (n-1)) | one(K))
        for bits in (17,23,31)
            left=multivector(ga,Dict(mask=>BigInt(rand(rng,1:1000))
                for mask in masks);storage=:sparse)
            right=multivector(ga,Dict(mask=>BigInt(rand(rng,1:1000))
                for mask in masks);storage=:sparse)
            plan=prepare_modular_product(left,right;prime_bits=bits,max_primes=8)
            product,info=run_modular_product(plan,left,right;diagnostics=true)
            @test product==geometric_product(left,right)
            @test !info.used_fallback
            @test 1<=info.prime_count<=8
            @test modular_geometric_product(left,right;prime_bits=bits)==product
            for _ in 1:8
                changed_left=multivector(ga,Dict(mask=>BigInt(rand(rng,1:1000))
                    for mask in masks);storage=:sparse)
                changed_right=multivector(ga,Dict(mask=>BigInt(rand(rng,1:1000))
                    for mask in masks);storage=:sparse)
                @test run_modular_product(plan,changed_left,changed_right)==
                    geometric_product(changed_left,changed_right)
            end
        end
    end

    ga=algebra(Int64[2 0;0 -3])
    huge=multivector(ga,Dict(UInt64(0)=>big(2)^120,
        UInt64(1)=>big(2)^121);storage=:sparse)
    other=multivector(ga,Dict(UInt64(0)=>big(7),
        UInt64(2)=>big(5));storage=:sparse)
    plan=prepare_modular_product(huge,other;prime_bits=17,max_primes=1)
    fallback,info=run_modular_product(plan,huge,other;diagnostics=true)
    @test fallback==geometric_product(huge,other)
    @test info.used_fallback && info.prime_count==0
    @test_throws ArgumentError run_modular_product(plan,huge,other;fallback=false)
    @test_throws ArgumentError prepare_modular_product(huge,other;max_paths=1)
    @test_throws ArgumentError prepare_modular_product(huge,other;max_terms=1)
    @test_throws ArgumentError prepare_modular_product(huge,other;prime_bits=16)
    @test_throws ArgumentError prepare_modular_product(huge,other;max_primes=65)
    different=basisblade(ga,[1,2];coefficient=big(1),storage=:sparse)
    @test_throws ArgumentError run_modular_product(plan,different,other)
    metric(ga)[1,1]=4
    @test_throws ArgumentError run_modular_product(plan,huge,other)
    noninteger=multivector(ga,Dict(UInt64(0)=>1.5);storage=:sparse)
    @test_throws ArgumentError prepare_modular_product(noninteger,other)
    nondiagonal=algebra(Int64[1 1;1 1])
    @test_throws ArgumentError prepare_modular_product(
        basisvector(nondiagonal,1;storage=:sparse),
        basisvector(nondiagonal,2;storage=:sparse))
end
