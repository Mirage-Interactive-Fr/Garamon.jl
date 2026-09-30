using Test, Random, LinearAlgebra, Garamon

@testset "Exact target-ordered signed disjoint convolution" begin
    rng=MersenneTwister(3901)
    for n in 2:11
        ga=algebra(Matrix{BigInt}(I,n,n))
        masks=sort!(unique!(UInt64[rand(rng,0:(1<<n)-1)
            for _ in 1:min(20,1<<n)]))
        isempty(masks) && push!(masks,UInt64(0))
        a=multivector(ga,Dict(mask=>BigInt(rand(rng,1:17))
            for mask in masks);storage=:sparse)
        b=multivector(ga,Dict(mask=>BigInt(rand(rng,1:17))
            for mask in reverse(masks));storage=:sparse)
        plan=prepare_signed_disjoint_convolution(a,b)
        @test run_signed_disjoint_convolution(plan,a,b)==wedge(a,b)
        @test signed_disjoint_convolution(a,b)==wedge(a,b)
        for _ in 1:3
            changed_a=multivector(ga,Dict(mask=>BigInt(rand(rng,1:17))
                for mask in masks);storage=:sparse)
            changed_b=multivector(ga,Dict(mask=>BigInt(rand(rng,1:17))
                for mask in masks);storage=:sparse)
            @test run_signed_disjoint_convolution(plan,changed_a,changed_b)==
                wedge(changed_a,changed_b)
        end
    end
    ga=algebra(Matrix{BigInt}(I,3,3))
    a=multivector(ga,Dict(UInt64(1)=>BigInt(2));storage=:sparse)
    b=multivector(ga,Dict(UInt64(2)=>BigInt(3));storage=:sparse)
    plan=prepare_signed_disjoint_convolution(a,b)
    @test_throws ArgumentError prepare_signed_disjoint_convolution(a,b;max_paths=26)
    @test_throws ArgumentError prepare_signed_disjoint_convolution(a,b;max_terms=0)
    @test_throws ArgumentError prepare_signed_disjoint_convolution(a,b;max_dimension=2)
    changed=multivector(ga,Dict(UInt64(4)=>BigInt(3));storage=:sparse)
    @test_throws ArgumentError run_signed_disjoint_convolution(plan,a,changed)
    fractional=multivector(ga,Dict(UInt64(2)=>3//2);storage=:sparse)
    @test_throws ArgumentError prepare_signed_disjoint_convolution(a,fractional)
    metric(ga)[1,1]=2
    @test_throws ArgumentError run_signed_disjoint_convolution(plan,a,b)
end

