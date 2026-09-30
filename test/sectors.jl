using Test, Random, LinearAlgebra, Garamon

@testset "Exact grade-involution sectors" begin
    rng=MersenneTwister(3201)
    for n in (2,3,4,5,6,8,16,32,65,129)
        diagonal=[i%5==0 ? 0 : iseven(i) ? -1 : 1 for i in 1:n]
        ga=algebra(Matrix(Diagonal(Int64.(diagonal))))
        K=n<=64 ? UInt64 : n<=128 ? UInt128 : BigInt
        masks=K[0,one(K),one(K)<<(n-1),
            (one(K)<<(n-1))|one(K)]
        make()=multivector(ga,Dict(mask=>BigInt(rand(rng,1:9))*(rand(rng,Bool) ? 1 : -1)
            for mask in masks);storage=:sparse)
        a,b=make(),make()
        for parity in (:all,:even,:odd)
            plan=prepare_invariant_sectors(a,b;output_parity=parity)
            for _ in 1:3
                aa,bb=make(),make()
                actual,info=run_invariant_sectors(plan,aa,bb;diagnostics=true)
                direct=geometric_product(aa,bb)
                wanted=Garamon._sector_projection(direct,parity)
                @test actual==wanted
                @test !info.used_fallback
                @test info.selected_pairs==(parity==:all ? 4 : 2)
            end
        end
        changed=multivector(ga,Dict(K(0)=>BigInt(1));storage=:sparse)
        plan=prepare_invariant_sectors(a,b;output_parity=:even)
        @test_throws ArgumentError run_invariant_sectors(plan,changed,b)
        fallback,info=run_invariant_sectors(plan,changed,b;
            on_invalid=:direct,diagnostics=true)
        @test info.used_fallback
        @test fallback==Garamon._sector_projection(
            geometric_product(changed,b),:even)
    end
    ga=algebra(Int64[1 0;0 1])
    large=multivector(ga,Dict(UInt64(0)=>typemax(Int64),
        UInt64(1)=>typemax(Int64));storage=:sparse)
    for parity in (:all,:even,:odd)
        plan=prepare_invariant_sectors(large,large;output_parity=parity)
        actual=run_invariant_sectors(plan,large,large)
        big=Garamon._certificate_bigint(large)
        @test actual==Garamon._sector_projection(geometric_product(big,big),parity)
    end
    @test_throws ArgumentError prepare_invariant_sectors(large,large;
        output_parity=:invalid)
    @test_throws ArgumentError prepare_invariant_sectors(large,large;max_paths=0)
    floating=multivector(ga,Dict(UInt64(0)=>1.5);storage=:sparse)
    @test_throws ArgumentError prepare_invariant_sectors(floating,large)
end
