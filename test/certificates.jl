using Test, Random, Garamon

@testset "Propagated exact structural product certificates" begin
    rng=MersenneTwister(4601)
    for n in (2,3,4,5,6,8,16,32,65,129)
        matrix=zeros(Int64,n,n)
        for i in 1:n
            matrix[i,i]=i%5==0 ? 0 : iseven(i) ? -1 : 1
        end
        ga=algebra(matrix)
        K=n<=64 ? UInt64 : n<=128 ? UInt128 : BigInt
        hi=one(K)<<(n-1)
        masks=K[0,one(K),hi]
        make()=multivector(ga,Dict(mask=>BigInt(rand(rng,1:9))
            for mask in masks);storage=:sparse)
        a,b,c=make(),make(),make()
        certificate=prepare_propagated_product(a,b,c)
        for _ in 1:4
            aa,bb,cc=make(),make(),make()
            actual,info=run_propagated_product(certificate,aa,bb,cc;
                diagnostics=true)
            @test actual==geometric_product(geometric_product(aa,bb),cc)
            @test !info.used_fallback
        end
        changed=multivector(ga,Dict(K(0)=>BigInt(1));storage=:sparse)
        @test_throws ArgumentError run_propagated_product(
            certificate,changed,b,c)
        fallback,info=run_propagated_product(certificate,changed,b,c;
            on_invalid=:direct,diagnostics=true)
        @test info.used_fallback
        @test fallback==geometric_product(geometric_product(changed,b),c)
    end
    ga=algebra(Int64[1 0;0 1])
    a=multivector(ga,Dict(UInt64(0)=>BigInt(1),
        UInt64(1)=>BigInt(1));storage=:sparse)
    b=multivector(ga,Dict(UInt64(0)=>BigInt(1),
        UInt64(1)=>BigInt(-1));storage=:sparse)
    c=multivector(ga,Dict(UInt64(0)=>BigInt(1),
        UInt64(2)=>BigInt(2));storage=:sparse)
    certificate=prepare_propagated_product(a,b,c)
    result,info=run_propagated_product(certificate,a,b,c;diagnostics=true)
    @test isempty(result.values)
    @test info.predicted_support>info.actual_support
    @test !info.used_fallback
    changed=multivector(ga,Dict(UInt64(0)=>BigInt(2),
        UInt64(1)=>BigInt(3));storage=:sparse)
    @test run_propagated_product(certificate,changed,b,c)==
        geometric_product(geometric_product(changed,b),c)
    @test_throws ArgumentError prepare_propagated_product(a,b,c;max_paths=1)
    @test_throws ArgumentError run_propagated_product(certificate,a,b,c;
        on_invalid=:unknown)
    floating=multivector(ga,Dict(UInt64(0)=>1.5,
        UInt64(1)=>2.5);storage=:sparse)
    @test_throws ArgumentError run_propagated_product(certificate,floating,b,c;
        on_invalid=:direct)
    big_input=multivector(ga,Dict(UInt64(0)=>typemax(Int64),
        UInt64(1)=>typemax(Int64));storage=:sparse)
    large_certificate=prepare_propagated_product(big_input,big_input,big_input)
    big_result=run_propagated_product(
        large_certificate,big_input,big_input,big_input)
    exact_input=multivector(ga,Dict(UInt64(0)=>BigInt(typemax(Int64)),
        UInt64(1)=>BigInt(typemax(Int64)));storage=:sparse)
    @test big_result==geometric_product(
        geometric_product(exact_input,exact_input),exact_input)
    metric(ga)[1,1]=2
    @test_throws ArgumentError run_propagated_product(certificate,a,b,c)
    @test run_propagated_product(certificate,a,b,c;on_invalid=:direct)==
        geometric_product(geometric_product(a,b),c)
end
