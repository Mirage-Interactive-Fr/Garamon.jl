using Test
include("n04_short_identity.jl")

@testset "N04 declared coverage" begin
    records=n04_grid()
    @test length(records)==30720
    @test length(unique(r.n for r in records))==12
    @test all(n->n in getproperty.(records,:n),(64,65,128,129))
    @test getproperty.(records,:case_id)==1:30720
end

@testset "N04 exact identity admission and independent oracle" begin
    for n in (2,4,65,129),family in (:euclidean,:signed,:null,:general),
        shape in (:vector,:simple_bivector,:mixed_positive,:negative_scalar_vector)
        state=n04_fixture(n,3,1,family,shape,:stable)
        a=first(state.inputs);cert=n04_certify(state.g,a)
        square=n04_oracle_product(state.g,a,a)
        admitted=all(iszero,keys(square))
        @test (cert!==nothing)==admitted
        @test admitted==(shape!=:negative_scalar_vector)
        @test n04_garamon_power(state.g,a,2,:linear)==square
        for p in (0,1,2,3,16)
            expected=n04_oracle_power(state.g,a,p)
            for strategy in (:linear,:binary,:expression_plan)
                @test n04_garamon_power(state.g,a,p,strategy)==expected
            end
            admitted && (@test n04_power(cert,state.g,a,p)==expected)
        end
    end
end

@testset "N04 nilpotence, refusal, binding and ownership" begin
    g=Matrix{N04Q}(I,4,4)
    nil=N04Terms(big(1)=>1,big(3)=>1) # (e1+e12)^2=1-1=0.
    cert=n04_certify(g,nil)
    @test cert!==nothing && iszero(cert.lambda)
    @test n04_power(cert,g,nil,0)==N04Terms(big(0)=>1)
    @test n04_power(cert,g,nil,1)==nil
    @test isempty(n04_power(cert,g,nil,1024))
    @test_throws ArgumentError n04_inverse(cert,g,nil)
    negative=N04Terms(big(3)=>1,big(12)=>1) # e12+e34 has a nonzero grade-four square.
    @test n04_oracle_product(g,negative,negative)==N04Terms(big(0)=>-2,big(15)=>2)
    @test n04_certify(g,negative)===nothing
    tiny=N04Terms(big(0)=>1//big(2)^100,big(1)=>1)
    @test n04_certify(g,tiny)===nothing # No numerical tolerance may discard its vector square.
    @test_throws ArgumentError n04_certify(Float64.(g),Dict(big(1)=>1.0))
    @test_throws ArgumentError n04_certify(g,Dict(big(16)=>1))
    changed=copy(nil);changed[big(1)]=2
    @test !n04_bound(cert,g,changed)
    @test_throws ArgumentError n04_power(cert,g,changed,2)
    changed_metric=copy(g);changed_metric[1,1]=2
    @test !n04_bound(cert,changed_metric,nil)
    out=n04_power(cert,g,nil,1);out[big(1)]=20
    @test n04_power(cert,g,nil,1)==nil
    nil[big(1)]=2
    @test !n04_bound(cert,g,nil)
    @test_throws ArgumentError n04_power(cert,g,nil,-1)
    zero_a=N04Terms();zero_cert=n04_certify(g,zero_a)
    @test isempty(n04_power(zero_cert,g,zero_a,1))
    @test n04_power(zero_cert,g,zero_a,0)==N04Terms(big(0)=>1)
    scalar_a=N04Terms(big(0)=>3//2);scalar_cert=n04_certify(g,scalar_a)
    @test n04_power(scalar_cert,g,scalar_a,3)==N04Terms(big(0)=>27//8)
end

@testset "N04 polynomial, inverse, and full episode routes" begin
    g=N04Q[2 1//2;1//2 -1]
    a=N04Terms(big(1)=>2,big(2)=>1)
    cert=n04_certify(g,a)
    @test cert!==nothing
    coeff=N04Q[2,-1,3,4]
    expected=N04Terms()
    for (p,c) in enumerate(coeff),(mask,value) in n04_oracle_power(g,a,p-1)
        expected[mask]=get(expected,mask,zero(N04Q))+c*value
    end
    filter!(kv->!iszero(last(kv)),expected)
    @test n04_polynomial(cert,g,a,coeff)==expected
    @test n04_oracle_product(g,a,n04_inverse(cert,g,a))==N04Terms(big(0)=>1)
    @test n04_oracle_product(g,n04_inverse(cert,g,a),a)==N04Terms(big(0)=>1)
    @test isempty(n04_polynomial(cert,g,a,N04Q[]))
    for changes in (:stable,:coefficients),shape in (:vector,:negative_scalar_vector)
        state=n04_fixture(4,16,32,:signed,shape,changes)
        for strategy in N04_ROUTES
            @test n04_qualify(state,n04_episode(state,strategy))
        end
        diagnostic=n04_episode(state,:certify_reuse;trace=true)
        expected_certifications=changes==:stable && shape==:vector ? 1 : 32
        @test diagnostic.certifications==expected_certifications
        @test diagnostic.refusals==(shape==:negative_scalar_vector ? 32 : 0)
    end
end
