using Test
include("n05_pfaffian.jl")
const N05_ACCURACY=Dict{String,Any}[]

@testset "N05 scalar Clifford Pfaffian contract" begin
    rng=MersenneTwister(2026092805);Q=Rational{BigInt}
    for n in (2,3,5),k in (0,2,4,6,8),trial in 1:3
        G=Q.(rand(rng,-2:2,n,n));G=G+transpose(G)
        V=Q.(rand(rng,-2:2,n,k))
        expected=get(n05_oracle(G,V),big(0),zero(Q))
        @test n05_scalar(G,V)==expected
        @test n05_scalar(G,reverse(V;dims=2))==expected
        if k>0
            permutation=randperm(rng,k);W=V[:,permutation]
            @test n05_scalar(G,W)==get(n05_oracle(G,W),big(0),zero(Q))
        end
    end
    for family in (:euclidean,:signed,:null,:zero,:general),n in (2,8,65),k in (2,4,8)
        state=n05_fixture(n,k,1,family)
        @test n05_qualify(state,n05_episode(state,:pfaffian))
        for (V,expected) in zip(state.matrices,state.exact)
            @test n05_scalar(Q.(state.G),Q.(V))==expected
        end
    end
    G=Q[1 0;0 1];e1=Q[1,0];e2=Q[0,1]
    @test n05_scalar(G,hcat(e1,e2,e1,e2))==-1
    @test n05_scalar(G,hcat(e1,e1,e2,e2))==1
    @test n05_scalar(G,hcat(e1,e2))==0
    @test n05_oracle(G,hcat(e1,e2))[big(3)]==1 # bivector survives
    @test_throws ArgumentError n05_scalar(G,hcat(e1,e2);output=:bivector)
    @test_throws ArgumentError n05_scalar(G,hcat(e1))
    @test_throws ArgumentError n05_scalar(Q[1 1;0 1],hcat(e1,e2))
    @test_throws ArgumentError n05_scalar(Float64[1 0;0 1],fill(Inf,2,2))
    # First candidate pivot zero, a later nonzero pivot: swap parity matters.
    A=Q[0 0 2 0;0 0 0 3;-2 0 0 0;0 -3 0 0]
    @test n05_pfaffian!(copy(A))==-6
    @test n05_pfaffian!(zeros(Q,4,4))==0
    @test n05_pfaffian!(zeros(Q,0,0))==1
    @test n05_scalar(Q[1 0;0 -1],hcat(e1,e2,e1,e2))==1
    @test n05_scalar(Q[1 0;0 0],hcat(e1,e2,e1,e2))==0
    @test n05_scalar(Q[1 1;1 1],hcat(e1,e2,e1,e2))==1
    # Exact congruence: transformed coordinates and metric preserve all contractions.
    C=Q[1 1;0 1];V=hcat(e1,e2,e1+e2,2*e1-e2)
    @test n05_scalar(transpose(C)*G*C,inv(C)*V)==n05_scalar(G,V)
    for family in (:euclidean,:signed,:null,:zero,:general)
        state=n05_fixture(2,4,1,family)
        @test n05_qualify(state,n05_episode(state,:full))
        family==:general || @test n05_qualify(state,n05_episode(state,:recursive))
    end
    # Near cancellation, exact rational reference and actual floating input oracle.
    for exponent in (10,30,50)
        delta=2.0^(-exponent);Gf=[1. 0.;0. -1.]
        V=[1. 1. 1. 1.;1. 1-delta 1. 1-delta]
        exact=get(n05_oracle(Gf,V),big(0),zero(Q))
        @test n05_scalar(Q.(Gf),Q.(V))==exact
        @test isfinite(n05_scalar(Gf,V))
    end
    # Contraction cancellation itself can erase the answer before elimination.
    for exponent in (30,50)
        delta=2.0^-exponent;Gf=Matrix(Diagonal([1.,1.,-1.]))
        V=repeat([1.,delta,1.],1,4)
        exact=get(n05_oracle(Gf,V),big(0),zero(Q));observed=n05_scalar(Gf,V)
        @test exact==(big(1)//big(2)^exponent)^4
        @test n05_scalar(Q.(Gf),Q.(V))==exact
        @test observed==0 # Float64 loses delta^2 in 1 + delta^2 - 1.
        push!(N05_ACCURACY,Dict("delta_exponent"=>exponent,"exact"=>string(exact),
            "float64"=>observed,"relative_error"=>1.0,"rational_pfaffian_equal"=>true,
            "cause"=>"Gram construction cancellation: 1 + delta^2 - 1"))
    end
end

if length(ARGS)==1
    open(io->TOML.print(io,Dict("ill_conditioned_contractions"=>N05_ACCURACY)),only(ARGS),"w")
elseif !isempty(ARGS)
    error("optional argument: numerical-diagnostics output TOML")
end
