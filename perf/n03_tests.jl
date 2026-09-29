using Test
include("n03_radical.jl")
const N03_TEST_START=time()
@testset "N03 rational quotient and radical inversion" begin
    Q=N03_Q
    for r in 1:3, family in (:positive,:signed), phase in 1:2
        n=2+r; input=n03_input(n,r,2,family,phase,Q)
        actual=n03_inverse(input.a,input.g;diagnostics=true)
        expected=n03_oracle_inverse(input.a,input.g)
        @test actual.inverse==expected
        @test n03_oracle_product(input.a,actual.inverse,input.g)==n03_unit(Q)
        @test n03_oracle_product(actual.inverse,input.a,input.g)==n03_unit(Q)
        @test actual.radical_dimension==r
        @test all(degree>=k for (k,degree) in enumerate(actual.minimum_radical_degrees))
    end
    # The full r+1 ideal bound is attained by these even mixed terms.
    for r in 1:3
        g=vcat(ones(Q,r),zeros(Q,r));j=Dict(n03_bit(i)|n03_bit(r+i)=>one(Q) for i in 1:r)
        power=n03_unit(Q)
        for k in 1:r
            power=n03_product(power,j,g)
            @test !isempty(power)
        end
        @test isempty(n03_product(power,j,g))
        a=n03_add(n03_unit(Q),j);inverse=n03_inverse(a,g)
        @test n03_oracle_product(a,inverse,g)==n03_unit(Q)
        @test n03_oracle_product(inverse,a,g)==n03_unit(Q)
    end
    # Quotient invertibility, not merely a nonzero scalar coefficient.
    g=Q[1,0];singular=Dict(UInt128(0)=>one(Q),n03_bit(1)=>one(Q),n03_bit(2)=>one(Q))
    @test_throws N03NonUnit n03_inverse(singular,g)
    @test_throws N03NonUnit n03_oracle_inverse(singular,g)
    @test_throws N03NonUnit n03_inverse(Dict(n03_bit(2)=>one(Q)),g)
    scalar_zero=Dict(n03_bit(1)=>one(Q),n03_bit(2)=>one(Q))
    @test n03_inverse(scalar_zero,g)==n03_oracle_inverse(scalar_zero,g)
    # Nonradical isotropic vector in a nondegenerate signature (1,1).
    g=Q[1,-1];v=Q[1,1];mv=Dict(n03_bit(1)=>one(Q),n03_bit(2)=>one(Q))
    @test sum(g.*v.^2)==0
    @test !n03_vector_in_radical(g,v)
    @test n03_radical_mask(g)==0
    @test isempty(n03_oracle_product(mv,mv,g))
    @test_throws N03NonUnit n03_inverse(mv,g)
    @test n03_inverse(n03_add(n03_unit(Q),mv),g)==n03_add(n03_unit(Q),mv,-one(Q))
    # In a null-coordinate hyperbolic basis, BOTH diagonal entries vanish but
    # the metric is nonsingular: interpreting them as radical would be wrong.
    G=Q[0 1;1 0]
    @test dot(Q[1,0],G*Q[1,0])==0
    @test !n03_vector_in_radical(G,Q[1,0])
    @test_throws ArgumentError n03_inverse(n03_unit(Q),G)
    # Independent product oracle across signatures and sparse masks.
    rng=MersenneTwister(2026092803)
    for r in 1:3,trial in 1:5
        g=vcat(Q[1,-1],zeros(Q,r));maximum_mask=(1<<length(g))-1
        a=Dict(UInt128(rand(rng,0:maximum_mask))=>Q(rand(rng,-2:2)) for _ in 1:6)
        b=Dict(UInt128(rand(rng,0:maximum_mask))=>Q(rand(rng,-2:2)) for _ in 1:6)
        @test n03_product(a,b,g)==n03_oracle_product(a,b,g)
    end
    for n in (2,4,65,128)
        r=min(3,n-1);s=min(2,n-r);state=n03_fixture(n,r,s,:positive,2)
        @test n03_qualify(state,n03_episode(state,:radical))
        @test n03_qualify(state,n03_episode(state,:regular))
    end
    @test_throws ArgumentError n03_inverse(n03_unit(Q),zeros(Q,4))
    @test_throws ArgumentError n03_fixture(5,3,2,:positive,129)
    @test_throws ArgumentError n03_regular_inverse(Dict(n03_bit(i)=>one(Q) for i in 1:9),ones(Q,9))
end
println("N03 exact tests finished in ",round(time()-N03_TEST_START;digits=3)," s; no performance claim")
