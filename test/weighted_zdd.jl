using Random

# Independent blade-word oracle: lists of basis directions, inversion count,
# and common-direction metric factors. It never traverses ZDD nodes.
function weighted_zdd_word_oracle(a,b,diagonal)
    Q=Rational{BigInt}
    n=length(diagonal)
    output=Dict{BigInt,Q}()
    for (am,av) in a,(bm,bv) in b
        ai=[i for i in 1:n if !iszero(am & (big(1)<<(i-1)))]
        bi=[i for i in 1:n if !iszero(bm & (big(1)<<(i-1)))]
        inversions=sum((i>j for i in ai for j in bi);init=0)
        factor=isodd(inversions) ? -one(Q) : one(Q)
        for i in intersect(ai,bi);factor*=diagonal[i];end
        mask=xor(BigInt(am),BigInt(bm))
        output[mask]=get(output,mask,zero(Q))+factor*av*bv
    end
    filter!(kv->!iszero(last(kv)),output)
end

@testset "coefficient ZDD exact branch product" begin
    Q=Rational{BigInt}
    rng=MersenneTwister(2026092820)
    for n in (2,3,4,5),trial in 1:3
        diagonal=Q[mod(i+n+trial,3)==0 ? 0 : isodd(i) ? 2 : -1 for i in 1:n]
        masks=0:(1<<n)-1
        a=Dict(BigInt(m)=>Q(rand(rng,-3:3)) for m in rand(rng,masks,6))
        b=Dict(BigInt(m)=>Q(rand(rng,-3:3)) for m in rand(rng,masks,6))
        az=weighted_zdd(n,a);bz=weighted_zdd(n,b)
        result=weighted_zdd_product(az,bz,diagonal)
        expected=weighted_zdd_word_oracle(a,b,diagonal)
        @test weighted_zdd_terms(result)==expected
        @test all(weighted_zdd_coefficient(result,m)==get(expected,big(m),zero(Q))
                  for m in masks)
    end
    a=weighted_zdd(2,Dict(1=>Q(1)))
    b=weighted_zdd(2,Dict(1=>Q(-1)))
    @test weighted_zdd_terms(weighted_zdd_product(a,b,Q[0,1]))==Dict{BigInt,Q}()
    @test weighted_zdd_terms(weighted_zdd(2,Dict(1=>Q(0))))==Dict{BigInt,Q}()
    @test_throws ArgumentError weighted_zdd(2,Dict(1=>1.0))
    @test_throws ArgumentError weighted_zdd(2,Dict(4=>1))
    @test_throws ArgumentError weighted_zdd(2,Dict(0=>1,1=>1);max_nodes=1)
    @test_throws ArgumentError weighted_zdd_product(a,b,Q[1,1];max_work=1)
    @test_throws DimensionMismatch weighted_zdd_product(a,weighted_zdd(3,Dict(0=>1)),Q[1,1])
    high=weighted_zdd(65,Dict(big(1)<<64=>Q(2),0=>Q(1)))
    product=weighted_zdd_product(high,high,vcat(fill(Q(1),64),Q[-1]))
    @test weighted_zdd_terms(product)==Dict{BigInt,Q}(0=>-3,(big(1)<<64)=>4)
end
