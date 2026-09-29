using Garamon, LinearAlgebra
include("binary_rank.jl")
using .BinaryRankPrototype

const RANK_SCREEN_DIMS=(4,6,8,12,16,24,32,48,64,65,96,128)
const RANK_FAMILIES=(:low_rank_low_grade,:low_rank_high_grade,:higher_rank)
const RANK_SIGNATURES=(:positive,:indefinite,:degenerate)
const RANK_METHODS=(:sparse,:prepared,:binary_rank)
const RANK_HORIZONS=(1,32,1024)
const RANK_COEFFICIENTS=(-2,-1,1,2)

function rank_fixture(n,family,signature)
    signature in (:positive,:indefinite,:mixed,:degenerate) ||
        throw(ArgumentError("unknown rank fixture signature"))
    K=n<=64 ? UInt64 : n<=128 ? UInt128 : BigInt
    d=family==:higher_rank ? min(n,12) : min(n,3)
    ambient_full=(big(1)<<n)-1
    generators=K[]
    for i in 1:d
        # Prefix directions occur only in the first generator. Its unique high
        # bit guarantees independence and gives grade n-d+1 (often near n).
        mask=family==:low_rank_high_grade && i==1 ?
            K(ambient_full ⊻ ((big(1)<<(d-1))-1)) :
            one(K)<<(family==:low_rank_high_grade ? i-2 : i-1)
        push!(generators,mask)
    end
    coords=family==:higher_rank ? unique(vcat(0,[1<<(i-1) for i in 1:d],[(1<<i)-1 for i in 2:d])) : collect(0:(1<<d)-1)
    masks=[foldl(xor,(generators[j] for j in 1:d if !iszero(c & (1<<(j-1))));init=zero(K)) for c in coords]
    diagonal=ones(Int64,n)
    signature in (:indefinite,:mixed) && (diagonal[2:2:n].=-1)
    signature==:degenerate && (diagonal[end]=0)
    ga=algebra(Diagonal(Float64.(diagonal)))
    inputs=[(multivector(ga,Dict(m=>Float64(RANK_COEFFICIENTS[mod1(i+step,4)]) for (i,m) in enumerate(masks));storage=:sparse),
             multivector(ga,Dict(m=>Float64(RANK_COEFFICIENTS[mod1(2i+step,4)]) for (i,m) in enumerate(reverse(masks)));storage=:sparse)) for step in 1:4]
    (;n,family,signature,d,diagonal,masks,inputs)
end

# Independent Int64 oracle: insertion into an ordered ambient basis word;
# every exchange negates the factor, equal indices contract by the metric.
# No coordinate map, XOR output, popcount parity or Garamon product is used.
function rank_oracle(fixture,a,b,query)
    K=eltype(fixture.masks);result=Dict{K,Int64}()
    for (am,av) in a.values,(bm,bv) in b.values
        word=[i for i in 1:fixture.n if !iszero(am & (one(K)<<(i-1)))]
        factor=Int64(1)
        for j in 1:fixture.n
            iszero(bm & (one(K)<<(j-1))) && continue
            p=length(word)+1
            while p>1 && word[p-1]>j
                factor=-factor;p-=1
            end
            if p>1 && word[p-1]==j
                factor*=fixture.diagonal[j];deleteat!(word,p-1)
            else
                insert!(word,p,j)
            end
        end
        m=foldl(|,(one(K)<<(i-1) for i in word);init=zero(K))
        selected=query[1]==:all || (query[1]==:grade && length(word)==query[2]) || (query[1]==:coefficient && m==query[2])
        selected || continue
        value=Base.checked_mul(Base.checked_mul(factor,Int64(av)),Int64(bv))
        result[m]=Base.checked_add(get(result,m,Int64(0)),value)
    end
    filter!(p->!iszero(last(p)),result)
end

function rank_artifact(fixture,method,query)
    a,b=first(fixture.inputs)
    method==:binary_rank ? prepare_rank(a,b;query) : method==:prepared ? prepare_product(a,b) : nothing
end

function rank_execute(fixture,method,artifact,query,horizon)
    out=first(fixture.inputs)[1]
    checksum=0.0
    for i in 1:horizon
        a,b=fixture.inputs[mod1(i,4)]
        out=method==:binary_rank ? rank_product(artifact,a,b) :
            select_query(method==:prepared ? run_product(artifact,a,b) : a*b,query)
        # Consume each owned output; includes the same output traversal in all arms.
        checksum+=sum(values(out.values);init=0.0)
    end
    (;out,checksum)
end

# Each observation owns every output until the observation returns. This path
# deliberately differs from the first screening, which retained only the last.
function rank_owned_execute(fixture,method,artifact,query,horizon)
    outputs=Vector{typeof(first(fixture.inputs)[1])}(undef,horizon)
    checksum=0.0
    for i in 1:horizon
        a,b=fixture.inputs[mod1(i,4)]
        out=method==:binary_rank ? rank_product(artifact,a,b) :
            select_query(method==:prepared ? run_product(artifact,a,b) : a*b,query)
        outputs[i]=out
        checksum+=sum(values(out.values);init=0.0)
    end
    (;outputs,checksum)
end

# Artifact construction is inside the observed call, for every observation.
function rank_owned_episode(fixture,method,query,horizon)
    artifact=rank_artifact(fixture,method,query)
    rank_owned_execute(fixture,method,artifact,query,horizon)
end

function rank_owned_correct(fixture,result,query,horizon)
    length(result.outputs)==horizon || return false
    length(Set(objectid(out.values) for out in result.outputs))==horizon || return false
    expected=[rank_oracle(fixture,a,b,query) for (a,b) in fixture.inputs]
    checksum=0.0
    for i in 1:horizon
        oracle=expected[mod1(i,4)]
        result.outputs[i].values==oracle || return false
        checksum+=sum(values(oracle);init=Int64(0))
    end
    result.checksum==checksum
end

function rank_correct(fixture,method,artifact,query)
    for (a,b) in fixture.inputs
        expected=rank_oracle(fixture,a,b,query)
        actual=method==:binary_rank ? rank_product(artifact,a,b) : select_query(method==:prepared ? run_product(artifact,a,b) : a*b,query)
        actual.values==expected || return false
    end
    true
end
