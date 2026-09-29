# N04 research prototype. Exact rational scalar-square certificates only.
# No Float64 recognition, approximate identity, minimal-polynomial search or exp.
push!(LOAD_PATH,dirname(@__DIR__))
using Garamon, LinearAlgebra

const N04Q=Rational{BigInt}
const N04Terms=Dict{BigInt,N04Q}
const N04_ROUTES=(:linear,:binary,:expression_plan,:certify_each,:certify_reuse)
const N04_DIMENSIONS=(2,3,4,5,8,12,16,32,64,65,128,129)

function n04_grid()
    records=NamedTuple[]
    for n in N04_DIMENSIONS,p in (0,1,2,3,16,64,256,1024),H in (1,32),
        family in (:euclidean,:signed,:null,:general),
        shape in (:vector,:simple_bivector,:mixed_positive,:negative_scalar_vector),
        changes in (:stable,:coefficients),strategy in N04_ROUTES
        push!(records,(;case_id=length(records)+1,n,p,H,family,shape,changes,strategy))
    end
    records
end

n04_coefficient_budget(q)=ndigits(abs(numerator(q));base=2)<=4096 && ndigits(denominator(q);base=2)<=4096

function n04_rational(x)
    x isa Union{Integer,Rational} || throw(ArgumentError("N04 requires integer/rational input; floats are not certificates"))
    N04Q(x)
end

function n04_inputs(G,A)
    n=size(G,1)
    size(G)==(n,n) && 1<=n<=129 || throw(ArgumentError("N04 square metric dimension budget 1..129"))
    g=map(n04_rational,G)
    issymmetric(g) || throw(ArgumentError("symmetric metric required"))
    all(n04_coefficient_budget,g) || throw(ArgumentError("N04 metric coefficient bit budget"))
    a=N04Terms()
    for (mask,value) in A
        mask isa Integer && 0<=mask<(big(1)<<n) || throw(ArgumentError("blade mask outside algebra"))
        count_ones(mask)<=4 || throw(ArgumentError("N04 prototype grade budget is four"))
        q=n04_rational(value)
        iszero(q) || (a[BigInt(mask)]=q)
    end
    length(a)<=256 || throw(ArgumentError("N04 input support budget"))
    n04_budget(a)
    g,a
end

function n04_budget(a)
    length(a)<=4096 || throw(ArgumentError("N04 output support budget"))
    all(n04_coefficient_budget,values(a)) ||
        throw(ArgumentError("N04 coefficient bit budget"))
    a
end

n04_terms(a::SparseMultiVector)=n04_budget(N04Terms(BigInt(k)=>N04Q(v) for (k,v) in a.values if !iszero(v)))
n04_owned(a)=N04Terms(k=>v for (k,v) in a if !iszero(v))
n04_signature(g,a)=(Tuple(vec(g)),Tuple(sort!(collect(a);by=first)))

struct N04ScalarSquare
    dimension::Int
    metric_values::Tuple
    terms::Tuple
    lambda::N04Q
end

"""Full exact square is the certificate construction cost. Nothing means refusal."""
function n04_certify(G,A)
    g,a=n04_inputs(G,A)
    ga=algebra(g);mv=multivector(ga,a;storage=:sparse)
    square=n04_terms(geometric_product(mv,mv;max_terms=4096))
    any(!iszero(mask) for mask in keys(square)) && return nothing
    metric_values,terms=n04_signature(g,a)
    N04ScalarSquare(size(g,1),metric_values,terms,get(square,big(0),zero(N04Q)))
end

function n04_bound(cert::N04ScalarSquare,G,A)
    g,a=n04_inputs(G,A)
    size(g,1)==cert.dimension && n04_signature(g,a)==(cert.metric_values,cert.terms)
end

function n04_combine(cert,u,v)
    out=N04Terms()
    iszero(u) || (out[big(0)]=u)
    for (mask,value) in cert.terms
        out[mask]=get(out,mask,zero(N04Q))+v*value
    end
    filter!(kv->!iszero(last(kv)),out)
    n04_budget(out)
end

function n04_power(cert::N04ScalarSquare,G,A,p::Integer)
    0<=p<=4096 || throw(ArgumentError("N04 exponent budget 0..4096"))
    n04_bound(cert,G,A) || throw(ArgumentError("certificate no longer binds metric and coefficients"))
    scale=cert.lambda^div(p,2) # Includes 0^0=1 for p=0/1.
    iseven(p) ? n04_combine(cert,scale,zero(N04Q)) : n04_combine(cert,zero(N04Q),scale)
end

function n04_polynomial(cert::N04ScalarSquare,G,A,coefficients)
    length(coefficients)<=4097 || throw(ArgumentError("N04 polynomial degree budget"))
    n04_bound(cert,G,A) || throw(ArgumentError("certificate no longer binds inputs"))
    u=zero(N04Q);v=zero(N04Q)
    for c in Iterators.reverse(coefficients)
        u,v=v*cert.lambda+n04_rational(c),u
        n04_budget(N04Terms(big(0)=>u,big(1)=>v))
    end
    n04_combine(cert,u,v)
end

function n04_inverse(cert::N04ScalarSquare,G,A)
    n04_bound(cert,G,A) || throw(ArgumentError("certificate no longer binds inputs"))
    iszero(cert.lambda) && throw(ArgumentError("scalar-square inverse requires nonzero lambda"))
    n04_combine(cert,zero(N04Q),inv(cert.lambda))
end

# Independent oracle: antisymmetrized compositions of Chevalley vector action.
# No Garamon product, scalar-square reduction, or diagonal blade-product formula.
function n04_vector_action(g,i,state)
    out=N04Terms();n=size(g,1);bit=big(1)<<(i-1)
    for (mask,c) in state
        indices=[j for j in 1:n if !iszero(mask&(big(1)<<(j-1)))]
        if iszero(mask&bit)
            sign=isodd(count(<(i),indices)) ? -1 : 1
            target=mask|bit;out[target]=get(out,target,zero(N04Q))+sign*c
        end
        for (position,j) in enumerate(indices)
            target=mask⊻(big(1)<<(j-1))
            out[target]=get(out,target,zero(N04Q))+(isodd(position) ? 1 : -1)*g[i,j]*c
        end
    end
    filter!(kv->!iszero(last(kv)),out);n04_budget(out)
end

function n04_permutations(indices)
    isempty(indices) && return [(Int[],1)]
    [(vcat(i,tail), (isodd(position) ? 1 : -1)*sign)
        for (position,i) in enumerate(indices)
        for (tail,sign) in n04_permutations(vcat(indices[1:position-1],indices[position+1:end]))]
end

function n04_oracle_product(G,A,B)
    g,a=n04_inputs(G,A);_,b=n04_inputs(G,B);out=N04Terms();n=size(g,1)
    for (mask,c) in a
        indices=[j for j in 1:n if !iszero(mask&(big(1)<<(j-1)))]
        divisor=factorial(big(length(indices)))
        for (permutation,sign) in n04_permutations(indices)
            state=n04_owned(b)
            for i in Iterators.reverse(permutation);state=n04_vector_action(g,i,state);end
            for (target,value) in state
                out[target]=get(out,target,zero(N04Q))+sign*c*value/divisor
            end
        end
    end
    filter!(kv->!iszero(last(kv)),out);n04_budget(out)
end

function n04_oracle_power(G,A,p)
    0<=p<=4096 || throw(ArgumentError("N04 exponent budget"))
    state=N04Terms(big(0)=>one(N04Q))
    for _ in 1:p;state=n04_oracle_product(G,A,state);end
    state
end

function n04_garamon_power(G,A,p,strategy)
    0<=p<=4096 || throw(ArgumentError("N04 exponent budget"))
    g,a=n04_inputs(G,A);ga=algebra(g);mv=multivector(ga,a;storage=:sparse)
    identity=scalar(ga,one(N04Q);storage=:sparse)
    if strategy==:linear
        out=identity
        for _ in 1:p;out=geometric_product(out,mv;max_terms=4096);n04_terms(out);end
        return n04_terms(out)
    elseif strategy==:binary
        out=identity;base=mv;k=p
        while k>0
            isodd(k) && (out=geometric_product(out,base;max_terms=4096);n04_terms(out))
            k>>=1
            k>0 && (base=geometric_product(base,base;max_terms=4096);n04_terms(base))
        end
        return n04_terms(out)
    elseif strategy==:expression_plan
        memo=Dict(0=>GAExpr(:leaf,(identity,),0),1=>GAExpr(:leaf,(mv,),0))
        function node(k)
            get!(memo,k) do
                iseven(k) ? GAExpr(:*,(node(div(k,2)),node(div(k,2))),0) : GAExpr(:*,(node(k-1),node(1)),0)
            end
        end
        return n04_terms(evaluate(prepare_expression(node(p);max_nodes=256)))
    end
    throw(ArgumentError("unknown N04 reference route"))
end

function n04_fixture(n,p,H,family,shape,changes)
    2<=n<=129 && p in (0,1,2,3,16,64,256,1024) && H in (1,32) || error("outside N04 grid")
    family in (:euclidean,:signed,:null,:general) || error("N04 metric family")
    shape in (:vector,:simple_bivector,:mixed_positive,:negative_scalar_vector) || error("N04 shape")
    changes in (:stable,:coefficients) || error("N04 changes")
    g=Matrix{N04Q}(I,n,n)
    family==:signed && (g[end,end]=-1)
    family==:null && (g[end,end]=0)
    family==:general && (g[1,end]=g[end,1]=1//2)
    firstbit=big(1);lastbit=big(1)<<(n-1)
    inputs=map(1:3) do phase
        c=N04Q(changes==:stable ? 1 : phase)
        shape==:vector ? N04Terms(lastbit=>c) :
        shape==:simple_bivector ? N04Terms(firstbit|lastbit=>c) :
        shape==:mixed_positive ? N04Terms(firstbit=>c,firstbit|lastbit=>1) :
        N04Terms(big(0)=>1,lastbit=>c)
    end
    exact=[n04_oracle_power(g,a,p) for a in inputs]
    state=(;n,p,H,family,shape,changes,g,inputs,exact)
    Base.summarysize(state)<=64<<20 || error("N04 fixture budget")
    state
end

function n04_episode(state,strategy;trace=false)
    strategy in N04_ROUTES || throw(ArgumentError("unknown N04 route"))
    output=Vector{N04Terms}(undef,state.H);cert=nothing;certifications=0;refusals=0;reuse=0
    for t in 1:state.H
        a=state.inputs[1+mod(t-1,3)]
        if strategy in (:certify_each,:certify_reuse)
            if strategy==:certify_each || cert===nothing || !n04_bound(cert,state.g,a)
                cert=n04_certify(state.g,a);certifications+=1
                cert===nothing && (refusals+=1)
            else
                reuse+=1
            end
            output[t]=cert===nothing ? n04_garamon_power(state.g,a,state.p,:binary) : n04_power(cert,state.g,a,state.p)
        else
            output[t]=n04_garamon_power(state.g,a,state.p,strategy)
        end
    end
    trace ? (;output,certifications,refusals,reuse) : output
end

n04_qualify(state,output)=length(output)==state.H && all(output[t]==state.exact[1+mod(t-1,3)] for t in 1:state.H)
