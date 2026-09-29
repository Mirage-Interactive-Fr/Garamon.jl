# N05 research prototype: scalar projection of ordered VECTOR chains only.
# Signed skew elimination, motivated by Wimmer, arXiv:1102.3440v2.
# No sqrt(det), no inverse metric, no approximation of the Clifford product.
push!(LOAD_PATH, dirname(@__DIR__))
using Garamon, LinearAlgebra, Random, SHA, TOML

n05_field(::Type{T}) where {T<:Integer} = Rational{BigInt}
n05_field(::Type{T}) where {T<:Union{AbstractFloat,Rational}} = T

function n05_contractions(G::AbstractMatrix, V::AbstractMatrix)
    n,k=size(V)
    size(G)==(n,n) || throw(DimensionMismatch("metric/vector dimensions"))
    issymmetric(G) || throw(ArgumentError("symmetric bilinear metric required"))
    all(isfinite,G) && all(isfinite,V) || throw(ArgumentError("finite inputs required"))
    k<=64 || throw(ArgumentError("prototype chain budget: 64 vectors"))
    T=n05_field(promote_type(eltype(G),eltype(V)))
    A=zeros(T,k,k)
    # Dense reference construction: O(n^2*k + n*k^2), included in primary timing.
    W=Matrix{T}(G)*Matrix{T}(V)
    for j in 2:k,i in 1:j-1
        value=zero(T)
        for r in 1:n; value+=V[r,i]*W[r,j]; end
        A[i,j]=value; A[j,i]=-value
    end
    A
end

function n05_pfaffian!(A::AbstractMatrix{T}) where T
    n=size(A,1)
    size(A,2)==n || throw(DimensionMismatch("square skew matrix required"))
    iseven(n) || throw(ArgumentError("even matrix order required"))
    all(i->iszero(A[i,i]),1:n) && all(A.==-transpose(A)) ||
        throw(ArgumentError("skew-symmetric matrix required"))
    T<:Union{AbstractFloat,Rational} || throw(ArgumentError("field coefficients required"))
    all(isfinite,A) || throw(ArgumentError("finite contractions required"))
    result=one(T)
    for k in 1:2:n-1
        p=k+argmax(abs.(@view A[k,k+1:n]))
        iszero(A[k,p]) && return zero(T)
        if p!=k+1
            for j in 1:n; A[k+1,j],A[p,j]=A[p,j],A[k+1,j]; end
            for i in 1:n; A[i,k+1],A[i,p]=A[i,p],A[i,k+1]; end
            result=-result
        end
        pivot=A[k,k+1]; result*=pivot
        for j in k+3:n,i in k+2:j-1
            A[i,j]+=(A[k+1,i]*A[k,j]-A[k,i]*A[k+1,j])/pivot
            A[j,i]=-A[i,j]
        end
    end
    isfinite(result) || throw(OverflowError("nonfinite Pfaffian result"))
    result
end

function n05_scalar(G,V;output=:scalar)
    output==:scalar || throw(ArgumentError("N05 admits scalar projection only"))
    iseven(size(V,2)) || throw(ArgumentError("N05 admits even vector chains only"))
    n05_pfaffian!(n05_contractions(G,V))
end

# Independent oracle: Chevalley action v∧ + contraction on an exterior basis,
# applied right-to-left. It computes ALL grades and never uses a Pfaffian,
# pairing enumeration, or Garamon multiplication.
function n05_oracle(G,V)
    n,k=size(V); Q=Rational{BigInt}; g=Q.(G); v=Q.(V)
    state=Dict(big(0)=>one(Q))
    for col in k:-1:1
        next=Dict{BigInt,Q}()
        for (mask,coefficient) in state
            indices=[i for i in 1:n if !iszero(mask&(big(1)<<(i-1)))]
            for i in 1:n
                bit=big(1)<<(i-1)
                if iszero(mask&bit) && !iszero(v[i,col])
                    sign=isodd(count(j->j<i,indices)) ? -1 : 1
                    target=mask|bit
                    next[target]=get(next,target,zero(Q))+sign*v[i,col]*coefficient
                end
            end
            for (position,i) in enumerate(indices)
                contraction=sum((v[j,col]*g[j,i] for j in 1:n);init=zero(Q))
                target=mask⊻(big(1)<<(i-1))
                next[target]=get(next,target,zero(Q))+(isodd(position) ? 1 : -1)*contraction*coefficient
            end
        end
        filter!(x->!iszero(last(x)),next);state=next
        length(state)<=4096 || error("oracle support budget exceeded")
    end
    state
end

function n05_garamon(ga,V,strategy)
    n,k=size(V);k==0 && return one(eltype(V))
    K=n<=64 ? UInt64 : n<=128 ? UInt128 : BigInt
    vectors=[multivector(ga,Dict{K,eltype(V)}(one(K)<<(i-1)=>V[i,j] for i in 1:n if !iszero(V[i,j]));storage=:sparse) for j in 1:k]
    if strategy==:full
        return coefficient(foldl(*,vectors),Int[])
    end
    strategy==:recursive || error("unknown Garamon route")
    node=Garamon.GAExpr(:leaf,(vectors[1],),0)
    for value in vectors[2:end]
        node=Garamon.GAExpr(:*,(node,Garamon.GAExpr(:leaf,(value,),0)),0)
    end
    evaluate(prepare_expression(node);output=Int[],strategy=:recursive,max_support=4096,max_pairs=1<<20)
end

function n05_fixture(n,k,horizon,family;precontracted=false)
    2<=n<=129 && k in (0,2,4,6,8,12,16) && horizon in (1,32,1024) || error("outside N05 protocol")
    G=Matrix{Float64}(I,n,n)
    family==:signed && (G[2:2:n,2:2:n].=-Matrix{Float64}(I,length(2:2:n),length(2:2:n)))
    family==:null && (G[end,end]=0)
    family==:zero && fill!(G,0)
    family==:general && n>=2 && (G[1,end]=G[end,1]=0.5;G[end,end]=-1)
    family in (:euclidean,:signed,:null,:zero,:general) || error("unknown metric")
    directions=unique(round.(Int,range(1,n;length=min(n,4))))
    matrices=map(1:3) do phase
        V=zeros(Float64,n,k)
        for j in 1:k,(slot,i) in enumerate(directions)
            V[i,j]=(-2.,-1.,1.,2.)[1+mod(phase+slot*j+div(j,2),4)]
        end
        V
    end
    exact=[get(n05_oracle(G,V),big(0),big(0)//1) for V in matrices]
    expected=Float64.(exact)
    # Qualification is numerical for pivoted Float64 elimination, exact for Q.
    all(isfinite,expected) || error("oracle exceeds finite Float64 domain")
    contractions=precontracted ? [n05_contractions(G,V) for V in matrices] : nothing
    state=(;n,k,horizon,family,G,ga=algebra(G),matrices,contractions,exact,expected)
    Base.summarysize(state)<=64<<20 || error("N05 fixture budget exceeded")
    state
end

function n05_episode(state,strategy)
    result=Vector{Float64}(undef,state.horizon)
    for t in eachindex(result)
        phase=1+mod(t-1,3);V=state.matrices[phase]
        result[t]=strategy==:pfaffian ? n05_scalar(state.G,V) :
            strategy==:precontracted ? n05_pfaffian!(copy(state.contractions[phase])) :
            n05_garamon(state.ga,V,strategy)
    end
    result
end

function n05_qualify(state,result)
    Sys.maxrss()<=2<<30 || error("N05 worker RSS checkpoint budget exceeded")
    all(t->isapprox(result[t],state.expected[1+mod(t-1,3)];atol=1e-8,rtol=1e-10),eachindex(result))
end
