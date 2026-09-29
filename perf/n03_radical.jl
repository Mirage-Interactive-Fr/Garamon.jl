# N03 experimental kernel. Diagonal metrics only; independent of Garamon src/.
using LinearAlgebra, Random, SHA, TOML
const N03_Q = Rational{BigInt}
const N03_LIMITS = (support=256, active=8, pair_paths=65536, object_bytes=8<<20,
    matrix_entries=65536, rational_bits=4096, horizon=128, rss_bytes=2<<30)
struct N03NonUnit <: Exception end
Base.showerror(io::IO,::N03NonUnit)=print(io,"quotient or element is not invertible")
n03_bit(i)=UInt128(1)<<(i-1)
n03_unit(::Type{T}) where T=Dict(UInt128(0)=>one(T))
function n03_check(a)
    length(a)<=N03_LIMITS.support || throw(ArgumentError("N03 support budget"))
    Base.summarysize(a)<=N03_LIMITS.object_bytes || throw(ArgumentError("N03 object memory budget"))
    for value in values(a)
        isfinite(value) || throw(ArgumentError("nonfinite coefficient"))
        if value isa Rational{BigInt}
            max(ndigits(abs(numerator(value));base=2),ndigits(denominator(value);base=2))<=N03_LIMITS.rational_bits ||
                throw(ArgumentError("N03 rational bit budget"))
        end
    end
    a
end
function n03_validate(g,a)
    g isa AbstractVector || throw(ArgumentError("N03 requires a diagonal metric vector; a zero diagonal is not a radical test for a general matrix"))
    1<=length(g)<=128 || throw(ArgumentError("N03 dimensions 1..128"))
    all(isfinite,g) || throw(ArgumentError("finite diagonal metric required"))
    n03_check(a)
    length(g)<128 && any(mask>>length(g)!=0 for mask in keys(a)) && throw(ArgumentError("mask outside dimension"))
    nothing
end
n03_radical_mask(g)=foldl(|,(n03_bit(i) for i in eachindex(g) if iszero(g[i]));init=UInt128(0))
n03_vector_in_radical(g::AbstractVector,v)=length(g)==length(v) && all(iszero,g.*v)
n03_vector_in_radical(g::AbstractMatrix,v)=size(g,2)==length(v) && all(iszero,g*v)
function n03_product(a::Dict{UInt128,T},b::Dict{UInt128,T},g) where T
    length(a)*length(b)<=N03_LIMITS.pair_paths || throw(ArgumentError("N03 pair-path budget"))
    out=Dict{UInt128,T}()
    for x in sort!(collect(keys(a))),y in sort!(collect(keys(b)))
        coefficient=a[x]*b[y]; common=x&y; bits=common
        while bits!=0
            k=trailing_zeros(bits)+1; coefficient*=g[k]; bits&=bits-1
        end
        iszero(coefficient) && continue
        parity=false; bits=x
        while bits!=0
            low=bits&(-bits); parity ⊻=isodd(count_ones(y&(low-1))); bits&=bits-1
        end
        parity && (coefficient=-coefficient)
        mask=x⊻y; value=get(out,mask,zero(T))+coefficient
        iszero(value) ? delete!(out,mask) : (out[mask]=value)
        length(out)<=N03_LIMITS.support || throw(ArgumentError("N03 intermediate support budget"))
    end
    n03_check(out)
end
function n03_add(a::Dict{UInt128,T},b::Dict{UInt128,T},scale=one(T)) where T
    out=copy(a)
    for (mask,value) in b
        value=get(out,mask,zero(T))+scale*value
        iszero(value) ? delete!(out,mask) : (out[mask]=value)
    end
    n03_check(out)
end
function n03_basis(a)
    used=foldl(|,keys(a);init=UInt128(0))
    indices=[i for i in 1:128 if !iszero(used&n03_bit(i))]
    length(indices)<=N03_LIMITS.active || throw(ArgumentError("N03 active-coordinate budget"))
    [foldl(|,(n03_bit(indices[j]) for j in eachindex(indices) if !iszero(m&(1<<(j-1))));init=UInt128(0)) for m in 0:(1<<length(indices))-1]
end
function n03_regular_inverse(a::Dict{UInt128,T},g) where T
    n03_validate(g,a); basis=n03_basis(a); d=length(basis)
    d*d<=N03_LIMITS.matrix_entries || throw(ArgumentError("N03 regular matrix budget"))
    index=Dict(mask=>i for (i,mask) in enumerate(basis)); matrix=zeros(T,d,d)
    for (column,mask) in enumerate(basis)
        for (out,value) in n03_product(a,Dict(mask=>one(T)),g)
            matrix[index[out],column]=value
        end
    end
    rhs=zeros(T,d);rhs[1]=one(T)
    solution=try matrix\rhs catch error
        error isa SingularException && throw(N03NonUnit())
        rethrow()
    end
    out=Dict(basis[i]=>solution[i] for i in eachindex(basis) if !iszero(solution[i]))
    n03_check(out)
end
function n03_inverse(a::Dict{UInt128,T},g;diagnostics=false) where T
    n03_validate(g,a)
    radical=n03_radical_mask(g); r=count_ones(radical)
    r<=3 || throw(ArgumentError("N03 prototype corank budget 0..3"))
    q=Dict(mask=>value for (mask,value) in a if iszero(mask&radical))
    j=Dict(mask=>value for (mask,value) in a if !iszero(mask&radical))
    # Nonzero scalar part is NOT the criterion: solve the actual quotient map.
    qi=n03_regular_inverse(q,g)
    z=n03_product(qi,j,g); term=n03_unit(T); total=copy(term); degrees=Int[]
    for k in 1:r
        term=n03_product(term,z,g)
        push!(degrees,isempty(term) ? r+1 : minimum(count_ones(mask&radical) for mask in keys(term)))
        total=n03_add(total,term,isodd(k) ? -one(T) : one(T))
        isempty(term) && break
    end
    isempty(n03_product(term,z,g)) || error("nilpotence invariant failed")
    inverse=n03_product(total,qi,g)
    diagnostics ? (;inverse,radical_dimension=r,quotient_support=length(q),
        quotient_matrix_order=length(n03_basis(q)),nilpotent_support=length(z),
        series_terms=length(degrees),minimum_radical_degrees=degrees) : inverse
end

# Independent oracle. A basis blade is a sorted list; apply e_i∧ + contraction
# right-to-left. No production XOR/sign kernel or radical series is used.
function n03_oracle_product(a,b,g)
    out=Dict{UInt128,N03_Q}()
    for (am,av) in a,(bm,bv) in b
        word=[i for i in eachindex(g) if !iszero(bm&n03_bit(i))]
        coefficient=N03_Q(av)*N03_Q(bv)
        left=[i for i in eachindex(g) if !iszero(am&n03_bit(i))]
        for i in reverse(left)
            position=searchsortedfirst(word,i)
            coefficient*=isodd(position-1) ? -1 : 1
            if position<=length(word) && word[position]==i
                coefficient*=N03_Q(g[i]); deleteat!(word,position)
            else
                insert!(word,position,i)
            end
        end
        mask=foldl(|,(n03_bit(i) for i in word);init=UInt128(0))
        value=get(out,mask,zero(N03_Q))+coefficient
        iszero(value) ? delete!(out,mask) : (out[mask]=value)
    end
    n03_check(out)
end
function n03_oracle_inverse(a,g)
    basis=n03_basis(a); d=length(basis); d<=128 || throw(ArgumentError("oracle dimension budget"))
    index=Dict(mask=>i for (i,mask) in enumerate(basis)); matrix=zeros(N03_Q,d,d+1);matrix[1,end]=1
    for (column,mask) in enumerate(basis), (out,value) in n03_oracle_product(a,Dict(mask=>1),g)
        matrix[index[out],column]=value
    end
    # Separate Gauss-Jordan implementation, not the production LinearAlgebra solve.
    for column in 1:d
        pivot=findfirst(row->!iszero(matrix[row,column]),column:d)
        isnothing(pivot) && throw(N03NonUnit())
        row=column-1+pivot
        if row!=column
            for j in 1:d+1;matrix[row,j],matrix[column,j]=matrix[column,j],matrix[row,j];end
        end
        divisor=matrix[column,column]
        for j in column:d+1;matrix[column,j]/=divisor;end
        for row in 1:d
            row==column && continue;coefficient=matrix[row,column]
            iszero(coefficient) && continue
            for j in column:d+1;matrix[row,j]-=coefficient*matrix[column,j];end
        end
    end
    n03_check(Dict(basis[i]=>matrix[i,end] for i in 1:d if !iszero(matrix[i,end])))
end

# Deterministic coordinates and changing coefficients; construction is repeated
# inside each measured episode. Only the three independent oracle answers are cached.
function n03_input(n,r,s,family,phase,::Type{T}=Float64) where T
    n>=r+s && 1<=r<=3 && 1<=s<=3 || throw(ArgumentError("inadmissible active dimensions"))
    family in (:positive,:signed) || throw(ArgumentError("unknown metric family"))
    g=ones(T,n);radicals=collect(n-r+1:n);g[radicals].=zero(T)
    core=s==1 ? [1] : unique(round.(Int,range(1,n-r;length=s)))
    length(core)==s || error("core placement")
    family==:signed && (g[core[min(2,length(core))]]=-one(T))
    a=Dict{UInt128,T}(); qsum=zero(T)
    for (k,i) in enumerate(core)
        value=T(isodd(k+phase) ? 1 : -1)/T(8)
        a[n03_bit(i)]=value;qsum+=abs(value)
    end
    s>=2 && (a[n03_bit(core[1])|n03_bit(core[2])]=T(1)/T(16);qsum+=T(1)/T(16))
    a[UInt128(0)]=T(2)+qsum
    for (k,i) in enumerate(radicals)
        a[n03_bit(i)]=T(k+phase)/T(16)
        a[n03_bit(core[1+mod(k-1,s)])|n03_bit(i)]=T(isodd(k+phase) ? 1 : -1)/T(8)
    end
    r>=2 && (a[n03_bit(radicals[1])|n03_bit(radicals[2])]=T(1)/T(32))
    n03_validate(g,a);(;g,a)
end
function n03_fixture(n,r,s,family,horizon)
    1<=horizon<=N03_LIMITS.horizon || throw(ArgumentError("N03 horizon budget"))
    exact=[begin input=n03_input(n,r,s,family,phase,N03_Q);n03_oracle_inverse(input.a,input.g) end for phase in 1:3]
    (;n,r,s,family,horizon,exact)
end
function n03_episode(state,strategy)
    strategy in (:regular,:radical) || throw(ArgumentError("unknown strategy"))
    outputs=Vector{Dict{UInt128,Float64}}(undef,state.horizon)
    for t in 1:state.horizon
        input=n03_input(state.n,state.r,state.s,state.family,1+mod(t-1,3))
        outputs[t]=strategy==:regular ? n03_regular_inverse(input.a,input.g) : n03_inverse(input.a,input.g)
    end
    Base.summarysize(outputs)<=64<<20 || error("N03 episode output budget")
    outputs
end
function n03_qualify(state,outputs)
    length(outputs)==state.horizon || return false
    for (t,output) in enumerate(outputs)
        exact=state.exact[1+mod(t-1,3)]
        masks=union(keys(output),keys(exact))
        all(mask->isapprox(get(output,mask,0.0),Float64(get(exact,mask,0));atol=1e-10,rtol=1e-9),masks) || return false
    end
    true
end
