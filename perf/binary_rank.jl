"""Experimental N01: a twisted F₂ support algebra, never Cl(d,0)."""
module BinaryRankPrototype
using Garamon, LinearAlgebra
export RankPlan, prepare_rank, rank_product, select_query, binary_basis

# Each reduced mask has its own coordinate bit, including high ambient grades.
function binary_basis(masks, n; max_rank=16)
    K=eltype(masks); pivots=zeros(K,n); basis=K[]
    for mask in masks
        residual=mask
        for i in n:-1:1
            iszero(residual & (one(K)<<(i-1))) && continue
            if iszero(pivots[i])
                length(basis)<max_rank || throw(ArgumentError("rank budget exceeded"))
                pivots[i]=residual;push!(basis,residual);break
            end
            residual ⊻=pivots[i]
        end
    end
    return basis
end

struct RankPlan{K,T,A,Q}
    algebra::A
    diagonal::Vector{T}
    names::Vector{String}
    algebra_kind::Symbol
    basis::Vector{K}
    phi::Vector{K}
    ambient_grades::Vector{Int}
    left_masks::Vector{K}
    right_masks::Vector{K}
    left_coordinates::Vector{Int}
    right_coordinates::Vector{Int}
    cocycle::Matrix{T}
    selected::BitVector
    query::Q
end

function accept_query(query,mask)
    query[1]==:all && return true
    query[1]==:grade && return count_ones(mask)==query[2]
    query[1]==:coefficient && return mask==query[2]
    throw(ArgumentError("unknown query"))
end

function select_query(mv,query)
    query[1]==:all && return mv
    multivector(mv.algebra,Dict(m=>v for (m,v) in mv.values if accept_query(query,m));storage=:sparse)
end

"""Build φ and the complete diagonal cocycle for all input support pairs.

Table size is 2^d for decoding and |A|×|B| for the cocycle, not 4^d.
The byte estimate is checked before exponential allocation. Input values are
not captured. Output queries refer to ambient grades/masks.
"""
function prepare_rank(a,b;query=(:all,0),max_rank=16,max_pairs=65_536,max_bytes=64<<20)
    a.algebra==b.algebra || throw(ArgumentError("different algebras"))
    ga=a.algebra;n=dimension(ga)
    isdiag(metric(ga)) || throw(ArgumentError("diagonal metric required"))
    eltype(a)<:Real && eltype(b)<:Real || throw(ArgumentError("real coefficients required by this prototype"))
    am=sort!(collect(keys(a.values)));bm=sort!(collect(keys(b.values)))
    K=eltype(am);T=eltype(metric(ga))
    big(length(am))*length(bm)<=max_pairs || throw(ArgumentError("pair budget exceeded"))
    basis=binary_basis(vcat(am,bm),n;max_rank)
    d=length(basis);slots=big(1)<<d
    # BigInt masks require additional storage proportional to ambient width.
    maskbytes=K==BigInt ? 48+cld(n,8) : sizeof(K)
    estimate=slots*(maskbytes+8+1+48)+big(length(am))*length(bm)*sizeof(T)
    estimate<=max_bytes || throw(ArgumentError("rank memory budget exceeded"))
    slots<=typemax(Int) || throw(ArgumentError("coordinate size exceeded"))
    phi=zeros(K,Int(slots))
    for c in 1:length(phi)-1
        bit=trailing_zeros(c)
        phi[c+1]=phi[xor(c,1<<bit)+1] ⊻ basis[bit+1]
    end
    encode=Dict(mask=>c-1 for (c,mask) in enumerate(phi))
    ac=[encode[m] for m in am];bc=[encode[m] for m in bm]
    diagonal=collect(diag(metric(ga)));cocycle=Matrix{T}(undef,length(am),length(bm))
    for i in eachindex(am),j in eachindex(bm)
        x=am[i];y=bm[j];factor=one(T)
        # The cocycle uses ambient intersections and ambient order, so a
        # reduced generator can square to -1 or 0 and need not have grade 1.
        for axis in 1:n
            iszero(x & (one(K)<<(axis-1))) && continue
            isodd(count_ones(y & ((one(K)<<(axis-1))-1))) && (factor=-factor)
            iszero(y & (one(K)<<(axis-1))) || (factor*=diagonal[axis])
        end
        isfinite(factor) || throw(ArgumentError("nonfinite cocycle"))
        cocycle[i,j]=factor
    end
    selected=BitVector(accept_query(query,m) for m in phi)
    RankPlan(ga,diagonal,copy(Garamon.basis(ga)),kind(ga),basis,phi,count_ones.(phi),am,bm,ac,bc,cocycle,selected,query)
end

function rank_product(plan::RankPlan{K,T},a,b;allow_subsets=false) where {K,T}
    for mv in (a,b)
        ga=mv.algebra
        dimension(ga)==length(plan.diagonal) && isdiag(metric(ga)) &&
            basis(ga)==plan.names && kind(ga)==plan.algebra_kind &&
            all(metric(ga)[i,i]==plan.diagonal[i] for i in eachindex(plan.diagonal)) ||
            throw(ArgumentError("algebra changed"))
    end
    for (mv,support) in ((a,plan.left_masks),(b,plan.right_masks))
        all(m in support for m in keys(mv.values)) &&
            (allow_subsets || length(mv.values)==length(support)) ||
            throw(ArgumentError("support changed; rebuild or use a declared subset"))
    end
    S=promote_type(T,eltype(a),eltype(b))
    av=S[get(a.values,m,zero(S)) for m in plan.left_masks]
    bv=S[get(b.values,m,zero(S)) for m in plan.right_masks]
    out=zeros(S,length(plan.phi))
    for j in eachindex(bv),i in eachindex(av)
        c=xor(plan.left_coordinates[i],plan.right_coordinates[j])+1
        plan.selected[c] || continue
        out[c]+=plan.cocycle[i,j]*av[i]*bv[j]
    end
    multivector(a.algebra,Dict{K,S}(plan.phi[i]=>out[i] for i in eachindex(out) if !iszero(out[i]));storage=:sparse)
end
end
