"""Experimental K1: reachable XOR outputs, with optional reusable buffers."""
module BinaryRankCompactPrototype
using Garamon, LinearAlgebra
using ..BinaryRankPrototype: binary_basis, accept_query
export CompactRankPlan, CompactRankWorkspace, prepare_compact_rank,
       compact_rank_product, compact_rank_product!

struct CompactRankPlan{K,T,A,Q}
    algebra::A
    diagonal::Vector{T}
    names::Vector{String}
    algebra_kind::Symbol
    basis::Vector{K}
    left_masks::Vector{K}
    right_masks::Vector{K}
    left_coordinates::Vector{Int}
    right_coordinates::Vector{Int}
    output_coordinates::Vector{Int}
    phi::Vector{K}
    ambient_grades::Vector{Int}
    output_index::Matrix{Int}
    cocycle::Matrix{T}
    query::Q
    estimated_bytes::Int
end

function encode_support(masks,basis,n)
    K=eltype(masks)
    pivots=zeros(K,n);bits=zeros(Int,n)
    for (j,mask) in enumerate(basis)
        axis=findlast(i->!iszero(mask & (one(K)<<(i-1))),1:n)
        pivots[axis]=mask;bits[axis]=1<<(j-1)
    end
    map(masks) do mask
        residual=mask;coordinate=0
        for axis in n:-1:1
            iszero(residual & (one(K)<<(axis-1))) && continue
            iszero(bits[axis]) && error("support is outside its computed F2 span")
            residual ⊻=pivots[axis];coordinate ⊻=bits[axis]
        end
        coordinate
    end
end

"""Construct only nonzero, query-admissible reachable outputs.

`phi[i]` is the ambient image of `output_coordinates[i]`; no length-2^d
array is allocated. The full ambient sign and metric factor is retained for
every input pair, and `output_index[i,j]` preindexes its reachable output.
"""
function prepare_compact_rank(a,b;query=(:all,0),max_rank=16,max_pairs=65_536,
                              max_outputs=65_536,max_bytes=64<<20)
    a.algebra==b.algebra || throw(ArgumentError("different algebras"))
    ga=a.algebra;n=dimension(ga);T=eltype(metric(ga))
    isdiag(metric(ga)) || throw(ArgumentError("diagonal metric required"))
    isbitstype(T) && T<:Real || throw(ArgumentError("fixed-size real metric required for bounded prototype"))
    0<=max_rank<8sizeof(Int)-1 || throw(ArgumentError("invalid coordinate rank budget"))
    max_pairs>=0 && max_outputs>=0 && max_bytes>=0 || throw(ArgumentError("negative budget"))
    am=sort!(collect(keys(a.values)));bm=sort!(collect(keys(b.values)));K=eltype(am)
    pairs=big(length(am))*length(bm)
    pairs<=max_pairs || throw(ArgumentError("pair budget exceeded"))
    basis=binary_basis(vcat(am,bm),n;max_rank);d=length(basis)
    reach_bound=min(big(1)<<d,pairs)
    maskbytes=K==BigInt ? 48+cld(n,8) : sizeof(K)
    # Conservative structure/lookup estimate, checked before pair allocation.
    estimate=512+n*(maskbytes+sizeof(T)+32)+d*maskbytes+
        (length(am)+length(bm))*(maskbytes+48)+pairs*(sizeof(Int)+sizeof(T))+
        reach_bound*(maskbytes+2sizeof(Int)+80)
    estimate<=max_bytes || throw(ArgumentError("compact plan memory estimate exceeded"))
    ac=encode_support(am,basis,n);bc=encode_support(bm,basis,n)
    diagonal=collect(diag(metric(ga)));indices=zeros(Int,length(am),length(bm))
    factors=Matrix{T}(undef,length(am),length(bm))
    coordinates=Int[];phi=K[];grades=Int[];lookup=Dict{Int,Int}()
    for j in eachindex(bm),i in eachindex(am)
        x=am[i];y=bm[j];factor=one(T)
        for axis in 1:n
            iszero(x & (one(K)<<(axis-1))) && continue
            isodd(count_ones(y & ((one(K)<<(axis-1))-1))) && (factor=-factor)
            iszero(y & (one(K)<<(axis-1))) || (factor*=diagonal[axis])
        end
        isfinite(factor) || throw(ArgumentError("nonfinite cocycle"))
        factors[i,j]=factor
        mask=xor(x,y)
        # Grade selection is always performed in the ambient space.
        admissible=accept_query(query,mask)
        (iszero(factor) || !admissible) && continue
        coordinate=xor(ac[i],bc[j])
        oi=get(lookup,coordinate,0)
        if iszero(oi)
            length(phi)<max_outputs || throw(ArgumentError("reachable output budget exceeded"))
            push!(coordinates,coordinate);push!(phi,mask);push!(grades,count_ones(mask))
            oi=length(phi);lookup[coordinate]=oi
        else
            phi[oi]==mask || error("noninjective ambient map")
        end
        indices[i,j]=oi
    end
    plan=CompactRankPlan(ga,diagonal,copy(Garamon.basis(ga)),kind(ga),basis,am,bm,ac,bc,
                         coordinates,phi,grades,indices,factors,query,Int(estimate))
    Base.summarysize(plan)<=max_bytes || throw(ArgumentError("actual compact plan size exceeded"))
    plan
end

struct CompactRankWorkspace{S,P}
    plan::P
    left_values::Vector{S}
    right_values::Vector{S}
    output_values::Vector{S}
end

function CompactRankWorkspace(plan,::Type{S}=Float64;max_bytes=64<<20) where S
    isbitstype(S) && S<:Real || throw(ArgumentError("fixed-size real workspace coefficients required"))
    promote_type(S,eltype(plan.diagonal))==S || throw(ArgumentError("workspace coefficient type loses metric precision"))
    bytes=Base.summarysize(plan)+256+big(sizeof(S))*(length(plan.left_masks)+length(plan.right_masks)+length(plan.phi))
    bytes<=max_bytes || throw(ArgumentError("compact workspace memory budget exceeded"))
    CompactRankWorkspace(plan,zeros(S,length(plan.left_masks)),zeros(S,length(plan.right_masks)),zeros(S,length(plan.phi)))
end

function validate_compact(plan,a,b;allow_subsets=false)
    for (mv,support) in ((a,plan.left_masks),(b,plan.right_masks))
        ga=mv.algebra
        dimension(ga)==length(plan.diagonal) && isdiag(metric(ga)) &&
            Garamon.basis(ga)==plan.names && kind(ga)==plan.algebra_kind &&
            all(metric(ga)[i,i]==plan.diagonal[i] for i in eachindex(plan.diagonal)) ||
            throw(ArgumentError("algebra changed"))
        all(m in support for m in keys(mv.values)) &&
            (allow_subsets || length(mv.values)==length(support)) ||
            throw(ArgumentError("support changed; rebuild or declare subsets"))
        eltype(mv)<:Real && all(isfinite,values(mv.values)) ||
            throw(ArgumentError("finite real input coefficients required"))
    end
    nothing
end

function accumulate_compact!(out,av,bv,plan,a,b)
    fill!(out,zero(eltype(out)))
    for i in eachindex(av);av[i]=get(a.values,plan.left_masks[i],zero(eltype(av)));end
    for j in eachindex(bv);bv[j]=get(b.values,plan.right_masks[j],zero(eltype(bv)));end
    for j in eachindex(bv),i in eachindex(av)
        oi=plan.output_index[i,j]
        iszero(oi) && continue
        out[oi]+=plan.cocycle[i,j]*av[i]*bv[j]
    end
    out
end

function owned_compact_result(plan::CompactRankPlan{K},out,ga) where K
    S=eltype(out)
    # Owned dictionary values are copied out of the reusable numerical buffer.
    SparseMultiVector(ga,Dict{K,S}(plan.phi[i]=>out[i] for i in eachindex(out) if !iszero(out[i])))
end

function compact_rank_product(plan::CompactRankPlan,a,b;allow_subsets=false)
    validate_compact(plan,a,b;allow_subsets)
    S=promote_type(eltype(plan.diagonal),eltype(a),eltype(b))
    av=Vector{S}(undef,length(plan.left_masks));bv=Vector{S}(undef,length(plan.right_masks))
    out=Vector{S}(undef,length(plan.phi))
    accumulate_compact!(out,av,bv,plan,a,b)
    owned_compact_result(plan,out,a.algebra)
end

"""Reuse one workspace serially; each returned multivector owns its values."""
function compact_rank_product!(workspace::CompactRankWorkspace{S},a,b;allow_subsets=false) where S
    plan=workspace.plan
    validate_compact(plan,a,b;allow_subsets)
    promote_type(eltype(plan.diagonal),eltype(a),eltype(b))==S ||
        throw(ArgumentError("workspace coefficient type differs from product type"))
    accumulate_compact!(workspace.output_values,workspace.left_values,workspace.right_values,plan,a,b)
    owned_compact_result(plan,workspace.output_values,a.algebra)
end
end
