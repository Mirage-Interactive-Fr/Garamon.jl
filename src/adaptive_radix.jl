# Exact bytewise index for sparse blade supports. Unary paths are compressed;
# branch lookup changes from a short linear key array to a 256-slot byte map.
abstract type AbstractARTNode{K,T} end

struct ARTLeaf{K,T} <: AbstractARTNode{K,T}
    mask::K
    value::T
end

mutable struct ARTBranch{K,T} <: AbstractARTNode{K,T}
    prefix::Vector{UInt8}
    keys::Vector{UInt8}
    children::Vector{Union{ARTLeaf{K,T},ARTBranch{K,T}}}
    lookup::Union{Nothing,Vector{UInt16}}
    capacity::Int
end

function _art_branch(::Type{K},::Type{T},prefix::Vector{UInt8}) where {K,T}
    keys=UInt8[]
    children=Union{ARTLeaf{K,T},ARTBranch{K,T}}[]
    sizehint!(keys,4)
    sizehint!(children,4)
    ARTBranch{K,T}(prefix,keys,children,nothing,4)
end

@inline _art_byte(mask::Integer,place::Int)=UInt8((mask >> (8*(place-1))) & 0xff)

function _art_child_index(node::ARTBranch,key::UInt8)
    if node.lookup===nothing
        found=findfirst(==(key),node.keys)
        return isnothing(found) ? 0 : found
    end
    Int(node.lookup[Int(key)+1])
end

function _art_add_child!(node::ARTBranch{K,T},key::UInt8,
                         child::Union{ARTLeaf{K,T},ARTBranch{K,T}}) where {K,T}
    _art_child_index(node,key)==0 || error("duplicate radix edge")
    count=length(node.children)+1
    count<=256 || error("byte radix fanout exceeded")
    if count>node.capacity
        node.capacity=count<=16 ? 16 : count<=48 ? 48 : 256
        sizehint!(node.keys,node.capacity)
        sizehint!(node.children,node.capacity)
    end
    push!(node.keys,key)
    push!(node.children,child)
    if count==17
        lookup=zeros(UInt16,256)
        for (position,edge) in enumerate(node.keys)
            lookup[Int(edge)+1]=UInt16(position)
        end
        node.lookup=lookup
    elseif node.lookup!==nothing
        node.lookup[Int(key)+1]=UInt16(count)
    end
    node
end

function _art_insert(node::ARTLeaf{K,T},mask::K,value::T,depth::Int,
                     nbytes::Int) where {K,T}
    node.mask==mask && return ARTLeaf{K,T}(mask,value)
    split=depth
    while split<=nbytes && _art_byte(node.mask,split)==_art_byte(mask,split)
        split+=1
    end
    split<=nbytes || error("distinct masks have identical byte keys")
    parent=_art_branch(K,T,UInt8[_art_byte(mask,i) for i in depth:split-1])
    _art_add_child!(parent,_art_byte(node.mask,split),node)
    _art_add_child!(parent,_art_byte(mask,split),ARTLeaf{K,T}(mask,value))
    parent
end

function _art_insert(node::ARTBranch{K,T},mask::K,value::T,depth::Int,
                     nbytes::Int) where {K,T}
    matched=0
    while matched<length(node.prefix) &&
          node.prefix[matched+1]==_art_byte(mask,depth+matched)
        matched+=1
    end
    if matched<length(node.prefix)
        parent=_art_branch(K,T,copy(node.prefix[1:matched]))
        oldedge=node.prefix[matched+1]
        node.prefix=copy(node.prefix[matched+2:end])
        _art_add_child!(parent,oldedge,node)
        _art_add_child!(parent,_art_byte(mask,depth+matched),
                        ARTLeaf{K,T}(mask,value))
        return parent
    end
    edge_place=depth+matched
    edge_place<=nbytes || error("radix branch extends beyond mask width")
    edge=_art_byte(mask,edge_place)
    position=_art_child_index(node,edge)
    if position==0
        _art_add_child!(node,edge,ARTLeaf{K,T}(mask,value))
    else
        node.children[position]=_art_insert(node.children[position],mask,value,
                                            edge_place+1,nbytes)
    end
    node
end

"""An owned snapshot of right-hand sparse blade coefficients for wedge queries."""
struct AdaptiveRadixIndex{K,T,A<:GeometricAlgebra}
    algebra::A
    root::Union{Nothing,ARTLeaf{K,T},ARTBranch{K,T}}
    nbytes::Int
    support::Int
end

"""Build a path-compressed adaptive byte radix index; no coefficients are dropped."""
function prepare_adaptive_radix(right::SparseMultiVector{T,K};
                                max_nodes::Int=65536) where {T,K}
    max_nodes>=1 || throw(ArgumentError("max_nodes must be positive"))
    count=length(right.values)
    count<=div(max_nodes+1,2) ||
        throw(ArgumentError("radix index node budget"))
    nbytes=cld(dimension(right.algebra),8)
    root::Union{Nothing,ARTLeaf{K,T},ARTBranch{K,T}}=nothing
    for mask in sort!(collect(keys(right.values)))
        value=right.values[mask]
        root=isnothing(root) ? ARTLeaf{K,T}(mask,value) :
             _art_insert(root,mask,value,1,nbytes)
    end
    AdaptiveRadixIndex{K,T,typeof(right.algebra)}(right.algebra,root,nbytes,count)
end

function _art_accumulate!(output::Dict{K,S},node::AbstractARTNode{K,T},
                          leftmask::K,leftvalue,depth::Int,
                          max_terms::Int) where {K,S,T}
    if node isa ARTLeaf{K,T}
        leaf=node::ARTLeaf{K,T}
        iszero(leftmask & leaf.mask) || return nothing
        mask=leftmask | leaf.mask
        value=get(output,mask,zero(S))+
            convert(S,_art_wedge_sign(leftmask,leaf.mask)*leftvalue*leaf.value)
        if iszero(value)
            delete!(output,mask)
        else
            output[mask]=value
            length(output)<=max_terms ||
                throw(ArgumentError("radix wedge output term budget"))
        end
        return nothing
    end
    branch=node::ARTBranch{K,T}
    for (offset,byte) in enumerate(branch.prefix)
        iszero(byte & _art_byte(leftmask,depth+offset-1)) || return nothing
    end
    edge_place=depth+length(branch.prefix)
    leftbyte=_art_byte(leftmask,edge_place)
    for position in eachindex(branch.keys)
        edge=branch.keys[position]
        iszero(edge & leftbyte) || continue
        _art_accumulate!(output,branch.children[position],leftmask,leftvalue,
                         edge_place+1,max_terms)
    end
    nothing
end

function _art_wedge_sign(left::K,right::K) where K<:Integer
    parity=false
    remaining=left
    while !iszero(remaining)
        position=trailing_zeros(remaining)
        parity ⊻=isodd(count_ones(right & ((one(K)<<position)-one(K))))
        remaining &= remaining-one(K)
    end
    parity ? -1 : 1
end

"""Exact sparse wedge against a prepared radix snapshot, with a term budget."""
function radix_wedge(left::SparseMultiVector{TL,K},
                     index::AdaptiveRadixIndex{K,TR};
                     max_terms::Int=65536) where {TL,TR,K}
    max_terms>=1 || throw(ArgumentError("max_terms must be positive"))
    ga=left.algebra
    target=index.algebra
    dimension(ga)==dimension(target) && metric(ga)==metric(target) &&
        basis(ga)==basis(target) && kind(ga)==kind(target) ||
        throw(ArgumentError("radix index belongs to another algebra"))
    output=Dict{K,promote_type(TL,TR)}()
    isnothing(index.root) && return SparseMultiVector(ga,output)
    for (leftmask,leftvalue) in left.values
        _art_accumulate!(output,index.root,leftmask,leftvalue,1,max_terms)
    end
    SparseMultiVector(ga,output)
end

function _art_stats(node::ARTLeaf)
    (branches=0,leaves=1,prefix_bytes=0,capacity4=0,capacity16=0,
     capacity48=0,capacity256=0)
end
function _art_stats(node::ARTBranch)
    children=map(_art_stats,node.children)
    sums=map(key->sum(getproperty(x,key) for x in children;init=0),
             (:branches,:leaves,:prefix_bytes,:capacity4,:capacity16,
              :capacity48,:capacity256))
    (branches=1+sums[1],leaves=sums[2],
     prefix_bytes=length(node.prefix)+sums[3],
     capacity4=(node.capacity==4)+sums[4],
     capacity16=(node.capacity==16)+sums[5],
     capacity48=(node.capacity==48)+sums[6],
     capacity256=(node.capacity==256)+sums[7])
end

"""Node counts and active capacity tiers of a prepared radix index."""
radix_stats(index::AdaptiveRadixIndex)=isnothing(index.root) ?
    (branches=0,leaves=0,prefix_bytes=0,capacity4=0,capacity16=0,
     capacity48=0,capacity256=0) : _art_stats(index.root)
