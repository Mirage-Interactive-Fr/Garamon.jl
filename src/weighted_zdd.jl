# An exact coefficient-valued zero-suppressed decision diagram. Node IDs are
# positive; negative IDs are nonzero rational terminals; zero is the zero DAG.
# Branching is by increasing basis index. Omitted levels mean the bit is absent.
const _WZQ = Rational{BigInt}

struct WeightedZDD
    dimension::Int
    nodes::Vector{NTuple{3,Int}}
    terminals::Vector{_WZQ}
    root::Int
end

mutable struct _WZBuilder
    dimension::Int
    max_nodes::Int
    max_work::Int
    work::Int
    nodes::Vector{NTuple{3,Int}}
    terminals::Vector{_WZQ}
    node_intern::Dict{NTuple{3,Int},Int}
    terminal_intern::Dict{_WZQ,Int}
    add_memo::Dict{NTuple{3,Int},Int}
    scale_memo::Dict{Tuple{Int,_WZQ},Int}
    product_memo::Dict{NTuple{4,Int},Int}
end

function _wz_builder(n,max_nodes,max_work)
    1<=n<=129 || throw(ArgumentError("weighted ZDD dimension budget is 1..129"))
    max_nodes>=1 && max_work>=1 || throw(ArgumentError("positive ZDD budgets required"))
    _WZBuilder(n,max_nodes,max_work,0,NTuple{3,Int}[],_WZQ[],
        Dict{NTuple{3,Int},Int}(),Dict{_WZQ,Int}(),
        Dict{NTuple{3,Int},Int}(),Dict{Tuple{Int,_WZQ},Int}(),
        Dict{NTuple{4,Int},Int}())
end

function _wz_work!(builder)
    builder.work+=1
    builder.work<=builder.max_work || throw(ArgumentError("weighted ZDD work budget"))
end

function _wz_rational(value)
    value isa Union{Integer,Rational} ||
        throw(ArgumentError("weighted ZDD requires exact integer/rational coefficients"))
    q=_WZQ(value)
    max(ndigits(abs(numerator(q));base=2),ndigits(denominator(q);base=2))<=8192 ||
        throw(ArgumentError("weighted ZDD coefficient bit budget"))
    q
end

function _wz_terminal!(builder,value)
    q=_wz_rational(value)
    iszero(q) && return 0
    get!(builder.terminal_intern,q) do
        length(builder.nodes)+length(builder.terminals)<builder.max_nodes ||
            throw(ArgumentError("weighted ZDD node budget"))
        push!(builder.terminals,q)
        -length(builder.terminals)
    end
end

function _wz_node!(builder,bit,lo,hi)
    hi==0 && return lo
    1<=bit<=builder.dimension || throw(ArgumentError("weighted ZDD node level"))
    get!(builder.node_intern,(bit,lo,hi)) do
        length(builder.nodes)+length(builder.terminals)<builder.max_nodes ||
            throw(ArgumentError("weighted ZDD node budget"))
        push!(builder.nodes,(bit,lo,hi))
        length(builder.nodes)
    end
end

function _wz_build!(builder,items,bit)
    isempty(items) && return 0
    _wz_work!(builder)
    bit>builder.dimension && return _wz_terminal!(builder,only(items).second)
    lo=Pair{BigInt,_WZQ}[]
    hi=Pair{BigInt,_WZQ}[]
    flag=big(1)<<(bit-1)
    for item in items
        push!(iszero(item.first & flag) ? lo : hi,item)
    end
    _wz_node!(builder,bit,_wz_build!(builder,lo,bit+1),
        _wz_build!(builder,hi,bit+1))
end

"""Build a bounded coefficient ZDD from exact blade-mask coefficients."""
function weighted_zdd(n::Integer,terms::AbstractDict;
                      max_nodes::Integer=65_536,max_work::Integer=1_000_000)
    builder=_wz_builder(Int(n),Int(max_nodes),Int(max_work))
    length(terms)<=4096 || throw(ArgumentError("weighted ZDD input support budget"))
    items=Pair{BigInt,_WZQ}[]
    for (mask,value) in terms
        mask isa Integer && 0<=mask<(big(1)<<n) ||
            throw(ArgumentError("weighted ZDD blade mask outside dimension"))
        q=_wz_rational(value)
        iszero(q) || push!(items,BigInt(mask)=>q)
    end
    sort!(items;by=first)
    root=_wz_build!(builder,items,1)
    WeightedZDD(builder.dimension,builder.nodes,builder.terminals,root)
end

function _wz_split(dag::WeightedZDD,id::Int,bit::Int)
    id<=0 && return (id,0)
    level,lo,hi=dag.nodes[id]
    level==bit && return (lo,hi)
    level>bit || error("weighted ZDD level invariant")
    (id,0)
end

function _wz_split(builder::_WZBuilder,id::Int,bit::Int)
    id<=0 && return (id,0)
    level,lo,hi=builder.nodes[id]
    level==bit && return (lo,hi)
    level>bit || error("weighted ZDD level invariant")
    (id,0)
end

function _wz_add!(builder,a::Int,b::Int,bit::Int)
    a==0 && return b
    b==0 && return a
    key=(min(a,b),max(a,b),bit)
    get!(builder.add_memo,key) do
        _wz_work!(builder)
        if bit>builder.dimension
            a<0 && b<0 || error("weighted ZDD terminal invariant")
            return _wz_terminal!(builder,builder.terminals[-a]+builder.terminals[-b])
        end
        a0,a1=_wz_split(builder,a,bit)
        b0,b1=_wz_split(builder,b,bit)
        _wz_node!(builder,bit,_wz_add!(builder,a0,b0,bit+1),
            _wz_add!(builder,a1,b1,bit+1))
    end
end

function _wz_scale!(builder,id::Int,factor::_WZQ)
    iszero(factor) && return 0
    id==0 && return 0
    isone(factor) && return id
    get!(builder.scale_memo,(id,factor)) do
        _wz_work!(builder)
        if id<0
            _wz_terminal!(builder,builder.terminals[-id]*factor)
        else
            bit,lo,hi=builder.nodes[id]
            _wz_node!(builder,bit,_wz_scale!(builder,lo,factor),
                _wz_scale!(builder,hi,factor))
        end
    end
end

function _wz_product!(builder,left::WeightedZDD,right::WeightedZDD,
                      diagonal::Vector{_WZQ},bit::Int,a::Int,b::Int,
                      right_lower_parity::Bool)
    (a==0 || b==0) && return 0
    key=(bit,a,b,Int(right_lower_parity))
    get!(builder.product_memo,key) do
        _wz_work!(builder)
        if bit>builder.dimension
            a<0 && b<0 || error("weighted ZDD product terminal invariant")
            return _wz_terminal!(builder,left.terminals[-a]*right.terminals[-b])
        end
        a0,a1=_wz_split(left,a,bit)
        b0,b1=_wz_split(right,b,bit)
        sign=right_lower_parity ? -one(_WZQ) : one(_WZQ)
        p00=_wz_product!(builder,left,right,diagonal,bit+1,a0,b0,right_lower_parity)
        p11=_wz_product!(builder,left,right,diagonal,bit+1,a1,b1,!right_lower_parity)
        lo=_wz_add!(builder,p00,_wz_scale!(builder,p11,sign*diagonal[bit]),bit+1)
        p10=_wz_product!(builder,left,right,diagonal,bit+1,a1,b0,right_lower_parity)
        p01=_wz_product!(builder,left,right,diagonal,bit+1,a0,b1,!right_lower_parity)
        hi=_wz_add!(builder,_wz_scale!(builder,p10,sign),p01,bit+1)
        _wz_node!(builder,bit,lo,hi)
    end
end

"""Exact geometric product by memoized ZDD branch recursion for a diagonal metric."""
function weighted_zdd_product(left::WeightedZDD,right::WeightedZDD,diagonal::AbstractVector;
                              max_nodes::Integer=65_536,max_work::Integer=1_000_000)
    n=left.dimension
    right.dimension==n && length(diagonal)==n ||
        throw(DimensionMismatch("weighted ZDD dimensions or diagonal differ"))
    g=_wz_rational.(diagonal)
    builder=_wz_builder(n,Int(max_nodes),Int(max_work))
    root=_wz_product!(builder,left,right,g,1,left.root,right.root,false)
    WeightedZDD(n,builder.nodes,builder.terminals,root)
end

function weighted_zdd_coefficient(dag::WeightedZDD,mask::Integer)
    0<=mask<(big(1)<<dag.dimension) ||
        throw(ArgumentError("weighted ZDD query mask outside dimension"))
    id=dag.root
    for bit in 1:dag.dimension
        id==0 && return zero(_WZQ)
        present=!iszero(mask & (big(1)<<(bit-1)))
        if id>0
            level,lo,hi=dag.nodes[id]
            if level==bit
                id=present ? hi : lo
            elseif present
                return zero(_WZQ)
            end
        elseif present
            return zero(_WZQ)
        end
    end
    id<0 ? dag.terminals[-id] : zero(_WZQ)
end

function weighted_zdd_terms(dag::WeightedZDD;max_terms::Integer=4096)
    max_terms>=1 || throw(ArgumentError("positive extraction budget required"))
    result=Dict{BigInt,_WZQ}()
    function visit(id,mask)
        id==0 && return
        if id<0
            length(result)<max_terms || throw(ArgumentError("weighted ZDD extraction budget"))
            result[mask]=dag.terminals[-id]
            return
        end
        bit,lo,hi=dag.nodes[id]
        visit(lo,mask)
        visit(hi,mask | (big(1)<<(bit-1)))
    end
    visit(dag.root,big(0))
    result
end
