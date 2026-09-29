# Bounded materialized trie for disjoint-pair wedge traversal. Construction
# and traversal remain separate, as in the original benchmark prototype.
mutable struct MaskTrie
    no::Union{Nothing,MaskTrie}
    yes::Union{Nothing,MaskTrie}
    value::Float64
end
MaskTrie() = MaskTrie(nothing, nothing, 0.0)

function build_trie(mv::SparseMultiVector, n::Int)
    root = MaskTrie()
    for (mask, value) in mv.values
        node = root
        for i in 0:(n - 1)
            side = iszero(mask & (one(mask) << i)) ? :no : :yes
            child = getfield(node, side)
            if child === nothing
                child = MaskTrie()
                setfield!(node, side, child)
            end
            node = child
        end
        node.value = value
    end
    return root
end

function trie_wedge!(output::Dict{UInt64,Float64}, left::MaskTrie,
                     right::MaskTrie, depth::Int, n::Int,
                     leftmask::UInt64, rightmask::UInt64)
    if depth == n
        mask = leftmask | rightmask
        sign = 1.0
        remaining = leftmask
        while !iszero(remaining)
            i = trailing_zeros(remaining)
            isodd(count_ones(rightmask & ((UInt64(1) << i) - 1))) &&
                (sign = -sign)
            remaining &= remaining - 1
        end
        output[mask] = get(output, mask, 0.0) + sign * left.value * right.value
        return
    end
    bit = UInt64(1) << depth
    if left.no !== nothing && right.no !== nothing
        trie_wedge!(output, left.no, right.no, depth + 1, n, leftmask, rightmask)
    end
    if left.no !== nothing && right.yes !== nothing
        trie_wedge!(output, left.no, right.yes, depth + 1, n, leftmask,
                    rightmask | bit)
    end
    if left.yes !== nothing && right.no !== nothing
        trie_wedge!(output, left.yes, right.no, depth + 1, n,
                    leftmask | bit, rightmask)
    end
end

function trie_wedge(left::MaskTrie, right::MaskTrie, n::Int)
    output = Dict{UInt64,Float64}()
    trie_wedge!(output, left, right, 0, n, 0x0000000000000000,
                0x0000000000000000)
    return output
end

count_nodes(node::MaskTrie) = 1 +
    (node.no === nothing ? 0 : count_nodes(node.no)) +
    (node.yes === nothing ? 0 : count_nodes(node.yes))

function wedge_visits(left::MaskTrie, right::MaskTrie, depth::Int, n::Int)
    depth == n && return (1, 1)
    nodes, leaves = 1, 0
    for (a, b) in ((left.no, right.no), (left.no, right.yes),
                   (left.yes, right.no))
        if a !== nothing && b !== nothing
            visited, useful = wedge_visits(a, b, depth + 1, n)
            nodes += visited
            leaves += useful
        end
    end
    return nodes, leaves
end
