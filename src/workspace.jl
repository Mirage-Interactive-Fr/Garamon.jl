"""
    ProductWorkspace(plan, a, b; max_bytes=64<<20)

Own reusable coefficient buffers for a diagonal `ProductPlan`. The plan and
current operands are checked at construction. `max_bytes` is an admission
limit for the retained plan, buffers, and reserved sparse-result dictionary;
the same admission also requires this footprint to fit within a quarter of
currently free system memory. This is a snapshot, not a hard process or JIT
memory limit. Arbitrary-precision coefficient growth is not bounded by it.

Each concurrent evaluation needs its own workspace. Returned buffers and the
result of `run_product!` are overwritten by the next call on that workspace.
"""
struct ProductWorkspace{S,K<:Integer,P<:ProductPlan}
    plan::P
    left_values::Vector{S}
    right_values::Vector{S}
    output_values::Vector{S}
    result::SparseMultiVector{S,K}
end

# The plan already checked these masks, and each call validates the operands'
# algebra and support before entering the hot coefficient loop.
@inline _workspace_coefficient(mv::SparseMultiVector{T,K},
                               mask::K) where {T,K} =
    get(mv.values, mask, zero(T))
@inline _workspace_coefficient(mv::DenseMultiVector,
                               mask::UInt64) = mv.values[Int(mask) + 1]

function ProductWorkspace(plan::ProductPlan, a::AbstractMultiVector,
                          b::AbstractMultiVector;
                          max_bytes::Integer=64 << 20)
    max_bytes >= 1 || throw(ArgumentError("workspace max_bytes must be positive"))
    ga = _validate_plan_inputs(plan, a, b)
    S = promote_type(eltype(plan.diagonal), eltype(a), eltype(b))
    K = eltype(plan.output_masks)
    limit = min(big(max_bytes), big(Sys.free_memory()) ÷ 4)
    Base.summarysize(plan) <= limit ||
        throw(ArgumentError("product workspace exceeds memory admission budget"))
    result = SparseMultiVector(ga, Dict{K,S}())
    sizehint!(result.values, length(plan.output_masks))
    workspace = ProductWorkspace{S,K,typeof(plan)}(
        plan, Vector{S}(undef, length(plan.left_masks)),
        Vector{S}(undef, length(plan.right_masks)),
        zeros(S, length(plan.output_masks)), result)
    Base.summarysize(workspace) <= limit ||
        throw(ArgumentError("product workspace exceeds memory admission budget"))
    return workspace
end

"""
    run_product_values!(workspace, a, b)

Evaluate into the workspace's output coefficient vector, ordered by
`workspace.plan.output_masks`. The vector is reused and overwritten by the
next call. Coefficients remain exact whenever the operand arithmetic is exact.
"""
function run_product_values!(workspace::ProductWorkspace{S},
                             a::AbstractMultiVector,
                             b::AbstractMultiVector) where S
    plan = workspace.plan
    _validate_plan_inputs(plan, a, b)
    for (i, mask) in enumerate(plan.left_masks)
        workspace.left_values[i] = _workspace_coefficient(a, mask)
    end
    for (i, mask) in enumerate(plan.right_masks)
        workspace.right_values[i] = _workspace_coefficient(b, mask)
    end
    output = workspace.output_values
    fill!(output, zero(S))
    for (ai, bi, oi, factor) in plan.paths
        @inbounds output[oi] += factor * workspace.left_values[ai] *
                                workspace.right_values[bi]
    end
    return output
end

"""
    run_product!(workspace, a, b)

Evaluate into a reusable sparse multivector. The returned object aliases the
workspace and changes on the next `run_product!` call. Use `sparse(result)` if a
persistent snapshot is needed.
"""
function run_product!(workspace::ProductWorkspace,
                      a::AbstractMultiVector, b::AbstractMultiVector)
    values = run_product_values!(workspace, a, b)
    result = workspace.result
    empty!(result.values)
    for (mask, value) in zip(workspace.plan.output_masks, values)
        iszero(value) || (result.values[mask] = value)
    end
    return result
end
