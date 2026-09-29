"""Reusable exact path graph for selected outputs of a diagonal triple product."""
struct TripleJoinPlan{T,K<:Integer,A<:GeometricAlgebra}
    algebra::A
    diagonal::Vector{T}
    basis_names::Vector{String}
    left_support::Set{K}
    middle_support::Set{K}
    right_support::Set{K}
    left_masks::Vector{K}
    middle_masks::Vector{K}
    right_masks::Vector{K}
    output_masks::Vector{K}
    paths::Vector{Tuple{Int,Int,Int,Int,T}}
end

"""
    prepare_triple_join(a, b, c, outputs; max_probes=1<<20,
                        max_paths=1<<16)

Prepare all nonzero paths to the requested coefficients of `(a*b)*c` once.
The two smallest supports are enumerated for each distinct output and XOR
determines the remaining mask. The operand order inside each product stays
`a`, `b`, `c`; no algebraic contribution is sampled or discarded. A path whose
metric factor is provably zero is omitted. Supports and the metric are snapshotted
and checked at every later call. Both probe and retained-path budgets are hard.
"""
function prepare_triple_join(a::AbstractMultiVector,
                             b::AbstractMultiVector,
                             c::AbstractMultiVector, outputs;
                             max_probes::Integer=1 << 20,
                             max_paths::Integer=1 << 16)
    ga = _same_algebra(a, b)
    _same_algebra(a, c)
    isdiag(metric(ga)) ||
        throw(ArgumentError("triple join plan requires a diagonal metric"))
    max_probes >= 0 && max_paths >= 0 ||
        throw(ArgumentError("triple join budgets must be nonnegative"))
    K = _masktype(ga)
    T = eltype(metric(ga))
    supports = (Set{K}(convert(K, mask) for (mask, _) in _terms(a)),
                Set{K}(convert(K, mask) for (mask, _) in _terms(b)),
                Set{K}(convert(K, mask) for (mask, _) in _terms(c)))
    masks = map(support -> sort!(collect(support)), supports)
    targets = unique(K[_mask(ga, indices) for indices in outputs])
    lengths = map(length, masks)
    order = sortperm(collect(1:3); by=i -> (lengths[i], i))
    first_id, second_id, third_id = order
    big(length(targets)) * lengths[first_id] * lengths[second_id] <= max_probes ||
        throw(ArgumentError("triple join plan exceeds max_probes"))
    third_index = Dict(mask => i for (i, mask) in enumerate(masks[third_id]))
    paths = Tuple{Int,Int,Int,Int,T}[]
    for (oi, target) in enumerate(targets)
        for (first_index, first_mask) in enumerate(masks[first_id]),
            (second_index, second_mask) in enumerate(masks[second_id])
            remaining = first_mask ⊻ second_mask ⊻ target
            third_index_value = get(third_index, remaining, 0)
            iszero(third_index_value) && continue
            ai, bi, ci = ntuple(3) do id
                id == first_id ? first_index :
                id == second_id ? second_index : third_index_value
            end
            factor = _triple_diagonal_factor(ga, masks[1][ai],
                                              masks[2][bi], masks[3][ci], T)
            iszero(factor) && continue
            length(paths) < max_paths ||
                throw(ArgumentError("triple join plan exceeds max_paths"))
            push!(paths, (ai, bi, ci, oi, factor))
        end
    end
    return TripleJoinPlan(ga, collect(diag(metric(ga))), copy(basis(ga)),
                          supports..., masks..., targets, paths)
end

function _validate_triple_join_inputs(plan::TripleJoinPlan,
                                      a::AbstractMultiVector,
                                      b::AbstractMultiVector,
                                      c::AbstractMultiVector)
    ga = _same_algebra(a, b)
    _same_algebra(a, c)
    valid = dimension(ga) == length(plan.diagonal) && isdiag(metric(ga)) &&
            basis(ga) == plan.basis_names &&
            kind(ga) == kind(plan.algebra)
    if valid
        for i in eachindex(plan.diagonal)
            if metric(ga)[i, i] != plan.diagonal[i]
                valid = false
                break
            end
        end
    end
    valid || throw(ArgumentError("triple join plan belongs to a different algebra"))
    _matches_plan_support(a, plan.left_support) &&
        _matches_plan_support(b, plan.middle_support) &&
        _matches_plan_support(c, plan.right_support) ||
        throw(ArgumentError("triple join input support differs from the plan"))
    return ga
end

"""Buffers for an exact triple join; never share one instance between tasks."""
struct TripleJoinWorkspace{S,P<:TripleJoinPlan}
    plan::P
    left_values::Vector{S}
    middle_values::Vector{S}
    right_values::Vector{S}
    output_values::Vector{S}
end

function TripleJoinWorkspace(plan::TripleJoinPlan,
                             a::AbstractMultiVector,
                             b::AbstractMultiVector,
                             c::AbstractMultiVector;
                             max_bytes::Integer=64 << 20)
    max_bytes >= 1 || throw(ArgumentError("workspace max_bytes must be positive"))
    _validate_triple_join_inputs(plan, a, b, c)
    S = promote_type(eltype(plan.diagonal), eltype(a), eltype(b), eltype(c))
    limit = min(big(max_bytes), big(Sys.free_memory()) ÷ 4)
    Base.summarysize(plan) <= limit ||
        throw(ArgumentError("triple join workspace exceeds memory admission budget"))
    workspace = TripleJoinWorkspace{S,typeof(plan)}(
        plan, Vector{S}(undef, length(plan.left_masks)),
        Vector{S}(undef, length(plan.middle_masks)),
        Vector{S}(undef, length(plan.right_masks)),
        zeros(S, length(plan.output_masks)))
    Base.summarysize(workspace) <= limit ||
        throw(ArgumentError("triple join workspace exceeds memory admission budget"))
    return workspace
end

"""
    run_triple_join_values!(workspace, a, b, c)

Return the workspace-owned vector in `plan.output_masks` order. It is overwritten
on the next call. Metric and exact support are validated before writing.
"""
function run_triple_join_values!(workspace::TripleJoinWorkspace{S},
                                 a::AbstractMultiVector,
                                 b::AbstractMultiVector,
                                 c::AbstractMultiVector) where S
    plan = workspace.plan
    _validate_triple_join_inputs(plan, a, b, c)
    for (i, mask) in enumerate(plan.left_masks)
        workspace.left_values[i] = _workspace_coefficient(a, mask)
    end
    for (i, mask) in enumerate(plan.middle_masks)
        workspace.middle_values[i] = _workspace_coefficient(b, mask)
    end
    for (i, mask) in enumerate(plan.right_masks)
        workspace.right_values[i] = _workspace_coefficient(c, mask)
    end
    output = workspace.output_values
    fill!(output, zero(S))
    for (ai, bi, ci, oi, factor) in plan.paths
        @inbounds output[oi] += factor * workspace.left_values[ai] *
                                workspace.middle_values[bi] * workspace.right_values[ci]
    end
    return output
end
