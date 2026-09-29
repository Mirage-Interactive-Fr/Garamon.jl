"""
    GeneratedProduct

A bounded Julia specialization of a structural `ProductPlan`. The path indices
are in the method's type, while metric factors and operand coefficients remain
runtime values. The first execution may incur Julia compilation latency.
"""
struct GeneratedProduct{P,PL<:ProductPlan,T}
    plan::PL
    factors::Vector{T}
end

"""
    generate_product(plan; max_paths=64)

Generate a path-unrolled Julia kernel on demand. The hard ceiling of 256 paths
limits method size even if the caller raises `max_paths`. This function prepares
the specialization; Julia compiles it on first use. Exact support, metric and
basis checks still run for each call.
"""
function generate_product(plan::ProductPlan; max_paths::Integer=64)
    0 <= max_paths <= 256 ||
        throw(ArgumentError("generated product max_paths must be within 0:256"))
    length(plan.paths) <= max_paths ||
        throw(ArgumentError("generated product exceeds max_paths"))
    P = Tuple((ai, bi, oi) for (ai, bi, oi, _) in plan.paths)
    T = eltype(plan.diagonal)
    factors = T[factor for (_, _, _, factor) in plan.paths]
    return GeneratedProduct{P,typeof(plan),T}(plan, factors)
end

@generated function _generated_path_values(::Val{P}, factors::Vector{F},
                                           left::Vector{S}, right::Vector{S},
                                           ::Val{O}) where {P,F,S,O}
    length(P) <= 256 || error("generated path ceiling exceeded")
    statements = [:(output[$oi] += factors[$i] * left[$ai] * right[$bi])
                  for (i, (ai, bi, oi)) in enumerate(P)]
    return quote
        output = zeros(promote_type(F, S), O)
        $(statements...)
        output
    end
end

"""Run a generated product after checking support and algebra snapshots."""
function run_generated_product(program::GeneratedProduct{P},
                               a::AbstractMultiVector,
                               b::AbstractMultiVector) where P
    plan = program.plan
    ga = _validate_plan_inputs(plan, a, b)
    S = promote_type(eltype(plan.diagonal), eltype(a), eltype(b))
    left = S[coefficient_mask(a, mask) for mask in plan.left_masks]
    right = S[coefficient_mask(b, mask) for mask in plan.right_masks]
    values = _generated_path_values(Val(P), program.factors, left, right,
                                    Val(length(plan.output_masks)))
    K = eltype(plan.output_masks)
    return SparseMultiVector(ga,
        Dict{K,eltype(values)}(mask => value
            for (mask, value) in zip(plan.output_masks, values)
            if !iszero(value)))
end
