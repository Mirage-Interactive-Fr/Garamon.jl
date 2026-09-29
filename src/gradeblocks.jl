"""A bounded full-grade layout for a top-grade exterior coefficient."""
struct TopWedgePlan{K<:Integer}
    dimension::Int
    basis_names::Vector{String}
    algebra_kind::Symbol
    left_grade::Int
    full_mask::K
    left_masks::Vector{K}
    positions::Dict{K,Int}
    signs::Vector{Int8}
end

"""
    prepare_top_wedge(ga, left_grade; max_slots=65536)

Enumerate only the `binomial(n,left_grade)` masks in one full grade, not `2^n`.
The plan is useful when repeated homogeneous wedges request only the top
coefficient. Setup and two block-packing passes should be counted separately.
"""
function prepare_top_wedge(ga::GeometricAlgebra, left_grade::Integer;
                           max_slots::Integer=1 << 16)
    n = dimension(ga)
    0 <= left_grade <= n || throw(ArgumentError("left grade is out of range"))
    max_slots >= 1 || throw(ArgumentError("max_slots must be positive"))
    binomial(big(n), left_grade) <= max_slots ||
        throw(ArgumentError("grade block exceeds max_slots"))
    K = _masktype(ga)
    masks = K[]
    function visit(start::Int, remaining::Int, mask)
        if iszero(remaining)
            push!(masks, mask)
            return
        end
        for i in start:(n - remaining + 1)
            visit(i + 1, remaining - 1, mask | (one(K) << (i - 1)))
        end
    end
    visit(1, Int(left_grade), zero(K))
    sort!(masks)
    positions = Dict(mask => i for (i, mask) in enumerate(masks))
    full_mask = (one(K) << n) - one(K)
    signs = Int8[_shuffle_sign(ga, mask, full_mask ⊻ mask) for mask in masks]
    return TopWedgePlan(n, copy(basis(ga)), kind(ga), Int(left_grade),
                        full_mask, masks, positions, signs)
end

"""Compute the top coefficient of a homogeneous wedge using packed grades."""
function top_wedge_coefficient(plan::TopWedgePlan,
                               a::AbstractMultiVector, b::AbstractMultiVector)
    ga = _same_algebra(a, b)
    dimension(ga) == plan.dimension && basis(ga) == plan.basis_names &&
        kind(ga) == plan.algebra_kind ||
        throw(ArgumentError("top-wedge plan belongs to a different basis"))
    T = promote_type(eltype(a), eltype(b))
    left_values = zeros(T, length(plan.left_masks))
    right_values = zeros(T, length(plan.left_masks))
    for (mask, value) in _terms(a)
        _blade_grade(mask) == plan.left_grade ||
            throw(ArgumentError("left operand is not homogeneous at plan grade"))
        left_values[plan.positions[mask]] = value
    end
    for (mask, value) in _terms(b)
        _blade_grade(mask) == plan.dimension - plan.left_grade ||
            throw(ArgumentError("right operand is not homogeneous at plan grade"))
        right_values[plan.positions[plan.full_mask ⊻ mask]] = value
    end
    result = zero(T)
    for i in eachindex(left_values)
        result += plan.signs[i] * left_values[i] * right_values[i]
    end
    return result
end

"""A bounded grade block for one specified exterior-product coefficient."""
struct WedgeCoefficientPlan{K<:Integer}
    dimension::Int
    basis_names::Vector{String}
    algebra_kind::Symbol
    left_grade::Int
    target_mask::K
    left_masks::Vector{K}
    positions::Dict{K,Int}
    signs::Vector{Int8}
end

"""
    prepare_wedge_coefficient(ga, left_grade, output; max_slots=65536)

Enumerate the `binomial(length(output),left_grade)` left blades contained in
one requested output mask. The right partner is the complement within that
mask. No full `2^n` table is created, including beyond 64 dimensions.
"""
function prepare_wedge_coefficient(ga::GeometricAlgebra, left_grade::Integer,
                                   output; max_slots::Integer=1 << 16)
    target = _mask(ga, output)
    target_grade = _blade_grade(target)
    0 <= left_grade <= target_grade ||
        throw(ArgumentError("left grade is out of range for requested output"))
    max_slots >= 1 || throw(ArgumentError("max_slots must be positive"))
    binomial(big(target_grade), left_grade) <= max_slots ||
        throw(ArgumentError("grade block exceeds max_slots"))
    K = _masktype(ga)
    directions = [i for i in 1:dimension(ga)
                  if !iszero(target & (one(K) << (i - 1)))]
    masks = K[]
    function visit(start::Int, remaining::Int, mask)
        if iszero(remaining)
            push!(masks, mask)
            return
        end
        for i in start:(length(directions) - remaining + 1)
            visit(i + 1, remaining - 1,
                  mask | (one(K) << (directions[i] - 1)))
        end
    end
    visit(1, Int(left_grade), zero(K))
    sort!(masks)
    positions = Dict(mask => i for (i, mask) in enumerate(masks))
    signs = Int8[_shuffle_sign(ga, mask, target ⊻ mask) for mask in masks]
    return WedgeCoefficientPlan(dimension(ga), copy(basis(ga)), kind(ga),
                                Int(left_grade), target, masks, positions, signs)
end

"""Evaluate one wedge coefficient from packed homogeneous grade blocks."""
function wedge_coefficient(plan::WedgeCoefficientPlan,
                           a::AbstractMultiVector, b::AbstractMultiVector)
    ga = _same_algebra(a, b)
    dimension(ga) == plan.dimension && basis(ga) == plan.basis_names &&
        kind(ga) == plan.algebra_kind ||
        throw(ArgumentError("wedge coefficient plan belongs to a different basis"))
    T = promote_type(eltype(a), eltype(b))
    left_values = zeros(T, length(plan.left_masks))
    right_values = zeros(T, length(plan.left_masks))
    right_grade = _blade_grade(plan.target_mask) - plan.left_grade
    for (mask, value) in _terms(a)
        _blade_grade(mask) == plan.left_grade ||
            throw(ArgumentError("left operand is not homogeneous at plan grade"))
        slot = get(plan.positions, mask, 0)
        iszero(slot) || (left_values[slot] = value)
    end
    for (mask, value) in _terms(b)
        _blade_grade(mask) == right_grade ||
            throw(ArgumentError("right operand is not homogeneous at plan grade"))
        iszero(mask & ~plan.target_mask) || continue
        right_values[plan.positions[plan.target_mask ⊻ mask]] = value
    end
    result = zero(T)
    for i in eachindex(left_values)
        result += plan.signs[i] * left_values[i] * right_values[i]
    end
    return result
end
