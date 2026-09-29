"""Add a scalar to the grade-zero coefficient of a multivector."""
Base.:+(a::AbstractMultiVector, value::Number) =
    a + scalar(a.algebra, value;
               storage=a isa DenseMultiVector ? :dense : :sparse)
Base.:+(value::Number, a::AbstractMultiVector) = a + value
Base.:-(a::AbstractMultiVector, value::Number) = a + (-value)
Base.:-(value::Number, a::AbstractMultiVector) = value + (-a)

"""Divide by a checked multivector inverse, subject to the inverse's budget."""
Base.:/(a::AbstractMultiVector, b::AbstractMultiVector) = a * inv(b)
Base.:/(value::Number, b::AbstractMultiVector) = value * inv(b)

"""Scalar operands follow the grade-zero specializations of the C++ products."""
wedge(a::AbstractMultiVector, value::Number) = value * a
wedge(value::Number, a::AbstractMultiVector) = value * a
inner_product(a::AbstractMultiVector, ::Number) = zero(a)
inner_product(::Number, a::AbstractMultiVector) = zero(a)
dot_product(a::AbstractMultiVector, value::Number) = value * a
dot_product(value::Number, a::AbstractMultiVector) = value * a
left_contraction(value::Number, a::AbstractMultiVector) = value * a
right_contraction(a::AbstractMultiVector, value::Number) = value * a
left_contraction(a::AbstractMultiVector, value::Number) =
    scalar(a.algebra, scalarpart(a) * value;
           storage=a isa DenseMultiVector ? :dense : :sparse)
right_contraction(value::Number, a::AbstractMultiVector) =
    scalar(a.algebra, scalarpart(a) * value;
           storage=a isa DenseMultiVector ? :dense : :sparse)
scalar_product(a::AbstractMultiVector, value::Number) =
    scalar(a.algebra, scalarpart(a) * value;
           storage=a isa DenseMultiVector ? :dense : :sparse)
scalar_product(value::Number, a::AbstractMultiVector) = scalar_product(a, value)

"""Sorted grades having at least one nonzero coefficient."""
active_grades(a::AbstractMultiVector) =
    sort!(unique([_blade_grade(mask) for (mask, _) in _terms(a)]))

"""Highest active grade, with zero for the zero multivector."""
function highest_grade(a::AbstractMultiVector)
    grades = active_grades(a)
    return isempty(grades) ? 0 : last(grades)
end

"""Whether at most one grade has nonzero coefficients."""
is_homogeneous(a::AbstractMultiVector) = length(active_grades(a)) <= 1
has_grade(a::AbstractMultiVector, k::Integer) =
    k in active_grades(a) || (k == 0 && iszero(a))
same_grade(a::AbstractMultiVector, b::AbstractMultiVector) =
    highest_grade(a) == highest_grade(b)
Base.iszero(a::DenseMultiVector) = all(iszero, a.values)
Base.iszero(a::SparseMultiVector) = isempty(a.values)

Base.copy(a::DenseMultiVector) = DenseMultiVector(a.algebra, copy(a.values))
Base.copy(a::SparseMultiVector) = SparseMultiVector(a.algebra, a.values)

"""Read a coefficient by C++ compatible zero-based grade position."""
coefficient_grade(a::AbstractMultiVector, k::Integer, position::Integer) =
    coefficient_mask(a, blade_unrank(a.algebra, k, position))

"""Set a coefficient by C++ compatible zero-based grade position."""
set_grade_coefficient!(a::AbstractMultiVector, k::Integer,
                       position::Integer, value::Number) =
    set_coefficient!(a, _indices(a.algebra, blade_unrank(a.algebra, k, position)),
                     value)

"""Set one ordered-blade coefficient in a mutable coefficient container."""
function set_coefficient!(a::AbstractMultiVector, indices, value::Number)
    mask = _mask(a.algebra, indices)
    converted = convert(eltype(a), value)
    if a isa DenseMultiVector
        a.values[Int(mask) + 1] = converted
    elseif iszero(converted)
        delete!(a.values, mask)
    else
        a.values[mask] = converted
    end
    return a
end

"""Remove one complete grade from a multivector in place."""
function clear_grade!(a::AbstractMultiVector, k::Integer)
    0 <= k <= dimension(a.algebra) || throw(BoundsError(a, k))
    if a isa DenseMultiVector
        for i in eachindex(a.values)
            _blade_grade(UInt64(i - 1)) == k && (a.values[i] = zero(eltype(a)))
        end
    else
        for mask in collect(keys(a.values))
            _blade_grade(mask) == k && delete!(a.values, mask)
        end
    end
    return a
end

"""Clear all coefficients in place."""
function Base.empty!(a::DenseMultiVector)
    fill!(a.values, zero(eltype(a)))
    return a
end
function Base.empty!(a::SparseMultiVector)
    empty!(a.values)
    return a
end

"""Explicit numerical cleanup; exact products never invoke this operation."""
function round_zero!(a::AbstractMultiVector; atol::Real)
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    if a isa DenseMultiVector
        for i in eachindex(a.values)
            abs(a.values[i]) <= atol && (a.values[i] = zero(eltype(a)))
        end
    else
        for (mask, value) in collect(pairs(a.values))
            abs(value) <= atol && delete!(a.values, mask)
        end
    end
    return a
end

"""Scalar part of `reverse(a) * a`, retaining the metric sign."""
quadratic_norm(a::AbstractMultiVector) = scalarpart(scalar_product(reverse(a), a))

"""Square root of the magnitude of the quadratic norm."""
clifford_norm(a::AbstractMultiVector) = sqrt(abs(quadratic_norm(a)))

"""Wedge with the right complement of the second operand."""
outer_primal_dual(a::AbstractMultiVector, b::AbstractMultiVector) =
    wedge(a, right_complement(b))
"""Wedge with the right complement of the first operand."""
outer_dual_primal(a::AbstractMultiVector, b::AbstractMultiVector) =
    wedge(right_complement(a), b)
"""Wedge the right complements of both operands."""
outer_dual_dual(a::AbstractMultiVector, b::AbstractMultiVector) =
    wedge(right_complement(a), right_complement(b))
