"""
    blade_rank(ga, mask) -> (grade, position)

Return the zero-based position within Garamon C++'s lexicographic homogeneous
blade order. Uses binomial ranking without allocating a 2^n lookup table.
"""
function blade_rank(ga::GeometricAlgebra, mask::Integer)
    _mask_in_bounds(ga, mask) ||
        throw(BoundsError(ga, mask))
    typed_mask = convert(_masktype(ga), mask)
    k = _blade_grade(typed_mask)
    rank = big(0)
    previous = 0
    position = 0
    for index in _indices(ga, typed_mask)
        position += 1
        for skipped in (previous + 1):(index - 1)
            rank += binomial(big(dimension(ga) - skipped), k - position)
        end
        previous = index
    end
    return (k, rank)
end
blade_rank(ga::GeometricAlgebra, indices::AbstractVector{<:Integer}) =
    blade_rank(ga, _mask(ga, indices))

"""
    blade_unrank(ga, grade, position) -> mask

Invert blade_rank for the zero-based homogeneous position. The result is a
UInt64 through 64 dimensions, UInt128 through 128, and BigInt beyond 128.
"""
function blade_unrank(ga::GeometricAlgebra, grade::Integer, position::Integer)
    n = dimension(ga)
    0 <= grade <= n || throw(BoundsError(ga, grade))
    0 <= position < binomial(big(n), grade) ||
        throw(BoundsError(ga, position))
    remainder = big(position)
    result = zero(_masktype(ga))
    next_index = 1
    for selected in 1:grade
        for candidate in next_index:n
            count = binomial(big(n - candidate), grade - selected)
            if remainder < count
                result |= one(result) << (candidate - 1)
                next_index = candidate + 1
                break
            end
            remainder -= count
        end
    end
    return result
end
