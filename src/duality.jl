"""Negate every odd-grade component."""
function grade_involution(a::AbstractMultiVector{T}) where T
    result = _empty_mv(a.algebra, T, a isa DenseMultiVector ? :dense : :sparse)
    for (mask, value) in _terms(a)
        _accumulate!(result, mask, isodd(_blade_grade(mask)) ? -value : value)
    end
    return result
end

"""The composition of grade involution and reversion."""
clifford_conjugate(a::AbstractMultiVector) = grade_involution(reverse(a))

function _full_mask(ga::GeometricAlgebra)
    K = _masktype(ga)
    K <: Unsigned && dimension(ga) == 8sizeof(K) && return typemax(K)
    return (one(K) << dimension(ga)) - one(K)
end

"""
    right_complement(a)

The metric-independent exterior complement, defined blade-wise so that
`blade ∧ right_complement(blade)` equals the oriented unit pseudoscalar.
It remains defined for degenerate metrics and is not a metric dual.
"""
function right_complement(a::AbstractMultiVector{T}) where T
    ga = a.algebra
    full = _full_mask(ga)
    result = _empty_mv(ga, T, a isa DenseMultiVector ? :dense : :sparse)
    for (mask, value) in _terms(a)
        partner = full ⊻ mask
        _accumulate!(result, partner, _shuffle_sign(ga, mask, partner) * value)
    end
    return result
end

"""Inverse of `right_complement` on the whole exterior algebra."""
function right_uncomplement(a::AbstractMultiVector{T}) where T
    ga = a.algebra
    full = _full_mask(ga)
    result = _empty_mv(ga, T, a isa DenseMultiVector ? :dense : :sparse)
    for (mask, value) in _terms(a)
        partner = full ⊻ mask
        _accumulate!(result, partner, _shuffle_sign(ga, partner, mask) * value)
    end
    return result
end

function _metric_pseudoscalar(ga::GeometricAlgebra)
    n = dimension(ga)
    (isdiag(metric(ga)) || n <= 8) ||
        throw(ArgumentError("nonorthogonal metric dual is bounded to dimension 8"))
    Iblade = basisblade(ga, collect(1:n); storage=:sparse)
    q = scalarpart(Iblade * reverse(Iblade))
    iszero(q) && throw(DomainError(ga, "metric dual needs an invertible pseudoscalar"))
    return Iblade, reverse(Iblade) / q
end

"""Right metric dual `a * I⁻¹`; undefined for a degenerate metric."""
function metric_dual(a::AbstractMultiVector)
    _, inverse_I = _metric_pseudoscalar(a.algebra)
    return a * inverse_I
end

"""Inverse of `metric_dual`: right multiplication by the pseudoscalar `I`."""
function metric_undual(a::AbstractMultiVector)
    Iblade, _ = _metric_pseudoscalar(a.algebra)
    return a * Iblade
end
