"""
    AbstractMultiVector{T}

A multivector owns an algebra descriptor and coefficients in its ordered exterior
basis. Blade indices are one-based; a mask has bit `i-1` set for basis vector `i`.
"""
abstract type AbstractMultiVector{T} end
Base.eltype(::AbstractMultiVector{T}) where T = T

struct DenseMultiVector{T,A<:GeometricAlgebra} <: AbstractMultiVector{T}
    algebra::A
    values::Vector{T}
    function DenseMultiVector(ga::A, values::Vector{T}) where {A<:GeometricAlgebra,T}
        n = dimension(ga)
        n <= 20 || throw(ArgumentError("dense storage needs an explicit small dimension"))
        length(values) == (1 << n) || throw(DimensionMismatch("dense coefficient count must be 2^dimension"))
        new{T,A}(ga, values)
    end
end

struct SparseMultiVector{T,K<:Integer,A<:GeometricAlgebra} <: AbstractMultiVector{T}
    algebra::A
    values::Dict{K,T}
    function SparseMultiVector(ga::A, values::Dict{K,T}) where {A<:GeometricAlgebra,K<:Integer,T}
        expected = _masktype(ga)
        K == expected || throw(ArgumentError("mask type does not match algebra dimension"))
        clean = Dict{K,T}()
        for (mask, value) in values
            _mask_in_bounds(ga, mask) ||
                throw(BoundsError(ga, mask))
            iszero(value) || (clean[mask] = value)
        end
        new{T,K,A}(ga, clean)
    end
end

_masktype(ga::GeometricAlgebra) = dimension(ga) <= 64 ? UInt64 :
                                   dimension(ga) <= 128 ? UInt128 : BigInt
_masktype(::DenseMultiVector) = UInt64
_masktype(::SparseMultiVector{T,K}) where {T,K} = K

function _mask_in_bounds(ga::GeometricAlgebra, mask::Integer)
    mask >= 0 || return false
    n = dimension(ga)
    if mask isa UInt64 || mask isa UInt128
        return n >= 8sizeof(mask) || iszero(mask >> n)
    end
    return iszero(BigInt(mask) >> n)
end

function _mask(ga::GeometricAlgebra, indices, ::Type{K}) where {K<:Integer}
    result = zero(K)
    for index in indices
        1 <= index <= dimension(ga) || throw(BoundsError(basis(ga), index))
        bit = one(K) << (index - 1)
        iszero(result & bit) || throw(ArgumentError("repeated basis vector"))
        result |= bit
    end
    return result
end

_mask(ga::GeometricAlgebra, indices) = _mask(ga, indices, _masktype(ga))

_hasbit(mask::Integer, i::Integer) = !iszero(mask & (one(mask) << (i - 1)))
_blade_grade(mask::UInt64) = count_ones(mask)
_blade_grade(mask::UInt128) = count_ones(mask)
_blade_grade(mask::BigInt) = count_ones(mask)

function _empty_mv(ga::GeometricAlgebra, ::Type{T}, storage::Symbol) where T
    if storage == :dense
        n = dimension(ga)
        n <= 20 || throw(ArgumentError("dense expansion exceeds its hard dimension limit"))
        return DenseMultiVector(ga, zeros(T, 1 << n))
    elseif storage == :sparse
        K = _masktype(ga)
        return SparseMultiVector(ga, Dict{K,T}())
    else
        throw(ArgumentError("storage must be :dense or :sparse"))
    end
end

_default_storage(ga::GeometricAlgebra) = dimension(ga) <= 5 ? :dense : :sparse

function multivector(ga::GeometricAlgebra, terms::AbstractDict{<:Integer,T};
                     storage::Symbol=:auto) where T
    chosen = storage == :auto ? _default_storage(ga) : storage
    result = _empty_mv(ga, T, chosen)
    for (mask, value) in terms
        _mask_in_bounds(ga, mask) ||
            throw(BoundsError(ga, mask))
        _accumulate!(result, convert(_masktype(ga), mask), value)
    end
    return result
end

function basisblade(ga::GeometricAlgebra, indices::AbstractVector{<:Integer};
                    coefficient=one(eltype(metric(ga))), storage::Symbol=:auto)
    mask = _mask(ga, indices)
    T = promote_type(eltype(metric(ga)), typeof(coefficient))
    value = convert(T, coefficient)
    return multivector(ga, Dict(mask => value); storage)
end

basisblade(ga::GeometricAlgebra, indices::Tuple; kwargs...) =
    basisblade(ga, collect(Int, indices); kwargs...)
basisvector(ga::GeometricAlgebra, index::Integer; kwargs...) =
    basisblade(ga, [index]; kwargs...)
scalar(ga::GeometricAlgebra, value::Number; storage::Symbol=:auto) =
    multivector(ga, Dict(zero(_masktype(ga)) =>
                         convert(promote_type(eltype(metric(ga)), typeof(value)), value)); storage)

function coefficient_mask(mv::DenseMultiVector, mask::Integer)
    0 <= mask < length(mv.values) || throw(BoundsError(mv.values, mask))
    return mv.values[Int(mask) + 1]
end
function coefficient_mask(mv::SparseMultiVector{T}, mask::Integer) where T
    _mask_in_bounds(mv.algebra, mask) ||
        throw(BoundsError(mv.values, mask))
    return get(mv.values, convert(_masktype(mv.algebra), mask), zero(T))
end
coefficient(mv::AbstractMultiVector, indices) =
    coefficient_mask(mv, _mask(mv.algebra, indices))
scalarpart(mv::AbstractMultiVector) = coefficient_mask(mv, 0)

function _accumulate!(mv::DenseMultiVector, mask::Integer, value)
    mv.values[Int(mask) + 1] += value
    return mv
end
function _accumulate!(mv::SparseMultiVector{T,K}, mask::Integer, value) where {T,K}
    key = convert(K, mask)
    newvalue = get(mv.values, key, zero(T)) + value
    if iszero(newvalue)
        delete!(mv.values, key)
    else
        mv.values[key] = newvalue
    end
    return mv
end

_terms(mv::DenseMultiVector) =
    ((UInt64(i - 1), value) for (i, value) in enumerate(mv.values) if !iszero(value))
_terms(mv::SparseMultiVector) = pairs(mv.values)

function sparse(mv::AbstractMultiVector{T}) where T
    K = _masktype(mv.algebra)
    return SparseMultiVector(mv.algebra,
        Dict{K,T}(convert(K, mask) => value for (mask, value) in _terms(mv)))
end

function dense(mv::AbstractMultiVector{T}; max_coefficients::Integer=1 << 16) where T
    n = dimension(mv.algebra)
    (n <= 20 && (big(1) << n) <= max_coefficients) ||
        throw(ArgumentError("dense expansion exceeds the requested coefficient budget"))
    values = zeros(T, 1 << n)
    for (mask, value) in _terms(mv)
        values[Int(mask) + 1] = value
    end
    return DenseMultiVector(mv.algebra, values)
end

function _same_algebra(a::AbstractMultiVector, b::AbstractMultiVector)
    ga, gb = a.algebra, b.algebra
    ga === gb && return ga
    dimension(ga) == dimension(gb) && metric(ga) == metric(gb) &&
        basis(ga) == basis(gb) && kind(ga) == kind(gb) ||
        throw(ArgumentError("multivectors belong to different algebras or bases"))
    return ga
end

function Base.:(==)(a::AbstractMultiVector, b::AbstractMultiVector)
    _same_algebra(a, b)
    return Dict(_terms(a)) == Dict(_terms(b))
end

function Base.:+(a::AbstractMultiVector{TA}, b::AbstractMultiVector{TB}) where {TA,TB}
    ga = _same_algebra(a, b)
    T = promote_type(TA, TB)
    storage = a isa DenseMultiVector && b isa DenseMultiVector ? :dense : :sparse
    result = _empty_mv(ga, T, storage)
    for (mask, value) in _terms(a)
        _accumulate!(result, mask, value)
    end
    for (mask, value) in _terms(b)
        _accumulate!(result, mask, value)
    end
    return result
end
Base.:-(a::AbstractMultiVector, b::AbstractMultiVector) = a + (-b)
function Base.:-(a::AbstractMultiVector{T}) where T
    result = _empty_mv(a.algebra, T, a isa DenseMultiVector ? :dense : :sparse)
    for (mask, value) in _terms(a)
        _accumulate!(result, mask, -value)
    end
    return result
end
function Base.:*(value::Number, a::AbstractMultiVector{T}) where T
    S = promote_type(T, typeof(value))
    result = _empty_mv(a.algebra, S, a isa DenseMultiVector ? :dense : :sparse)
    for (mask, coefficient) in _terms(a)
        _accumulate!(result, mask, value * coefficient)
    end
    return result
end
Base.:*(a::AbstractMultiVector, value::Number) = value * a
function Base.:/(a::AbstractMultiVector{T}, value::Number) where T
    iszero(value) && throw(DivideError())
    factor = T <: Rational ? one(T) / value :
             T <: Integer && value isa Integer ? one(T) // value : inv(value)
    return factor * a
end

function grade(a::AbstractMultiVector{T}, k::Integer) where T
    0 <= k <= dimension(a.algebra) || throw(BoundsError(a, k))
    result = _empty_mv(a.algebra, T, a isa DenseMultiVector ? :dense : :sparse)
    for (mask, value) in _terms(a)
        _blade_grade(mask) == k && _accumulate!(result, mask, value)
    end
    return result
end

function Base.reverse(a::AbstractMultiVector{T}) where T
    result = _empty_mv(a.algebra, T, a isa DenseMultiVector ? :dense : :sparse)
    for (mask, value) in _terms(a)
        r = _blade_grade(mask)
        _accumulate!(result, mask, isodd((r * (r - 1)) ÷ 2) ? -value : value)
    end
    return result
end

Base.zero(a::AbstractMultiVector{T}) where T =
    _empty_mv(a.algebra, T, a isa DenseMultiVector ? :dense : :sparse)
