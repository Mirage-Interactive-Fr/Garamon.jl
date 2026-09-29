function _addterm!(terms::Dict{K,T}, mask::K, value;
                   max_terms::Integer=typemax(Int)) where {K,T}
    iszero(value) && return terms
    result = get(terms, mask, zero(T)) + value
    if iszero(result)
        delete!(terms, mask)
    else
        terms[mask] = result
    end
    length(terms) <= max_terms ||
        throw(ArgumentError("product expansion exceeds max_terms"))
    return terms
end

function _indices(ga::GeometricAlgebra, mask::Integer)
    indices = Int[]
    remaining = mask
    while !iszero(remaining)
        push!(indices, trailing_zeros(remaining) + 1)
        remaining &= remaining - one(remaining)
    end
    return indices
end

function _shuffle_sign(ga::GeometricAlgebra, a::Integer, b::Integer)
    parity = false
    remaining = a
    while !iszero(remaining)
        bit = trailing_zeros(remaining)
        lower = (one(a) << bit) - one(a)
        isodd(_blade_grade(b & lower)) && (parity = !parity)
        remaining &= remaining - one(remaining)
    end
    return parity ? -1 : 1
end

function _left_vector!(out::Dict{K,T}, ga::GeometricAlgebra,
                       i::Int, mask::K, value::T;
                       max_terms::Integer=typemax(Int)) where {K,T}
    if !_hasbit(mask, i)
        lower = (one(K) << (i - 1)) - one(K)
        sign = isodd(_blade_grade(mask & lower)) ? -1 : 1
        _addterm!(out, mask | (one(K) << (i - 1)), sign * value; max_terms)
    end
    position = 0
    for j in _indices(ga, mask)
        position += 1
        g = metric(ga)[i, j]
        iszero(g) && continue
        sign = isodd(position - 1) ? -1 : 1
        _addterm!(out, mask ⊻ (one(K) << (j - 1)), sign * g * value; max_terms)
    end
    return out
end

# e_i ∧ A = e_i A - (e_i ⌟ A), including non-orthogonal and degenerate metrics.
function _chevalley_blade!(out::Dict{K,T}, ga::GeometricAlgebra,
                           indices::Vector{Int}, b::K, value::T;
                           max_terms::Integer=typemax(Int)) where {K,T}
    if isempty(indices)
        return _addterm!(out, b, value; max_terms)
    end
    i = first(indices)
    tail = indices[2:end]
    intermediate = Dict{K,T}()
    _chevalley_blade!(intermediate, ga, tail, b, value; max_terms)
    for (mask, coefficient) in intermediate
        _left_vector!(out, ga, i, mask, coefficient; max_terms)
    end
    for position in eachindex(tail)
        g = metric(ga)[i, tail[position]]
        iszero(g) && continue
        remaining = [tail[k] for k in eachindex(tail) if k != position]
        signed = isodd(position) ? -g : g
        _chevalley_blade!(out, ga, remaining, b, signed * value; max_terms)
    end
    return out
end

function _diagonal_blade_coefficient(ga::GeometricAlgebra,
                                     a::K, b::K, value::T) where {K,T}
    coefficient = _shuffle_sign(ga, a, b) * value
    overlap = a & b
    while !iszero(overlap)
        i = trailing_zeros(overlap) + 1
        coefficient *= metric(ga)[i, i]
        iszero(coefficient) && return coefficient
        overlap &= overlap - one(overlap)
    end
    return coefficient
end

const _PRODUCT_OPERATIONS = (:geometric, :wedge, :left, :right, :inner,
                             :dot, :scalar)

function _selected_pair_grade(operation::Symbol, ra::Integer, rb::Integer)
    operation == :geometric && return -1
    operation == :left && return ra <= rb ? rb - ra : nothing
    operation == :right && return rb <= ra ? ra - rb : nothing
    operation == :inner && return ra > 0 && rb > 0 ? abs(ra - rb) : nothing
    operation == :dot && return abs(ra - rb)
    operation == :scalar && return ra == rb ? 0 : nothing
    throw(ArgumentError("unsupported product operation"))
end

function _product(a::AbstractMultiVector{TA}, b::AbstractMultiVector{TB},
                  operation::Symbol; max_terms::Integer=1 << 16) where {TA,TB}
    max_terms >= 1 || throw(ArgumentError("max_terms must be positive"))
    operation in _PRODUCT_OPERATIONS ||
        throw(ArgumentError("unsupported product operation"))
    ga = _same_algebra(a, b)
    T = promote_type(TA, TB, eltype(metric(ga)))
    diagonal = isdiag(metric(ga))
    if a isa DenseMultiVector && b isa DenseMultiVector
        n = dimension(ga)
        n <= 20 || throw(ArgumentError("dense expansion exceeds its hard dimension limit"))
        result = DenseMultiVector(ga, zeros(T, 1 << n))
        return _product_terms!(result, a, b, ga, operation, diagonal,
                               UInt64, max_terms)
    end
    K = _masktype(a)
    result = SparseMultiVector(ga, Dict{K,T}())
    return _product_terms!(result, a, b, ga, operation, diagonal, K, max_terms)
end

# Fix the result storage and mask representation before entering the pair loop.
# Neither dimension nor storage is encoded in GeometricAlgebra's type.
function _product_terms!(result::R, a::A, b::B, ga::G,
                         operation::Symbol, diagonal::Bool, ::Type{K},
                         max_terms::Integer) where
                         {T,R<:AbstractMultiVector{T},A<:AbstractMultiVector,
                          B<:AbstractMultiVector,G<:GeometricAlgebra,K<:Integer}
    for (amask, avalue) in _terms(a), (bmask, bvalue) in _terms(b)
        ra, rb = _blade_grade(amask), _blade_grade(bmask)
        if operation == :wedge
            iszero(amask & bmask) || continue
            _accumulate!(result, amask | bmask,
                         _shuffle_sign(ga, amask, bmask) * avalue * bvalue)
            result isa SparseMultiVector && length(result.values) > max_terms &&
                throw(ArgumentError("product expansion exceeds max_terms"))
            continue
        end
        selected_grade = _selected_pair_grade(operation, ra, rb)
        selected_grade === nothing && continue
        value = convert(T, avalue * bvalue)
        if diagonal
            left, right = convert(K, amask), convert(K, bmask)
            mask = left ⊻ right
            if selected_grade == -1 || _blade_grade(mask) == selected_grade
                coefficient = _diagonal_blade_coefficient(ga, left, right, value)
                if !iszero(coefficient)
                    _accumulate!(result, mask, coefficient)
                    result isa SparseMultiVector && length(result.values) > max_terms &&
                        throw(ArgumentError("product expansion exceeds max_terms"))
                end
            end
            continue
        end
        terms = Dict{K,T}()
        _chevalley_blade!(terms, ga, _indices(ga, amask),
                          convert(K, bmask), value; max_terms)
        for (mask, coefficient) in terms
            if selected_grade == -1 || _blade_grade(mask) == selected_grade
                _accumulate!(result, mask, coefficient)
                result isa SparseMultiVector && length(result.values) > max_terms &&
                    throw(ArgumentError("product expansion exceeds max_terms"))
            end
        end
    end
    return result
end

geometric_product(a::AbstractMultiVector, b::AbstractMultiVector;
                  max_terms::Integer=1 << 16) =
    _product(a, b, :geometric; max_terms)
wedge(a::AbstractMultiVector, b::AbstractMultiVector;
      max_terms::Integer=1 << 16) = _product(a, b, :wedge; max_terms)
left_contraction(a::AbstractMultiVector, b::AbstractMultiVector;
                 max_terms::Integer=1 << 16) =
    _product(a, b, :left; max_terms)
right_contraction(a::AbstractMultiVector, b::AbstractMultiVector;
                  max_terms::Integer=1 << 16) =
    _product(a, b, :right; max_terms)
"""Hestenes inner product: grade `|r-s|` from non-scalar grade pairs only."""
inner_product(a::AbstractMultiVector, b::AbstractMultiVector;
              max_terms::Integer=1 << 16) =
    _product(a, b, :inner; max_terms)
"""Dot product: grade `|r-s|` from every grade pair, including scalars."""
dot_product(a::AbstractMultiVector, b::AbstractMultiVector;
            max_terms::Integer=1 << 16) =
    _product(a, b, :dot; max_terms)
"""Scalar part of equal-grade geometric-product pairs, as a multivector."""
scalar_product(a::AbstractMultiVector, b::AbstractMultiVector;
               max_terms::Integer=1 << 16) =
    _product(a, b, :scalar; max_terms)

"""
    product_coefficient(a, b, indices; operation=:geometric)

Return one coefficient of a binary product. For a diagonal metric this uses
the unique partner mask `B = A xor K` for requested output mask `K`,
without constructing the other output coefficients. For a general metric the
reference product is used.
"""
function _product_coefficient_diagonal(known::AbstractMultiVector,
                                       other::AbstractMultiVector,
                                       target::K, operation::Symbol,
                                       ga::GeometricAlgebra,
                                       known_on_left::Bool,
                                       ::Type{T}) where {K<:Integer,T}
    result = zero(T)
    for (known_mask, known_value) in _terms(known)
        mask = convert(K, known_mask)
        partner = mask ⊻ target
        amask = known_on_left ? mask : partner
        bmask = known_on_left ? partner : mask
        avalue = known_on_left ? known_value : coefficient_mask(other, amask)
        bvalue = known_on_left ? coefficient_mask(other, bmask) : known_value
        (iszero(avalue) || iszero(bvalue)) && continue
        ra, rb, rk = _blade_grade(amask), _blade_grade(bmask), _blade_grade(target)
        if operation == :wedge
            iszero(amask & bmask) || continue
        else
            selected_grade = _selected_pair_grade(operation, ra, rb)
            selected_grade === nothing && continue
            selected_grade == -1 || rk == selected_grade || continue
        end
        coefficient = convert(T, _shuffle_sign(ga, amask, bmask) * avalue * bvalue)
        if operation != :wedge
            overlap = amask & bmask
            while !iszero(overlap)
                i = trailing_zeros(overlap) + 1
                coefficient *= metric(ga)[i, i]
                overlap &= overlap - one(overlap)
            end
        end
        result += coefficient
    end
    return result
end

function product_coefficient(a::AbstractMultiVector{TA}, b::AbstractMultiVector{TB},
                             indices; operation::Symbol=:geometric) where {TA,TB}
    ga = _same_algebra(a, b)
    operation in _PRODUCT_OPERATIONS ||
        throw(ArgumentError("unsupported product operation"))
    target = _mask(ga, indices, _masktype(a))
    if !isdiag(metric(ga))
        return coefficient_mask(_product(a, b, operation), target)
    end
    T = promote_type(TA, TB, eltype(metric(ga)))
    if _support_size(a) <= _support_size(b)
        return _product_coefficient_diagonal(a, b, target, operation, ga, true, T)
    end
    return _product_coefficient_diagonal(b, a, target, operation, ga, false, T)
end

function _shuffle_sign_bits(a::K, b::K) where {K<:Integer}
    parity = false
    remaining = a
    while !iszero(remaining)
        bit = trailing_zeros(remaining)
        lower = (one(K) << bit) - one(K)
        isodd(_blade_grade(b & lower)) && (parity = !parity)
        remaining &= remaining - one(K)
    end
    return parity ? -1 : 1
end

function _triple_diagonal_factor(ga::GeometricAlgebra,
                                 amask::K, bmask::K, cmask::K,
                                 ::Type{T}) where {K<:Integer,T}
    middle = amask ⊻ bmask
    factor = convert(T, _shuffle_sign_bits(amask, bmask) *
                        _shuffle_sign_bits(middle, cmask))
    overlap = amask & bmask
    while !iszero(overlap)
        factor *= metric(ga)[trailing_zeros(overlap) + 1,
                             trailing_zeros(overlap) + 1]
        overlap &= overlap - one(K)
    end
    overlap = middle & cmask
    while !iszero(overlap)
        factor *= metric(ga)[trailing_zeros(overlap) + 1,
                             trailing_zeros(overlap) + 1]
        overlap &= overlap - one(K)
    end
    return factor
end

_triple_diagonal_contribution(ga::GeometricAlgebra,
                              amask::K, bmask::K, cmask::K,
                              avalue, bvalue, cvalue,
                              ::Type{T}) where {K<:Integer,T} =
    _triple_diagonal_factor(ga, amask, bmask, cmask, T) *
    avalue * bvalue * cvalue

"""
    triple_product_coefficient(a, b, c, output; max_pairs=1<<20)

Return one coefficient of `(a*b)*c` in a diagonal metric without materializing
either intermediate product. The two smallest supports are joined; the third
blade mask is uniquely determined by `A xor B xor C = output`. All matching
triples contribute, including those whose intermediate masks cancel only
after coefficient accumulation. The pair budget is checked before traversal.
"""
function triple_product_coefficient(a::AbstractMultiVector,
                                    b::AbstractMultiVector,
                                    c::AbstractMultiVector, output;
                                    max_pairs::Integer=1 << 20)
    ga = _same_algebra(a, b)
    _same_algebra(a, c)
    isdiag(metric(ga)) ||
        throw(ArgumentError("triple coefficient join requires a diagonal metric"))
    max_pairs >= 0 || throw(ArgumentError("max_pairs must be nonnegative"))
    target = _mask(ga, output)
    T = promote_type(eltype(metric(ga)), eltype(a), eltype(b), eltype(c))
    result = zero(T)
    na, nb, nc = _support_size(a), _support_size(b), _support_size(c)
    if big(na) * nb <= big(na) * nc && big(na) * nb <= big(nb) * nc
        big(na) * nb <= max_pairs ||
            throw(ArgumentError("triple coefficient join exceeds max_pairs"))
        for (amask, avalue) in _terms(a), (bmask, bvalue) in _terms(b)
            cmask = amask ⊻ bmask ⊻ target
            cvalue = coefficient_mask(c, cmask)
            iszero(cvalue) && continue
            result += _triple_diagonal_contribution(ga, amask, bmask, cmask,
                                                     avalue, bvalue, cvalue, T)
        end
    elseif big(na) * nc <= big(nb) * nc
        big(na) * nc <= max_pairs ||
            throw(ArgumentError("triple coefficient join exceeds max_pairs"))
        for (amask, avalue) in _terms(a), (cmask, cvalue) in _terms(c)
            bmask = amask ⊻ cmask ⊻ target
            bvalue = coefficient_mask(b, bmask)
            iszero(bvalue) && continue
            result += _triple_diagonal_contribution(ga, amask, bmask, cmask,
                                                     avalue, bvalue, cvalue, T)
        end
    else
        big(nb) * nc <= max_pairs ||
            throw(ArgumentError("triple coefficient join exceeds max_pairs"))
        for (bmask, bvalue) in _terms(b), (cmask, cvalue) in _terms(c)
            amask = bmask ⊻ cmask ⊻ target
            avalue = coefficient_mask(a, amask)
            iszero(avalue) && continue
            result += _triple_diagonal_contribution(ga, amask, bmask, cmask,
                                                     avalue, bvalue, cvalue, T)
        end
    end
    return result
end

"""
    quadruple_product_coefficients(a, b, c, d, outputs;
                                   max_pairs=1<<20, max_support=1<<12)

Compute requested coefficients of `((a*b)*c)*d` through the associative
factorization `(a*b)*(c*d)`. Each pair product is formed once and the final
coefficients use XOR partner joins. This can avoid the large intermediate
support of a left-associated chain when few outputs are requested. No term is
pruned by value or grade; budgets cause an error before an incomplete result
is returned. A diagonal metric is required for the final XOR join.
"""
function quadruple_product_coefficients(a::AbstractMultiVector,
                                        b::AbstractMultiVector,
                                        c::AbstractMultiVector,
                                        d::AbstractMultiVector, outputs;
                                        max_pairs::Integer=1 << 20,
                                        max_support::Integer=1 << 12)
    ga = _same_algebra(a, b)
    _same_algebra(a, c)
    _same_algebra(a, d)
    isdiag(metric(ga)) ||
        throw(ArgumentError("quadruple coefficient join requires a diagonal metric"))
    max_pairs >= 0 && max_support >= 1 ||
        throw(ArgumentError("quadruple join budgets are invalid"))
    big(_support_size(a)) * _support_size(b) <= max_pairs &&
        big(_support_size(c)) * _support_size(d) <= max_pairs ||
        throw(ArgumentError("quadruple coefficient join exceeds max_pairs"))
    requests = collect(outputs)
    T = promote_type(eltype(metric(ga)), eltype(a), eltype(b),
                     eltype(c), eltype(d))
    isempty(requests) && return T[]
    left = geometric_product(a, b; max_terms=max_support)
    right = geometric_product(c, d; max_terms=max_support)
    big(length(Set(_mask(ga, indices) for indices in requests))) *
        min(_support_size(left), _support_size(right)) <= max_pairs ||
        throw(ArgumentError("quadruple coefficient join exceeds max_pairs"))
    return product_coefficients(left, right, requests; strategy=:targeted)
end

"""
    sorted_wedge_coefficient(a, b, output)

Compute one wedge coefficient from sparse `UInt64` operands by sorting the
right support once and binary-searching complementary masks. This is an
explicit alternative to dictionary lookup for small or medium dimensions;
sorting is included in each call. Metrics may be non-diagonal because the
exterior product does not depend on them.
"""
function sorted_wedge_coefficient(a::SparseMultiVector{TA,UInt64},
                                  b::SparseMultiVector{TB,UInt64},
                                  output) where {TA,TB}
    ga = _same_algebra(a, b)
    target = _mask(ga, output)
    right = sort!(collect(pairs(b.values)); by=first)
    T = promote_type(TA, TB)
    result = zero(T)
    for (amask, avalue) in a.values
        iszero(amask & ~target) || continue
        partner = target ⊻ amask
        lo, hi = 1, length(right)
        while lo <= hi
            mid = (lo + hi) >>> 1
            if first(right[mid]) < partner
                lo = mid + 1
            else
                hi = mid - 1
            end
        end
        lo <= length(right) && first(right[lo]) == partner || continue
        sign = one(T)
        remaining = amask
        while !iszero(remaining)
            direction = trailing_zeros(remaining)
            lower = (one(UInt64) << direction) - one(UInt64)
            isodd(count_ones(partner & lower)) && (sign = -sign)
            remaining &= remaining - one(UInt64)
        end
        result += sign * avalue * last(right[lo])
    end
    return result
end

_support_size(mv::SparseMultiVector) = length(mv.values)
_support_size(mv::DenseMultiVector) = length(mv.values)

"""
    product_coefficients(a, b, requests; operation=:geometric,
                         strategy=:auto)

Return coefficients in request order. `strategy=:targeted` performs one XOR
partner join per requested mask on a diagonal metric; `:full` builds the whole
product once. `:auto` chooses by a simple upper bound on blade-pair probes,
with full evaluation for nonorthogonal metrics. Duplicate requests are allowed.
"""
function product_coefficients(a::AbstractMultiVector{TA},
                              b::AbstractMultiVector{TB}, requests;
                              operation::Symbol=:geometric,
                              strategy::Symbol=:auto) where {TA,TB}
    ga = _same_algebra(a, b)
    operation in _PRODUCT_OPERATIONS ||
        throw(ArgumentError("unsupported product operation"))
    strategy in (:auto, :targeted, :full) ||
        throw(ArgumentError("strategy must be :auto, :targeted or :full"))
    requested = collect(requests)
    masks = [_mask(ga, indices) for indices in requested]
    T = promote_type(TA, TB, eltype(metric(ga)))
    isempty(masks) && return T[]
    diagonal = isdiag(metric(ga))
    strategy == :targeted && !diagonal &&
        throw(ArgumentError("targeted XOR join requires a diagonal metric"))
    unique_count = length(Set(masks))
    if strategy == :auto
        left_count = _support_size(a)
        right_count = _support_size(b)
        strategy = diagonal && big(unique_count) * min(left_count, right_count) <=
                   big(left_count) * right_count ? :targeted : :full
    end
    if strategy == :targeted
        if unique_count == length(masks)
            return T[product_coefficient(a, b, indices; operation)
                     for indices in requested]
        end
        values = Dict{eltype(masks),T}()
        for (mask, indices) in zip(masks, requested)
            get!(values, mask) do
                product_coefficient(a, b, indices; operation)
            end
        end
        return T[values[mask] for mask in masks]
    end
    result = _product(a, b, operation)
    return T[coefficient_mask(result, mask) for mask in masks]
end

Base.:*(a::AbstractMultiVector, b::AbstractMultiVector) = geometric_product(a, b)
∧(a::AbstractMultiVector, b::AbstractMultiVector) = wedge(a, b)

function _scalar_only(a::AbstractMultiVector)
    for (mask, _) in _terms(a)
        iszero(mask) || return false
    end
    return true
end

function Base.inv(a::AbstractMultiVector{TA}) where TA
    ga = a.algebra
    reversed = reverse(a)
    left = a * reversed
    right = reversed * a
    if _scalar_only(left) && _scalar_only(right) &&
       !iszero(scalarpart(left)) && scalarpart(left) == scalarpart(right)
        return reversed / scalarpart(left)
    end

    n = dimension(ga)
    n <= 5 || throw(ArgumentError("general inverse is available only up to dimension 5"))
    T0 = promote_type(TA, eltype(metric(ga)))
    T = T0 <: Integer ? Rational{BigInt} : T0
    count = 1 << n
    left_matrix = Matrix{T}(undef, count, count)
    for col in 0:(count - 1)
        blade = multivector(ga, Dict(UInt64(col) => one(T)); storage=:dense)
        product = a * blade
        for row in 0:(count - 1)
            left_matrix[row + 1, col + 1] = coefficient_mask(product, row)
        end
    end
    unit = zeros(T, count)
    unit[1] = one(T)
    solution = try
        left_matrix \ unit
    catch error
        if error isa LinearAlgebra.SingularException ||
           error isa LinearAlgebra.ZeroPivotException
            throw(DomainError(a, "multivector is not invertible"))
        end
        rethrow()
    end
    result = DenseMultiVector(ga, solution)
    if T <: AbstractFloat || T <: Complex{<:AbstractFloat}
        forward = dense(a * result).values
        backward = dense(result * a).values
        tolerance = 100 * eps(one(real(zero(T)))) *
                    max(opnorm(left_matrix) * norm(solution), one(real(zero(T))))
        norm(forward - unit) <= tolerance && norm(backward - unit) <= tolerance ||
            throw(DomainError(a, "inverse failed residual check"))
    else
        a * result == scalar(ga, one(T); storage=:dense) &&
        result * a == scalar(ga, one(T); storage=:dense) ||
            throw(DomainError(a, "inverse failed exact check"))
    end
    return result
end
