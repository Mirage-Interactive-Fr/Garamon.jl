"""
    CliffordTrain

An exact binary tensor train for ordered-blade coefficients. Core `i` has
two matrices, one for each bit of basis direction `i`. The first left rank
and last right rank are one. This is useful only when ranks stay small.
"""
struct CliffordTrain{T,A<:GeometricAlgebra}
    algebra::A
    cores::Vector{NTuple{2,Matrix{T}}}
    function CliffordTrain(ga::A, cores::Vector{NTuple{2,Matrix{T}}}) where
            {T,A<:GeometricAlgebra}
        length(cores) == dimension(ga) ||
            throw(DimensionMismatch("tensor-train core count must match dimension"))
        previous = 1
        for (zero_core, one_core) in cores
            size(zero_core) == size(one_core) ||
                throw(DimensionMismatch("the two bit cores must have equal shape"))
            size(zero_core, 1) == previous && size(zero_core, 2) >= 1 ||
                throw(DimensionMismatch("tensor-train ranks do not connect"))
            previous = size(zero_core, 2)
        end
        previous == 1 ||
            throw(DimensionMismatch("last tensor-train rank must be one"))
        new{T,A}(ga, cores)
    end
end

Base.eltype(::CliffordTrain{T}) where T = T

"""Create a rank-one train from independent zero/one blade-bit weights."""
function separable_train(ga::GeometricAlgebra, zero_values::AbstractVector,
                         one_values::AbstractVector)
    n = dimension(ga)
    length(zero_values) == n && length(one_values) == n ||
        throw(DimensionMismatch("one pair of bit weights is required per direction"))
    T = promote_type(eltype(zero_values), eltype(one_values))
    T <: Number || throw(ArgumentError("tensor-train weights must be numbers"))
    cores = NTuple{2,Matrix{T}}[]
    for i in 1:n
        push!(cores, (reshape(T[zero_values[i]], 1, 1),
                      reshape(T[one_values[i]], 1, 1)))
    end
    return CliffordTrain(ga, cores)
end

"""Contract one ordered-blade coefficient without expanding the train."""
function coefficient(train::CliffordTrain{T}, indices) where T
    mask = _mask(train.algebra, indices)
    values = T[one(T)]
    for i in eachindex(train.cores)
        core = train.cores[i][_hasbit(mask, i) ? 2 : 1]
        following = zeros(T, size(core, 2))
        for row in axes(core, 1), col in axes(core, 2)
            following[col] += values[row] * core[row, col]
        end
        values = following
    end
    return only(values)
end

"""
    train_product(a, b; max_rank=256, max_entries=1<<20)

Multiply two binary trains in a diagonal Clifford metric exactly. Processing
bits from low to high carries the parity of earlier right-operand bits. The
output rank is bounded by `2*rA*rB` before any compression; this function does
not round or truncate coefficients. Both rank and allocated core entries have
explicit budgets.
"""
function train_product(a::CliffordTrain{TA}, b::CliffordTrain{TB};
                       max_rank::Integer=256,
                       max_entries::Integer=1 << 20) where {TA,TB}
    ga, gb = a.algebra, b.algebra
    dimension(ga) == dimension(gb) && metric(ga) == metric(gb) &&
        basis(ga) == basis(gb) && kind(ga) == kind(gb) ||
        throw(ArgumentError("tensor trains belong to different algebras or bases"))
    isdiag(metric(ga)) ||
        throw(ArgumentError("exact tensor-train product requires a diagonal metric"))
    max_rank >= 1 && max_entries >= 1 ||
        throw(ArgumentError("tensor-train budgets must be positive"))
    n = dimension(ga)
    ranks = Tuple{Int,Int}[]
    entries = big(0)
    for i in 1:n
        ap, an = size(a.cores[i][1])
        bp, bn = size(b.cores[i][1])
        left_rank = i == 1 ? big(1) : big(2) * ap * bp
        right_rank = i == n ? big(1) : big(2) * an * bn
        left_rank <= max_rank && right_rank <= max_rank ||
            throw(ArgumentError("tensor-train product exceeds max_rank"))
        entries += 2 * left_rank * right_rank
        entries <= max_entries ||
            throw(ArgumentError("tensor-train product exceeds max_entries"))
        push!(ranks, (Int(left_rank), Int(right_rank)))
    end
    T = promote_type(TA, TB, eltype(metric(ga)))
    cores = NTuple{2,Matrix{T}}[]
    for i in 1:n
        left_rank, right_rank = ranks[i]
        output_cores = (zeros(T, left_rank, right_rank),
                        zeros(T, left_rank, right_rank))
        ap, an = size(a.cores[i][1])
        bp, bn = size(b.cores[i][1])
        prior_parities = i == 1 ? (0,) : (0, 1)
        for p in prior_parities, abit in 0:1, bbit in 0:1
            output_bit = xor(abit, bbit)
            next_parity = xor(p, bbit)
            factor = abit == 1 && p == 1 ? -one(T) : one(T)
            if abit == 1 && bbit == 1
                factor *= metric(ga)[i, i]
            end
            iszero(factor) && continue
            acore = a.cores[i][abit + 1]
            bcore = b.cores[i][bbit + 1]
            out = output_cores[output_bit + 1]
            for ia in 1:ap, ib in 1:bp, ja in 1:an, jb in 1:bn
                row = i == 1 ? 1 : p * ap * bp + (ia - 1) * bp + ib
                col = i == n ? 1 : next_parity * an * bn + (ja - 1) * bn + jb
                out[row, col] += factor * acore[ia, ja] * bcore[ib, jb]
            end
        end
        push!(cores, output_cores)
    end
    return CliffordTrain(ga, cores)
end

"""Expand a small train exactly, refusing more than `max_terms` blade slots."""
function expand(train::CliffordTrain; max_terms::Integer=1 << 16)
    max_terms >= 1 || throw(ArgumentError("max_terms must be positive"))
    n = dimension(train.algebra)
    (big(1) << n) <= max_terms ||
        throw(ArgumentError("tensor-train expansion exceeds max_terms"))
    K = _masktype(train.algebra)
    T = eltype(train)
    terms = Dict{K,T}()
    for mask in 0:((1 << n) - 1)
        value = coefficient(train, _indices(train.algebra, mask))
        iszero(value) || (terms[K(mask)] = value)
    end
    return multivector(train.algebra, terms; storage=:sparse)
end
