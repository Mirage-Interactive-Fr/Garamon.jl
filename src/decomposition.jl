"""A congruence `transpose(P) * G * P = D` and its diagnostics."""
struct MetricDiagonalization{T,A<:GeometricAlgebra}
    source_metric::Matrix{T}
    source_basis::Vector{String}
    source_kind::Symbol
    orthogonal::A
    forward::Matrix{T}
    backward::Matrix{T}
    rank::Int
    residual::Float64
    condition::Float64
end

"""
    diagonalize_metric(ga; max_dimension=16, atol=0, rtol=nothing)

Diagonalize a symmetric bilinear metric by congruence, including indefinite
and degenerate forms. Columns of `forward` are the orthogonal basis vectors in
the original basis; `backward` reverses the coordinate change. Integer and
rational metrics use `Rational{BigInt}` internally. This is a bounded setup
operation, not a promise that transforming multivectors will be economical.
"""
function diagonalize_metric(ga::GeometricAlgebra;
                            max_dimension::Integer=16,
                            atol::Real=0,
                            rtol=nothing)
    n = dimension(ga)
    1 <= n <= max_dimension ||
        throw(ArgumentError("metric diagonalization exceeds max_dimension"))
    U = eltype(metric(ga))
    S = U <: Integer || U <: Rational ? Rational{BigInt} : U
    S <: Real || throw(ArgumentError("metric diagonalization currently needs real coefficients"))
    G = Matrix{S}(metric(ga))
    P = Matrix{S}(I, n, n)
    floating = S <: AbstractFloat
    relative = rtol === nothing ?
        (floating ? sqrt(eps(one(S))) : 0) : rtol
    atol >= 0 && relative >= 0 ||
        throw(ArgumentError("diagonalization tolerances must be nonnegative"))
    scale = floating ? max(opnorm(G, Inf), one(S)) : one(S)
    tolerance = atol + relative * scale
    H = copy(G)
    for k in 1:n
        pivot = nothing
        for i in k:n
            if abs(H[i, i]) > tolerance
                pivot = i
                break
            end
        end
        if pivot === nothing
            pair = nothing
            for i in k:n, j in (i + 1):n
                if abs(H[i, j]) > tolerance
                    pair = (i, j)
                    break
                end
            end
            if pair === nothing
                any(!iszero, view(H, k:n, k:n)) &&
                    throw(ArgumentError("metric has nonzero pivots below tolerance; lower atol/rtol explicitly"))
                break
            end
            i, j = pair
            P[:, i] .+= P[:, j]
            H = transpose(P) * G * P
            pivot = i
        end
        if pivot != k
            old = copy(P[:, k])
            P[:, k] = P[:, pivot]
            P[:, pivot] = old
            H = transpose(P) * G * P
        end
        q = H[k, k]
        abs(q) > tolerance ||
            throw(ArgumentError("numerical pivot is below diagonalization tolerance"))
        for j in (k + 1):n
            ratio = H[k, j] / q
            P[:, j] .-= ratio .* P[:, k]
        end
        H = transpose(P) * G * P
    end
    transformed = transpose(P) * G * P
    D = zeros(S, n, n)
    for i in 1:n
        D[i, i] = transformed[i, i]
    end
    residual = Float64(norm(transformed - D))
    bound = floating ? Float64(10 * max(tolerance, eps(one(S))) * n) : 0.0
    residual <= bound ||
        throw(ArgumentError("metric congruence left an off-diagonal residual"))
    orthogonal = algebra(D; basis=["o$i" for i in 1:n])
    inverse_P = inv(P)
    rank = count(x -> abs(x) > tolerance, diag(D))
    condition = floating ? Float64(cond(P)) : NaN
    return MetricDiagonalization(G, copy(basis(ga)), kind(ga),
                                 orthogonal, P, inverse_P, rank,
                                 residual, condition)
end

"""Map an original multivector to the congruent orthogonal basis."""
function to_orthogonal(data::MetricDiagonalization, a::AbstractMultiVector;
                       max_terms::Integer=1 << 16)
    metric(a.algebra) == data.source_metric &&
        basis(a.algebra) == data.source_basis &&
        kind(a.algebra) == data.source_kind ||
        throw(ArgumentError("multivector does not match diagonalization source"))
    return outermorphism(data.backward, a, data.orthogonal; max_terms)
end

"""Map a multivector from the orthogonal basis to the original algebra."""
function from_orthogonal(data::MetricDiagonalization,
                         a::AbstractMultiVector, target::GeometricAlgebra;
                         max_terms::Integer=1 << 16)
    dimension(a.algebra) == dimension(data.orthogonal) &&
        metric(a.algebra) == metric(data.orthogonal) &&
        basis(a.algebra) == basis(data.orthogonal) ||
        throw(ArgumentError("multivector is not in this orthogonal basis"))
    metric(target) == data.source_metric && basis(target) == data.source_basis &&
        kind(target) == data.source_kind ||
        throw(ArgumentError("target metric does not match diagonalization"))
    return outermorphism(data.forward, a, target; max_terms)
end
