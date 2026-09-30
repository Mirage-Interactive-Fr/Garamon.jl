const _GivensRational = Rational{BigInt}

"""An exact sequence of Euclidean plane rotations, in application order."""
struct GivensPlan{A<:GeometricAlgebra}
    algebra::A
    basis_names::Vector{String}
    rotations::Vector{Tuple{Int,Int,_GivensRational,_GivensRational}}
    max_terms::Int
end

function _givens_euclidean(ga)
    isdiag(metric(ga)) && all(metric(ga)[i,i]==1 for i in 1:dimension(ga)) ||
        throw(ArgumentError("Givens basis change requires an identity Euclidean metric"))
    nothing
end

"""
    prepare_givens_basis_change(ga, rotations; max_terms=65536)

Each rotation is `(i,j,c,s)`, with `i<j`, exact rational `c^2+s^2=1`,
`e_i -> c*e_i+s*e_j`, and `e_j -> -s*e_i+c*e_j`. No matrix of
exterior-power coefficients is materialized.
"""
function prepare_givens_basis_change(ga::GeometricAlgebra, rotations;
                                     max_terms::Integer=1<<16)
    _givens_euclidean(ga)
    1<=max_terms<=typemax(Int) || throw(ArgumentError("invalid Givens term budget"))
    saved=Tuple{Int,Int,_GivensRational,_GivensRational}[]
    for rotation in rotations
        length(rotation)==4 || throw(ArgumentError("Givens rotation needs four fields"))
        i,j,c0,s0=rotation
        i isa Integer && j isa Integer && 1<=i<j<=dimension(ga) ||
            throw(ArgumentError("invalid Givens plane"))
        c0 isa Rational || c0 isa Integer ||
            throw(ArgumentError("Givens cosine must be exact rational"))
        s0 isa Rational || s0 isa Integer ||
            throw(ArgumentError("Givens sine must be exact rational"))
        c=_GivensRational(c0)
        s=_GivensRational(s0)
        c*c+s*s==1 || throw(ArgumentError("Givens rotation is not orthogonal"))
        push!(saved,(Int(i),Int(j),c,s))
    end
    GivensPlan(ga,copy(basis(ga)),saved,Int(max_terms))
end

"""Apply a prepared exact Givens outermorphism to a sparse or dense multivector."""
function run_givens_basis_change(plan::GivensPlan,a::AbstractMultiVector)
    ga=a.algebra
    dimension(ga)==dimension(plan.algebra) &&
        kind(ga)==kind(plan.algebra) && basis(ga)==plan.basis_names ||
        throw(ArgumentError("Givens plan belongs to another algebra"))
    _givens_euclidean(ga)
    eltype(a)<:Integer || eltype(a)<:Rational ||
        throw(ArgumentError("Givens input coefficients must be exact rational"))
    K=_masktype(ga)
    current=Dict{K,_GivensRational}()
    for (mask,value) in _terms(a)
        _addterm!(current,mask,_GivensRational(value);max_terms=plan.max_terms)
    end
    for (i,j,c,s) in plan.rotations
        bit_i=one(K)<<(i-1)
        bit_j=one(K)<<(j-1)
        between=((one(K)<<(j-i-1))-one(K))<<i
        next=Dict{K,_GivensRational}()
        for (mask,value) in current
            has_i=!iszero(mask&bit_i)
            has_j=!iszero(mask&bit_j)
            if has_i==has_j
                _addterm!(next,mask,value;max_terms=plan.max_terms)
            else
                _addterm!(next,mask,c*value;max_terms=plan.max_terms)
                sign=isodd(count_ones(mask&between)) ? -1 : 1
                factor=has_i ? s : -s
                _addterm!(next,mask⊻bit_i⊻bit_j,
                          sign*factor*value;max_terms=plan.max_terms)
            end
        end
        current=next
    end
    SparseMultiVector(ga,current)
end

