# Recursive right-spinor matrix transform for the split Clifford algebra.
# Basis order is (e1, t1, e2, t2, ...), with e_i^2=+1 and t_i^2=-1.
# The block recurrence follows Rumyantsev, arXiv:2410.06103, Eq. (12).

struct SplitMatrixPlan{T,A<:GeometricAlgebra}
    algebra::A
    left::Matrix{T}
    max_bytes::Int
end

_split_same_algebra(a::GeometricAlgebra,b::GeometricAlgebra)=
    dimension(a)==dimension(b) && metric(a)==metric(b) &&
    basis(a)==basis(b) && kind(a)==kind(b)

function _split_matrix_contract(ga::GeometricAlgebra,::Type{T},
                                max_bytes::Int) where T
    n=dimension(ga)
    iseven(n) && 2<=n<=20 ||
        throw(ArgumentError("split transform needs an even dimension from 2 to 20"))
    pairs=n÷2
    for i in 1:n
        metric(ga)[i,i]==(isodd(i) ? 1 : -1) ||
            throw(ArgumentError("split transform needs interleaved +1,-1 metric"))
        for j in 1:i-1
            iszero(metric(ga)[i,j]) ||
                throw(ArgumentError("split transform needs orthogonal basis"))
        end
    end
    T in (BigInt,Rational{BigInt},Float32,Float64,ComplexF32,ComplexF64) ||
        throw(ArgumentError("split transform needs BigInt, Rational{BigInt}, or floating coefficients"))
    T in (BigInt,Rational{BigInt}) && pairs>5 &&
        throw(ArgumentError("exact split matrix mode is bounded to five pairs"))
    max_bytes>=1 || throw(ArgumentError("matrix byte budget must be positive"))
    side=1<<pairs
    structural=8*big(side)*side*max(sizeof(T),8)
    structural<=max_bytes ||
        throw(ArgumentError("split transform matrix buffer budget"))
    pairs
end

function _split_forward(coefficients::AbstractVector{T},pairs::Int) where T
    pairs==0 && return reshape(T[coefficients[1]],1,1)
    quarter=1<<(2pairs-2)
    a=Vector{T}(undef,quarter)
    b=Vector{T}(undef,quarter)
    c=Vector{T}(undef,quarter)
    d=Vector{T}(undef,quarter)
    for j in 0:quarter-1
        parity=isodd(count_ones(j)) ? -one(T) : one(T)
        c00=coefficients[j+1]
        c01=coefficients[j+quarter+1]
        c10=coefficients[j+2quarter+1]
        c11=coefficients[j+3quarter+1]
        a[j+1]=c00-c11
        b[j+1]=c01+c10
        c[j+1]=parity*(c01-c10)
        d[j+1]=parity*(c00+c11)
    end
    aa=_split_forward(a,pairs-1)
    bb=_split_forward(b,pairs-1)
    cc=_split_forward(c,pairs-1)
    dd=_split_forward(d,pairs-1)
    side=1<<(pairs-1)
    result=Matrix{T}(undef,2side,2side)
    result[1:side,1:side]=aa
    result[1:side,side+1:2side]=bb
    result[side+1:2side,1:side]=cc
    result[side+1:2side,side+1:2side]=dd
    result
end

_split_half(value::BigInt)=iseven(value) ? div(value,2) :
    throw(ArgumentError("inverse split transform has a nonintegral coefficient"))
_split_half(value)=value/2

function _split_inverse(matrix::AbstractMatrix{T},pairs::Int) where T
    pairs==0 && return T[matrix[1,1]]
    side=1<<(pairs-1)
    a=_split_inverse(@view(matrix[1:side,1:side]),pairs-1)
    b=_split_inverse(@view(matrix[1:side,side+1:2side]),pairs-1)
    c=_split_inverse(@view(matrix[side+1:2side,1:side]),pairs-1)
    d=_split_inverse(@view(matrix[side+1:2side,side+1:2side]),pairs-1)
    quarter=length(a)
    result=Vector{T}(undef,4quarter)
    for j in 0:quarter-1
        parity=isodd(count_ones(j)) ? -one(T) : one(T)
        result[j+1]=_split_half(a[j+1]+parity*d[j+1])
        result[j+quarter+1]=_split_half(b[j+1]+parity*c[j+1])
        result[j+2quarter+1]=_split_half(b[j+1]-parity*c[j+1])
        result[j+3quarter+1]=_split_half(parity*d[j+1]-a[j+1])
    end
    result
end

"""Convert all coefficients of an interleaved split Clifford algebra to a matrix."""
function split_matrix_transform(value::DenseMultiVector{T};
                                max_bytes::Int=512<<20) where T
    pairs=_split_matrix_contract(value.algebra,T,max_bytes)
    _split_forward(value.values,pairs)
end

"""Invert a split Clifford matrix transform, checking dimensions and budget."""
function split_matrix_inverse(ga::GeometricAlgebra,
                              matrix::AbstractMatrix{T};
                              max_bytes::Int=512<<20) where T
    pairs=_split_matrix_contract(ga,T,max_bytes)
    side=1<<pairs
    size(matrix)==(side,side) ||
        throw(DimensionMismatch("split matrix side is 2^pairs"))
    DenseMultiVector(ga,_split_inverse(matrix,pairs))
end

"""Cache a left operand's matrix representation for repeated products."""
function prepare_split_matrix(left::DenseMultiVector{T};
                              max_bytes::Int=512<<20) where T
    matrix=split_matrix_transform(left;max_bytes)
    SplitMatrixPlan{T,typeof(left.algebra)}(left.algebra,matrix,max_bytes)
end

"""Multiply through the matrix representation and restore every coefficient."""
function run_split_matrix(plan::SplitMatrixPlan{T},
                          right::DenseMultiVector{T}) where T
    _split_same_algebra(plan.algebra,right.algebra) ||
        throw(ArgumentError("split matrix algebra changed"))
    pairs=dimension(plan.algebra)÷2
    transformed=_split_forward(right.values,pairs)
    result=plan.left*transformed
    DenseMultiVector(plan.algebra,_split_inverse(result,pairs))
end

"""Return matrix side length and retained left-matrix storage."""
split_matrix_stats(plan::SplitMatrixPlan)=(
    side=size(plan.left,1),left_matrix_bytes=Base.summarysize(plan.left),
    max_bytes=plan.max_bytes)
