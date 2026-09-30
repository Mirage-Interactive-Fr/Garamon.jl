using Random

function split_word_oracle(ga,a,b)
    n=dimension(ga)
    T=eltype(a)
    output=zeros(T,1<<n)
    for amask in 0:(1<<n)-1, bmask in 0:(1<<n)-1
        value=a[amask+1]*b[bmask+1]
        iszero(value) && continue
        inversions=sum((i>j for i in 1:n for j in 1:n
                        if !iszero(amask & (1<<(i-1))) &&
                           !iszero(bmask & (1<<(j-1))));init=0)
        isodd(inversions) && (value=-value)
        for i in 1:n
            if !iszero(amask & bmask & (1<<(i-1)))
                value*=metric(ga)[i,i]
            end
        end
        output[xor(amask,bmask)+1]+=value
    end
    output
end

@testset "split Clifford matrix transform and exact inverse" begin
    rng=MersenneTwister(22)
    for pairs in 1:4
        diagonal=Int64[isodd(i) ? 1 : -1 for i in 1:2pairs]
        gram=zeros(Int64,2pairs,2pairs)
        for i in eachindex(diagonal)
            gram[i,i]=diagonal[i]
        end
        ga=algebra(gram)
        coefficient_count=1<<(2pairs)
        for _ in 1:5
            left=BigInt[rand(rng,-2:2) for _ in 1:coefficient_count]
            right=BigInt[rand(rng,-2:2) for _ in 1:coefficient_count]
            a=DenseMultiVector(ga,left)
            b=DenseMultiVector(ga,right)
            transform=split_matrix_transform(a)
            @test size(transform)==(1<<pairs,1<<pairs)
            @test split_matrix_inverse(ga,transform)==a
            plan=prepare_split_matrix(a)
            result=run_split_matrix(plan,b)
            @test result.values==split_word_oracle(ga,left,right)
            @test result==geometric_product(a,b)
            @test split_matrix_transform(result)==
                transform*split_matrix_transform(b)
            @test split_matrix_stats(plan).side==1<<pairs
        end
    end
end

@testset "split matrix numeric and domain bounds" begin
    ga=algebra([1 0; 0 -1])
    a=DenseMultiVector(ga,Float64[1,2,3,4])
    b=DenseMultiVector(ga,Float64[3,1,-1,2])
    @test run_split_matrix(prepare_split_matrix(a),b)==geometric_product(a,b)
    rational=DenseMultiVector(ga,Rational{BigInt}[1//1,2//1,3//1,4//1])
    @test split_matrix_inverse(ga,split_matrix_transform(rational))==rational
    @test_throws ArgumentError split_matrix_transform(a;max_bytes=1)
    @test_throws ArgumentError split_matrix_transform(
        DenseMultiVector(ga,Int64[1,2,3,4]))
    invalid=algebra([1 0; 0 1])
    @test_throws ArgumentError split_matrix_transform(
        DenseMultiVector(invalid,Float64[1,2,3,4]))
    other=algebra([1 0; 0 -1],:none;basis=["x","y"])
    @test_throws ArgumentError run_split_matrix(prepare_split_matrix(a),
        DenseMultiVector(other,Float64[1,2,3,4]))
end
