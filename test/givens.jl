using Test, Garamon, Random, LinearAlgebra, SparseArrays

function givens_matrix(n,rotations)
    T=Rational{BigInt}
    q=Matrix{T}(I,n,n)
    for (i,j,c,s) in rotations
        row_i=copy(q[i,:])
        row_j=copy(q[j,:])
        q[i,:]=c.*row_i.-s.*row_j
        q[j,:]=s.*row_i.+c.*row_j
    end
    q
end

@testset "Exact factored Givens basis change" begin
    rng=MersenneTwister(3801)
    for n in (2,3,4,5,6,8,16,32,65,129)
        ga=algebra(spdiagm(0=>ones(Rational{BigInt},n)))
        K=n<=64 ? UInt64 : n<=128 ? UInt128 : BigInt
        rotations=[(1,n,3//5,4//5)]
        n>=3 && push!(rotations,(1,2,5//13,12//13))
        plan=prepare_givens_basis_change(ga,rotations)
        masks=K[0,one(K),one(K)<<(n-1),(one(K)<<(n-1))|one(K)]
        n>=3 && push!(masks,one(K)<<1)
        q=givens_matrix(n,rotations)
        for _ in 1:4
            a=multivector(ga,Dict(mask=>BigInt(rand(rng,-7:7))
                for mask in masks);storage=:sparse)
            result=run_givens_basis_change(plan,a)
            @test result==outermorphism(q,a,ga;check_metric=true)
            @test run_givens_basis_change(plan,a)==result
        end
    end
    ga=algebra(spdiagm(0=>ones(Rational{BigInt},3)))
    a=multivector(ga,Dict(UInt64(1)=>BigInt(1));storage=:sparse)
    p=prepare_givens_basis_change(ga,[(1,2,3//5,4//5)])
    @test_throws ArgumentError prepare_givens_basis_change(ga,[(1,2,1//2,1//2)])
    @test_throws ArgumentError prepare_givens_basis_change(ga,[(2,1,3//5,4//5)])
    @test_throws ArgumentError prepare_givens_basis_change(ga,[(1,2,0.6,0.8)])
    @test_throws ArgumentError prepare_givens_basis_change(ga,[(1,2,3//5,4//5)];max_terms=0)
    @test_throws ArgumentError run_givens_basis_change(
        prepare_givens_basis_change(ga,[(1,2,3//5,4//5)];max_terms=1),a)
    floating=multivector(ga,Dict(UInt64(1)=>1.0);storage=:sparse)
    @test_throws ArgumentError run_givens_basis_change(p,floating)
    metric(ga)[1,1]=2
    @test_throws ArgumentError run_givens_basis_change(p,a)
end

