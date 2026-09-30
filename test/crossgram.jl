using Test, Random, LinearAlgebra, Garamon

function cg_blade(ga,coordinates)
    K=Garamon._masktype(ga)
    blade=scalar(ga,BigInt(1);storage=:sparse)
    for j in axes(coordinates,2)
        terms=Dict{K,BigInt}()
        for i in axes(coordinates,1)
            iszero(coordinates[i,j]) ||
                (terms[one(K)<<(i-1)]=BigInt(coordinates[i,j]))
        end
        blade=wedge(blade,multivector(ga,terms;storage=:sparse))
    end
    blade
end

@testset "Exact cross-Gram rank and scalar pairing" begin
    rng=MersenneTwister(3601)
    for n in (2,3,4,5,6,8,16,32,65,129),
        signature in (:positive,:mixed,:degenerate)
        diagonal=ones(Int64,n)
        diagonal[1]=signature==:positive ? 1 :
            signature==:mixed ? -1 : 0
        ga=algebra(Matrix(Diagonal(diagonal)))
        k=min(n,3)
        left=zeros(Int64,n,k)
        right=zeros(Int64,n,k)
        for j in 1:k
            left[j,j]=rand(rng,1:5)
            right[j,j]=rand(rng,1:5)
            if n>k
                left[n,j]=rand(rng,-2:2)
                right[n,j]=rand(rng,-2:2)
            end
        end
        plan=prepare_cross_gram(ga,left,right)
        reference=scalarpart(reverse(cg_blade(ga,left))*cg_blade(ga,right))
        @test plan.determinant==reference
        @test plan.rank<=k
        @test (plan.rank<k)==iszero(plan.determinant)
        for ls in (-3,1,5),rs in (-2,1)
            actual,info=run_cross_gram(plan,ga,left,right;
                left_scale=ls,right_scale=rs,diagnostics=true)
            @test actual==BigInt(ls)*BigInt(rs)*reference
            @test !info.used_fallback
        end
        changed=copy(left)
        changed[1,1]+=1
        @test_throws ArgumentError run_cross_gram(plan,ga,changed,right)
        fallback,info=run_cross_gram(plan,ga,changed,right;
            on_invalid=:direct,diagnostics=true)
        @test info.used_fallback
        @test fallback==scalarpart(reverse(cg_blade(ga,changed))*
            cg_blade(ga,right))
    end
    ga=algebra(Int64[1 0 0;0 1 0;0 0 1])
    left=Int64[1 1;0 0;1 1]
    right=Int64[1 0;0 1;0 0]
    plan=prepare_cross_gram(ga,left,right)
    @test plan.rank==1
    @test iszero(plan.determinant)
    @test run_cross_gram(plan,ga,left,right)==0
    @test cross_gram_scalar(ga,left,right)==0
    @test_throws DimensionMismatch prepare_cross_gram(ga,ones(Int64,2,2),right)
    @test_throws DimensionMismatch prepare_cross_gram(ga,left,ones(Int64,3,1))
    @test_throws ArgumentError prepare_cross_gram(ga,Float64.(left),right)
    metric(ga)[1,1]=2
    @test_throws ArgumentError run_cross_gram(plan,ga,left,right)
    @test run_cross_gram(plan,ga,left,right;on_invalid=:direct)==
        scalarpart(reverse(cg_blade(ga,left))*cg_blade(ga,right))
    huge=fill(typemax(Int64),3,2)
    @test cross_gram_scalar(ga,huge,huge)==
        scalarpart(reverse(cg_blade(ga,huge))*cg_blade(ga,huge))
end
