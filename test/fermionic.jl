using Test, Random, LinearAlgebra, Garamon

function fg_blade(ga,coordinates)
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

@testset "Exact number-conserving fermionic Gaussian shears" begin
    rng=MersenneTwister(3701)
    for n in (2,3,4,5,8,16,65,129)
        k=min(3,n)
        occupied=zeros(Int64,n,k)
        target=zeros(Int64,n,k)
        for j in 1:k
            occupied[j,j]=rand(rng,1:5)
            target[j,j]=rand(rng,1:5)
            if n>k
                occupied[n,j]=rand(rng,-2:2)
                target[n,j]=rand(rng,-2:2)
            end
        end
        shears=n==2 ? [(1,2,2),(2,1,-1),(1,2,3)] :
            [(1,n,2),(n,2,-1),(2,1,3)]
        factored=prepare_fermionic_gaussian(n,shears)
        materialized=prepare_fermionic_gaussian(n,shears;materialize=true)
        for _ in 1:5
            a=copy(occupied)
            a[1,1]+=rand(rng,-3:3)
            left,diagnostics=run_fermionic_gaussian(factored,a,target;
                diagnostics=true)
            right=run_fermionic_gaussian(materialized,a,target)
            @test left==right
            @test diagnostics.representation==:shears
            transformed=Garamon._fermionic_orbitals(factored,a)
            @test transformed==materialized.matrix*BigInt.(a)
            ga=algebra(Matrix{Int64}(I,n,n))
            reference=scalarpart(reverse(fg_blade(ga,target))*
                fg_blade(ga,transformed))
            @test left==reference
        end
    end
    plan=prepare_fermionic_gaussian(3,[(1,2,2)])
    empty=zeros(Int64,3,0)
    @test run_fermionic_gaussian(plan,empty,empty)==1
    @test_throws ArgumentError prepare_fermionic_gaussian(3,[(1,1,2)])
    @test_throws ArgumentError prepare_fermionic_gaussian(3,[(1,2,2)];
        max_shears=0)
    @test_throws DimensionMismatch run_fermionic_gaussian(plan,
        ones(Int64,2,1),ones(Int64,3,1))
    @test_throws ArgumentError run_fermionic_gaussian(plan,
        ones(Float64,3,1),ones(Int64,3,1))
end
