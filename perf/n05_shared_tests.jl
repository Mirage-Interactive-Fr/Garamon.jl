using Test
include("n05_shared_workspace.jl")

@testset "K3 shared contractions and workspace" begin
    Q=Rational{BigInt};G=Q[1 1 0;1 -1 0;0 0 0]
    U=Q[1 2 0 1;2 1 1 -1;1 0 2 1]
    chains=[[1,2,3,4],[4,3,2,1],[1,1,2,2],Int[]]
    plan=n05_shared_plan(3,4,chains)
    workspace=n05_shared_workspace(plan,G,U;policy=:check_inputs)
    @test n05_shared_owned!(workspace,G,U)==n05_shared_oracle(G,U,chains)
    @test workspace.contraction_builds==1
    old=n05_shared_owned!(workspace,G,U);saved=copy(old)
    @test workspace.contraction_builds==1
    U[1,1]+=1
    @test n05_shared_owned!(workspace,G,U)==n05_shared_oracle(G,U,chains)
    @test old==saved
    @test workspace.contraction_builds==2
    G[1,2]=G[2,1]=2
    @test n05_shared_owned!(workspace,G,U)==n05_shared_oracle(G,U,chains)
    @test workspace.contraction_builds==3
    chains[1][1]=4
    @test plan.chains[1][1]==1
    @test_throws ArgumentError n05_shared_plan(3,4,[[1]])
    @test_throws ArgumentError n05_shared_plan(3,4,[[1,5]])
    @test_throws ArgumentError n05_shared_plan(3,4,[[1,2,3,4]];max_pairs=1)
    @test_throws ArgumentError n05_shared_plan(3,4,[[1,2]];max_bytes=1)
    @test_throws ArgumentError n05_shared_workspace(plan,G,U;max_bytes=1)
    @test_throws ArgumentError n05_shared_workspace(plan,G,U;policy=:unknown)
    @test_throws DimensionMismatch n05_shared_values!(workspace,G,U[:,1:2])
    for family in (:euclidean,:signed,:null,:general),sharing in (:shared,:disjoint),changes in (:stable,:all,:one)
        state=n05_shared_fixture(4,4,4,3,family,sharing,changes)
        for strategy in N05_SHARED_ROUTES
            observed=n05_shared_episode(state,strategy;trace=true)
            @test n05_shared_qualify(state,observed.output)
            strategy==:shared_cached && changes==:stable && @test observed.builds==1
            strategy==:shared_workspace && @test observed.builds==state.horizon
        end
    end
    # The shared path does not cure N05's ill-conditioned contraction failure.
    Gf=Matrix(Diagonal([1.,1.,-1.]));Uf=reshape([1.,2.0^-30,1.],3,1)
    p=n05_shared_plan(3,1,[[1,1,1,1]]);w=n05_shared_workspace(p,Gf,Uf)
    @test only(n05_shared_oracle(Gf,Uf,p.chains))==big(1)//big(2)^120
    @test only(n05_shared_values!(w,Gf,Uf))==0.0
end
