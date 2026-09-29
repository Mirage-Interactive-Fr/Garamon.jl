using Test
include("n05_shared_screen.jl")
@testset "K3 selective-screen coverage" begin
    records=n05_screen_grid()
    @test length(records)==440
    @test length(unique(r.n for r in records))==11
    @test all(n->n in getproperty.(records,:n),(64,65,128,129))
    @test getproperty.(records,:case_id)==1:440
    @test all(batch->count(r->r.batch==batch,records)==20,1:22)
    @test all(n->Set(r.horizon for r in records if r.n==n)==Set((1,32)),N05_SCREEN_DIMENSIONS)
    @test all(n->Set(r.changes for r in records if r.n==n)==Set((:stable,:all)),N05_SCREEN_DIMENSIONS)
    @test all(n->Set(r.strategy for r in records if r.n==n)==Set(N05_SHARED_ROUTES),N05_SCREEN_DIMENSIONS)
    @test n05_screen_grid_hash(records)==n05_screen_grid_hash()
    state=n05_shared_fixture(4,4,4,3,:signed,:shared,:all)
    @test n05_screen_prepare_probe(state,:independent)===nothing
    @test length(n05_screen_prepare_probe(state,:workspace_unshared))==4
    @test n05_screen_prepare_probe(state,:shared_allocated) isa N05SharedPlan
    @test n05_screen_prepare_probe(state,:shared_workspace).workspace.policy==:recompute
    @test n05_screen_prepare_probe(state,:shared_cached).workspace.policy==:check_inputs
    exploratory=n05_condition(Dict("GARAMONBENCH_CONDITION"=>"exploratory_interference","GARAMONBENCH_INTERFERENCE_LABEL"=>"Etendue3D"))
    isolated=n05_condition(Dict("GARAMONBENCH_CONDITION"=>"isolated"))
    @test n05_screen_routes(exploratory)==(:independent,:shared_workspace)
    @test n05_screen_routes(isolated)==N05_SHARED_ROUTES
    @test count(r->r.strategy in n05_screen_routes(exploratory),records)==176
    @test count(r->r.strategy in N05_SCREEN_EXPLORATORY_ROUTES,records)==264
    @test_throws ArgumentError n05_screen_routes(exploratory,N05_SHARED_ROUTES)
    @test_throws ArgumentError n05_screen_routes(exploratory,(:independent,:independent))
    @test_throws ArgumentError n05_screen_routes(exploratory,(:independent,:typo))
    @test_throws ArgumentError n05_screen_routes(exploratory,(:shared_workspace,))
    withenv("GARAMONBENCH_CONDITION"=>"isolated","GARAMONBENCH_INTERFERENCE_LABEL"=>"") do
        mktempdir() do root
            n05_screen_main([root,"--prepare-only"])
            @test length(readlines(joinpath(root,"n05-shared-screen-matrix.csv")))==441
            n05_screen_main([root,"--batch=2","--prepare-only"])
            metadata=TOML.parsefile(joinpath(root,"n05-shared-screen-batch-02","n05-shared-screen-environment.toml"))
            @test metadata["case_ids"]==collect(40:-1:21)
            @test metadata["status"]=="prepared"
            @test metadata["charged_seconds"]==0
            @test metadata["wall_seconds"]==900
        end
    end
    withenv("GARAMONBENCH_CONDITION"=>"exploratory_interference","GARAMONBENCH_INTERFERENCE_LABEL"=>"Etendue3D") do
        mktempdir() do root
            n05_screen_main([root,"--batch=2","--prepare-only"])
            file=joinpath(root,"n05-shared-screen-batch-02","n05-shared-screen-environment.toml")
            metadata=TOML.parsefile(file)
            @test metadata["case_ids"]==[39,36,34,31,29,26,24,21]
            @test metadata["wall_seconds"]==300
            @test metadata["root_wall_seconds"]==1800
            @test metadata["max_cases"]==8
            @test_throws ErrorException n05_screen_budget(root,isolated,N05_SHARED_ROUTES,n05_screen_hash())
            @test_throws ErrorException n05_screen_budget(root,exploratory,N05_SCREEN_EXPLORATORY_ROUTES,n05_screen_hash())
            metadata["charged_seconds"]=1760.0
            open(io->TOML.print(io,metadata),file,"w")
            @test_throws ErrorException n05_screen_budget(root,exploratory,n05_screen_routes(exploratory),n05_screen_hash())
            budget=n05_screen_budget(root,exploratory,n05_screen_routes(exploratory),n05_screen_hash();prepare_only=true)
            @test budget.remaining==40
            @test_throws ErrorException n05_screen_campaign(root;batch=2,prepare_only=true)
        end
        mktempdir() do root
            n05_screen_main([root,"--batch=1","--prepare-only","--routes=shared_cached,independent,shared_workspace"])
            metadata=TOML.parsefile(joinpath(root,"n05-shared-screen-batch-01","n05-shared-screen-environment.toml"))
            @test length(metadata["case_ids"])==12
            @test metadata["selected_routes"]==["independent","shared_workspace","shared_cached"]
        end
    end
end
