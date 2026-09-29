using Test
include("n05_compare.jl")
@testset "N05 stable grid shards" begin
    grid=n05_grid()
    @test length(grid)==36480
    @test first(grid).case_id==1 && last(grid).case_id==36480
    @test first(grid).n==2 && last(grid).n==129
    @test first(grid).strategy==:full && last(grid).strategy==:precontracted
    @test n05_grid_fingerprint(grid)==n05_grid_fingerprint(n05_grid())
    selections=[n05_select_cases(grid;shard=(i,1140)) for i in 1:1140]
    ids=vcat([getproperty.(selection,:case_id) for selection in selections]...)
    @test sort(ids)==collect(1:36480)
    @test length(unique(ids))==36480
    @test all(length(selection)==32 for selection in selections)
    @test getproperty.(n05_select_cases(grid;interval=(31,34)),:case_id)==31:34
    @test_throws ArgumentError n05_select_cases(grid;shard=(1,2))
    @test_throws ArgumentError n05_select_cases(grid;shard=(0,1140))
    @test_throws ArgumentError n05_select_cases(grid;interval=(1,33))
    @test_throws ArgumentError n05_select_cases(grid;interval=(36480,36481))
    @test n05_parse_selection(["--shard=2/1140"]).shard==(2,1140)
    @test n05_parse_selection(["--range=3:5"]).interval==(3,5)
    @test_throws ArgumentError n05_parse_selection(["--shard=2/1140","--range=3:5"])
    @test_throws ArgumentError n05_parse_selection(["--smoke","--range=3:5"])
    @test_throws ArgumentError n05_parse_selection(["--typo"])
    @test_throws ArgumentError n05_parse_selection(["--shard=2/1140","--shard=3/1140"])
    mktempdir() do root
        withenv("GARAMONBENCH_CONDITION"=>"isolated","GARAMONBENCH_INTERFERENCE_LABEL"=>"") do
            n05_shard_campaign(root;shard=(1,1140),prepare_only=true)
        end
        directory=joinpath(root,"n05-shard-0001-of-1140")
        metadata=TOML.parsefile(joinpath(directory,"n05-shard-environment.toml"))
        @test metadata["status"]=="prepared"
        @test metadata["measurement_condition"]=="isolated"
        @test metadata["grid_sha256"]==n05_grid_fingerprint(grid)
        @test length(readlines(joinpath(directory,"n05-shard-selection.csv")))==33
        @test !isfile(joinpath(directory,"n05-shard-samples.csv"))
        audit=n05_audit_shards(root)
        @test !audit.complete && audit.covered==0
        @test length(audit.missing)==36480
    end
end
