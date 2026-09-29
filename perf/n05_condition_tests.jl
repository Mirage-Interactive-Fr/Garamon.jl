using Test
include("n05_compare.jl")

@testset "N05 measurement-condition declaration" begin
    legacy=n05_condition(Dict{String,String}())
    @test legacy.condition=="exploratory_interference"
    @test legacy.label=="Etendue3D"
    @test legacy.source=="legacy_direct_default"
    @test legacy.concurrent_external_jobs=="Etendue3D; exploratory only"
    isolated=n05_condition(Dict("GARAMONBENCH_CONDITION"=>"isolated"))
    @test isolated.condition=="isolated"
    @test isempty(isolated.label)
    @test isolated.source=="environment"
    @test !occursin("Etendue3D",isolated.concurrent_external_jobs)
    @test n05_condition(Dict("GARAMONBENCH_CONDITION"=>"exploratory_interference",
        "GARAMONBENCH_INTERFERENCE_LABEL"=>"render_job")).label=="render_job"
    @test_throws ArgumentError n05_condition(Dict("GARAMONBENCH_CONDITION"=>"unknown"))
    @test_throws ArgumentError n05_condition(Dict("GARAMONBENCH_CONDITION"=>""))
    @test_throws ArgumentError n05_condition(Dict("GARAMONBENCH_CONDITION"=>"exploratory_interference"))
    @test_throws ArgumentError n05_condition(Dict("GARAMONBENCH_CONDITION"=>"exploratory_interference",
        "GARAMONBENCH_INTERFERENCE_LABEL"=>"  "))
    @test_throws ArgumentError n05_condition(Dict("GARAMONBENCH_CONDITION"=>"isolated",
        "GARAMONBENCH_INTERFERENCE_LABEL"=>"Etendue3D"))
    @test_throws ArgumentError n05_condition(Dict("GARAMONBENCH_INTERFERENCE_LABEL"=>"x\ny"))
    @test_throws ArgumentError n05_condition(Dict("GARAMONBENCH_INTERFERENCE_LABEL"=>"x"^257))
end
