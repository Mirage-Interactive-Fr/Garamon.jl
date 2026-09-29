using Test, Garamon, CUDA, LinearAlgebra
include("binary_rank_compact_cases.jl")
include("resident_sum.jl")
using .ResidentSumPrototype

@testset "Exact reusable CUDA graphs for resident sums" begin
    @test Threads.nthreads()==4
    @test CUDA.functional()
    for n in (8,65),family in (:low_rank_high_grade,:higher_rank),
        signature in (:positive,:mixed,:degenerate),horizon in (32,1024),
        repetitions in (1,8,32)
        fixture=compact_fixture(n,family,signature)
        gpu=build_resident_sum_gpu(fixture,horizon)
        graph=build_resident_sum_graph(gpu,repetitions)
        first_result=resident_sum_gpu_graph(graph)
        second_result=resident_sum_gpu_graph(graph)
        @test first_result==second_result
        @test first_result!==second_result
        @test first_result==resident_sum_gpu(gpu,repetitions)
        @test first_result==resident_sum_gpu_fused(gpu,repetitions)
        masks=first(gpu).source.plan.output_masks
        oracles=[[rank_oracle(fixture,fixture.inputs[mod1(column+shift,4)]...,
            (:all,0)) for column in 1:4] for shift in 0:3]
        for column in 1:horizon,(row,mask) in enumerate(masks)
            expected=sum(get(oracles[mod1(step,4)][mod1(column,4)],mask,Int64(0))
                for step in 1:repetitions)
            @test first_result[row,column]==expected
            @test second_result[row,column]==expected
        end
    end
    fixture=compact_fixture(8,:low_rank_high_grade,:positive)
    gpu=build_resident_sum_gpu(fixture,32)
    @test_throws ArgumentError build_resident_sum_graph(gpu,0)
    @test_throws ArgumentError build_resident_sum_graph(gpu,32;max_nodes=32)
end
