using Test, Garamon, CUDA, LinearAlgebra
include("resident_sum_compare.jl")

const HIGHDIM_DIMENSIONS=(129,192,256,384,512,768,1024,2048,4096,8192)
const HIGHDIM_METRIC_STORAGE_BUDGET=256<<20
const HIGHDIM_RSS_BUDGET=6<<30

@testset "Exact resident GPU routes beyond UInt128 mask dimensions" begin
    @test VERSION.major==1 && VERSION.minor==13
    @test Threads.nthreads()==4
    @test CUDA.functional()
    for n in HIGHDIM_DIMENSIONS
        fixture=compact_fixture(n,:higher_rank,:positive)
        stored_metric=metric(first(fixture.inputs)[1].algebra)
        @test stored_metric isa Diagonal
        @test Base.summarysize(stored_metric)<=HIGHDIM_METRIC_STORAGE_BUDGET
        expected=sum_expected(fixture,32,8)
        cpu=build_resident_sum_cpu(fixture,32)
        gpu=build_resident_sum_gpu(fixture,32)
        graph=build_resident_sum_graph(gpu,8)
        parallel=resident_sum_cpu(cpu,8;threaded=true)
        fused=resident_sum_gpu_fused(gpu,8)
        captured=resident_sum_gpu_graph(graph)
        @test size(parallel)==size(expected)
        @test size(fused)==size(expected)
        @test size(captured)==size(expected)
        for i in eachindex(expected)
            @test parallel[i]==expected[i]
            @test fused[i]==expected[i]
            @test captured[i]==expected[i]
        end
        @test Sys.maxrss()<=HIGHDIM_RSS_BUDGET
    end
end
