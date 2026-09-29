using Test, Garamon, CUDA, LinearAlgebra
include("resident_sum_compare.jl")
include("periodic_resident.jl")
using .PeriodicResidentPrototype

@testset "Exact four-column periodic resident sums" begin
    @test VERSION.major==1 && VERSION.minor==13
    @test Threads.nthreads()==4
    @test CUDA.functional()
    for n in (2,8,65,129),family in (:low_rank_high_grade,:higher_rank),
        signature in (:positive,:mixed,:degenerate),horizon in (1,5,32),
        repetitions in (1,8)
        fixture=compact_fixture(n,family,signature)
        signature==:mixed && @test any(<(0),fixture.diagonal)
        expected=sum_expected(fixture,horizon,repetitions)
        cpu=build_periodic_cpu(fixture,horizon)
        gpu=build_periodic_gpu(fixture,horizon)
        @test size(first(cpu.residents).source.left_values,2)==4
        @test size(first(gpu.residents).left,2)==4
        serial=periodic_sum_cpu(cpu,repetitions)
        threaded=periodic_sum_cpu(cpu,repetitions;threaded=true)
        device=periodic_sum_gpu_fused(gpu,repetitions)
        full=resident_sum_cpu(build_resident_sum_cpu(fixture,horizon),
            repetitions;threaded=true)
        @test serial==expected
        @test threaded==expected
        @test device==expected
        @test full==expected
        @test serial!==threaded
        @test serial!==device
    end
    for n in (256,1024,8192),signature in (:positive,:mixed,:degenerate)
        fixture=compact_fixture(n,:higher_rank,signature)
        signature==:mixed && @test any(<(0),fixture.diagonal)
        expected=sum_expected(fixture,32,8)
        cpu=build_periodic_cpu(fixture,32)
        gpu=build_periodic_gpu(fixture,32)
        @test periodic_sum_cpu(cpu,8;threaded=true)==expected
        @test periodic_sum_gpu_fused(gpu,8)==expected
    end
    fixture=compact_fixture(8192,:higher_rank,:positive)
    expected=sum_expected(fixture,1024,8)
    @test periodic_sum_cpu(build_periodic_cpu(fixture,1024),8;threaded=true)==expected
    @test periodic_sum_gpu_fused(build_periodic_gpu(fixture,1024),8)==expected
    @test_throws ArgumentError build_periodic_cpu(fixture,0)
    @test_throws ArgumentError build_periodic_gpu(fixture,0)
    @test_throws ArgumentError build_periodic_gpu(fixture,1024;max_bytes=1)
    @test_throws ArgumentError periodic_sum_cpu(build_periodic_cpu(fixture,1),0)
    @test_throws ArgumentError periodic_sum_gpu_fused(build_periodic_gpu(fixture,1),0)
end
