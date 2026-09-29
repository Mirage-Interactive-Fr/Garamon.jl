using Test
include("gpu_packed_compare.jl")

@testset "Pinned owned-host packed GPU output" begin
    @test CUDA.functional()
    for n in (8,65),family in (:low_rank_high_grade,:higher_rank),
        signature in (:positive,:mixed,:degenerate),horizon in (1,32,1024)
        fixture=compact_fixture(n,family,signature)
        batch=gpu_make_batch(fixture,horizon)
        resident=gpu_resident_batch(batch;max_bytes=512<<20)
        cpu=run_packed_batch(batch)
        first=gpu_owned_matrix_pinned(resident)
        second=gpu_owned_matrix_pinned(resident)
        @test first==cpu
        @test second==cpu
        @test gpu_exact_oracle(fixture,batch,first,horizon)
        @test gpu_exact_oracle(fixture,batch,second,horizon)
        @test first!==second
        @test pointer(first)!=pointer(second)
        @test_throws ArgumentError gpu_owned_matrix_pinned(resident;max_host_bytes=0)
        @test_throws ArgumentError gpu_owned_matrix_pinned(resident;
            max_host_bytes=sizeof(resident.output)-1)
        complete=gpu_complete_matrix_pinned(batch;max_bytes=512<<20)
        @test gpu_exact_oracle(fixture,batch,complete,horizon)
        @test complete!==first
    end
end
