using Garamon
using Test

include("constants.jl")
include("products.jl")
include("weighted_zdd.jl")
include("workspace.jl")
include("triplejoin.jl")
include("selector.jl")

@testset "Garamon.jl" begin
    @info "Print list of geometric algebras descriptors"
    list_geometric_algebras()

    @testset "Garamon.jl: EGAs" begin
        for (d, m) in EGAs
            ga1 = algebra(d, :ega)
            ga2 = algebra(m, :ega)
            @test metric(ga1) == metric(ga2)
            @test basis(ga1) == basis(ga2)
            @test dimension(ga1) == length(basis(ga1))
        end
    end

    @testset "Garamon.jl: CGAs" begin
        @info "Entering CGAs tests"
        for (d, m) in CGAs
            ga1 = algebra(d[1], :cga; multiplicity = d[2], object_dim = d[3])
            sdim = dimension(ga1) ÷ d[2]
            ga2 = algebra(m[1:sdim, 1:sdim], :cga; multiplicity = d[2], object_dim = d[3])
            @test metric(ga1) == metric(ga2)
            @test basis(ga1) == basis(ga2)
            @test dimension(ga1) == length(basis(ga1))
        end
    end

    @testset "Garamon.jl: PGAs" begin
        @info "Entering PGAs tests"
        for (d, m) in PGAs
            ga1 = algebra(d, :pga)
            ga2 = algebra(m, :pga)
            @test metric(ga1) == metric(ga2)
            @test basis(ga1) == basis(ga2)
            @test dimension(ga1) == length(basis(ga1))
        end
    end

    @testset "Garamon.jl: PSGAs" begin
        @info "Entering PSGAs tests"
        for (d, m) in PSGAs
            ga1 = algebra(d, :psga)
            ga2 = algebra(m, :psga)
            @test metric(ga1) == metric(ga2)
            @test basis(ga1) == basis(ga2)
            @test dimension(ga1) == length(basis(ga1))

        end
    end
end
