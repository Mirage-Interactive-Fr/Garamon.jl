# Included inside the temporary GaramonLifecycle package by the controller.
using Garamon, LinearAlgebra, StaticArrays

function lifecycle_fixture(index::Int)
    if index == 0
        ga = algebra(3, :ega)
        left = UInt64[0, 1, 2, 3, 4, 5, 6, 7]
        right = copy(left)
    else
        # The request creates both metric and supports at runtime. All dynamic
        # requests have the same Julia types but different values/topologies.
        diagonal = ntuple(i -> isodd(index >> (i-1)) ? -1.0 : 1.0, 4)
        ga = algebra(SMatrix{4,4}(Diagonal(collect(diagonal))))
        left = sort!(unique(UInt64[mod(3index + 5j, 16) for j in 0:(2 + mod(index, 4))]))
        right = sort!(unique(UInt64[mod(7index + 3j, 16) for j in 0:(2 + mod(index+1, 4))]))
    end
    a = multivector(ga, Dict(mask => Float64(1 + mod(j, 3))
        for (j, mask) in enumerate(left)); storage=:sparse)
    b = multivector(ga, Dict(mask => Float64(1 + mod(2j, 3))
        for (j, mask) in enumerate(right)); storage=:sparse)
    (; a, b, left, right, index)
end

lifecycle_plan(fixture) = prepare_product(fixture.a, fixture.b; max_paths=256)
function lifecycle_artifact(plan, fixture, strategy::Symbol)
    strategy == :prepared && return plan
    strategy == :generated && return generate_product(plan; max_paths=256)
    strategy == :workspace && return ProductWorkspace(plan, fixture.a, fixture.b; max_bytes=1<<20)
    error("unknown strategy")
end
function lifecycle_execute(artifact, fixture, strategy::Symbol)
    strategy == :prepared && return run_product(artifact, fixture.a, fixture.b)
    strategy == :generated && return run_generated_product(artifact, fixture.a, fixture.b)
    strategy == :workspace && return run_product!(artifact, fixture.a, fixture.b)
    error("unknown strategy")
end

function lifecycle_update!(fixture, t)
    for (j, mask) in enumerate(fixture.left)
        fixture.a.values[mask] = Float64(1 + mod(t+j, 3))
    end
    for (j, mask) in enumerate(fixture.right)
        fixture.b.values[mask] = Float64(1 + mod(t+2j, 3))
    end
end

function lifecycle_trace(fixtures, artifacts, strategy::Symbol, horizon::Int)
    output = zeros(Float64, 16, horizon)
    for t in 1:horizon
        i = 1 + mod(t-1, length(fixtures))
        fixture = fixtures[i]
        # Advance across visits too: a 12-context cycle must not resonate with
        # the three-value coefficient period and freeze each context's values.
        phase = t + fld(t-1, length(fixtures))
        lifecycle_update!(fixture, phase)
        value = lifecycle_execute(artifacts[i], fixture, strategy)
        # Copy each full output immediately: workspace results are aliases.
        for (mask, coefficient) in value.values
            output[Int(mask)+1, t] = coefficient
        end
    end
    output
end

# Independent integer oracle; no Garamon product or blade-sign routine.
function lifecycle_reference(fixture, t)
    output = zeros(Float64, 16)
    d = Int.(diag(metric(fixture.a.algebra)))
    for (ai, a) in enumerate(fixture.left), (bi, b) in enumerate(fixture.right)
        inversions = 0
        factor = 1
        for i in eachindex(d)
            ((a >> (i-1)) & 1) == 0 && continue
            for j in 1:i-1
                inversions += Int((b >> (j-1)) & 1)
            end
            ((b >> (i-1)) & 1) == 1 && (factor *= d[i])
        end
        isodd(inversions) && (factor = -factor)
        av = 1 + mod(t+ai, 3)
        bv = 1 + mod(t+2bi, 3)
        output[Int(xor(a,b))+1] += factor * av * bv
    end
    output
end

function lifecycle_validate(fixtures, output)
    for t in axes(output, 2)
        fixture = fixtures[1 + mod(t-1, length(fixtures))]
        phase = t + fld(t-1, length(fixtures))
        output[:, t] == lifecycle_reference(fixture, phase) || error("independent oracle failed at $t")
    end
    true
end

function lifecycle_catalog()
    # A finite EGA3 full-support catalogue. No dynamic 4D request is trained.
    for strategy in (:prepared, :generated, :workspace)
        fixture = lifecycle_fixture(0)
        plan = lifecycle_plan(fixture)
        artifact = lifecycle_artifact(plan, fixture, strategy)
        lifecycle_execute(artifact, fixture, strategy)
        lifecycle_trace(Any[fixture], Any[artifact], strategy, 3)
    end
    nothing
end
