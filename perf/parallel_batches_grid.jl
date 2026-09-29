# Sequential configuration launcher, including OS/runtime/import readiness time.
# Run only when the benchmark CPU slot has been granted.
length(ARGS) in (1,2) || error("usage: parallel_batches_grid.jl OUTPUT_DIRECTORY [--smoke]")
const GRID_OUTPUT=abspath(ARGS[1])
const GRID_SMOKE=length(ARGS)==2 && ARGS[2]=="--smoke"
mkpath(GRID_OUTPUT)
const GRID_CASES=GRID_SMOKE ? [(:threads,1),(:threads,2),(:processes,1),(:processes,2)] :
    vcat([(:threads,n) for n in (1,2,4,8,16)],[(:processes,n) for n in (1,2,4,8)])
const GRID_CONTROLLER=joinpath(@__DIR__,"controller")
const GRID_SCRIPT=joinpath(@__DIR__,"parallel_batches.jl")
const GRID_ROWS=NamedTuple[]
for (mode,lanes) in GRID_CASES
    threads=mode==:threads ? lanes : 1
    destination=joinpath(GRID_OUTPUT,"$(mode)_$(lanes).csv")
    command=`$(Base.julia_cmd()) --startup-file=no --threads=$threads,0 --gcthreads=1 --project=$GRID_CONTROLLER $GRID_SCRIPT $mode $lanes $destination`
    GRID_SMOKE && (command=`$command --smoke`)
    # GNU timeout terminates its process group, including spawned Julia workers.
    command=addenv(`timeout --signal=TERM --kill-after=20s 660s $command`,
                   "OPENBLAS_NUM_THREADS"=>"1","JULIA_NUM_THREADS"=>"$threads,0")
    started=time_ns(); ready=NaN; status="pass"
    try
        open(command,"r") do io
            while !eof(io)
                line=readline(io)
                if line=="PARALLEL_READY"
                    ready=(time_ns()-started)/1e9
                else
                    println(line)
                end
            end
        end
    catch exception
        status="failed"
        showerror(stderr,exception); println(stderr)
    end
    push!(GRID_ROWS,(;mode,lanes,status,startup_to_ready_seconds=ready,
        entire_configuration_seconds=(time_ns()-started)/1e9))
    open(joinpath(GRID_OUTPUT,"launch.csv"),"w") do io
        println(io,join(keys(first(GRID_ROWS)),','))
        foreach(row->println(io,join(values(row),',')),GRID_ROWS)
    end
    status=="pass" || error("configuration failed; review partial successful cases before continuing")
end
