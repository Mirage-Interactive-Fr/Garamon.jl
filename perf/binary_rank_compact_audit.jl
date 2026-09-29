using TOML, SHA

function compact_index_ranges(indices)
    values=sort!(unique(collect(indices))); result=String[]
    isempty(values) && return result
    first=last=values[1]
    for value in values[2:end]
        if value==last+1
            last=value
        else
            push!(result,first==last ? string(first) : "$first:$last")
            first=last=value
        end
    end
    push!(result,first==last ? string(first) : "$first:$last")
    result
end

"""Audit finalized dimension archives against one immutable manifest and condition.

Incomplete/running or incompatible archives are reported and never certified.
Repeated attempts are detected, not silently merged. Choose one successful archive
per dimension; retain other attempts separately. No benchmark is run by this audit.
"""
function compact_coverage_audit(manifest_path,archive_dirs)
    manifest=TOML.parsefile(manifest_path); cases=manifest["cases"]
    expected=Dict(c["execution_id"]=>c for c in cases)
    length(expected)==length(cases) || error("duplicate manifest execution IDs")
    sort([c["global_index"] for c in cases])==collect(1:length(cases)) || error("manifest indices are not exhaustive")
    planned=Dict(s["id"]=>[c for c in cases if c["shard_id"]==s["id"]] for s in manifest["shards"])
    sum(length,values(planned))==length(cases) || error("manifest partition is not exhaustive")
    issues=String[]; archives=Dict[]; started=Int[]; measured=Int[]; passed=Int[]
    for directory in archive_dirs
        path=joinpath(directory,"k1-binary-rank-protocol.toml")
        localissues=String[]
        if !isfile(path)
            push!(issues,"$directory: missing protocol"); continue
        end
        protocol=TOML.parsefile(path); shard=get(protocol,"selected_shard","")
        get(protocol,"mode","")=="shard" || push!(localissues,"not a dimension shard")
        haskey(planned,shard) || push!(localissues,"unknown shard")
        for key in ("source_sha256","measurement_condition","interference_label","output_contract","episode_contract")
            get(protocol,key,nothing)==get(manifest,key,nothing) || push!(localissues,"incompatible $key")
        end
        get(protocol,"source_unchanged",false) || push!(localissues,"source integrity not confirmed")
        get(protocol,"completed",false) || push!(localissues,"archive not finalized")
        haskey(planned,shard) && get(protocol,"cases",[])!=planned[shard] && push!(localissues,"planned cases differ")
        lists=[get(protocol,key,String[]) for key in ("started_execution_ids","measured_execution_ids","passed_execution_ids")]
        allowed=Set(c["execution_id"] for c in get(planned,shard,Dict[]))
        for (label,ids) in zip(("started","measured","passed"),lists)
            length(unique(ids))==length(ids) || push!(localissues,"duplicate $label IDs")
            issubset(Set(ids),allowed) || push!(localissues,"unknown $label IDs")
        end
        issubset(Set(lists[3]),Set(lists[2])) && issubset(Set(lists[2]),Set(lists[1])) || push!(localissues,"invalid phase subsets")
        if all(id->haskey(expected,id),lists[2])
            get(protocol,"measured_global_indices",[])==[expected[id]["global_index"] for id in lists[2]] || push!(localissues,"measured index mismatch")
        end
        summarypath=joinpath(directory,"k1-binary-rank-summary.csv")
        samplespath=joinpath(directory,"k1-binary-rank-samples.csv")
        hashes=Dict{String,String}()
        if isfile(summarypath) && isfile(samplespath)
            # This controller emits comma-free scalar fields; failure messages are sanitized.
            rows=[split(line,',') for line in Iterators.drop(eachline(summarypath),1)]
            if all(row->length(row)==27,rows)
                rowids=[row[25] for row in rows]
                Set(rowids)==allowed && length(rowids)==length(allowed) || push!(localissues,"summary coverage mismatch")
                Set(row[25] for row in rows if row[8]=="pass")==Set(lists[3]) || push!(localissues,"summary pass mismatch")
            else
                push!(localissues,"invalid summary schema")
            end
            episodeids=Set{String}()
            for line in Iterators.drop(eachline(samplespath),1)
                row=split(line,',')
                if length(row)!=17
                    push!(localissues,"invalid samples schema"); break
                end
                row[8]=="episode" && push!(episodeids,row[15])
            end
            episodeids==Set(lists[2]) || push!(localissues,"raw episode coverage mismatch")
            hashes["summary_sha256"]=bytes2hex(sha256(read(summarypath)))
            hashes["samples_sha256"]=bytes2hex(sha256(read(samplespath)))
        else
            push!(localissues,"missing raw summary or samples")
        end
        valid=isempty(localissues)
        push!(archives,Dict("directory"=>abspath(directory),"shard"=>shard,"protocol_sha256"=>bytes2hex(sha256(read(path))),
            "compatible_finalized"=>valid,"declared_started"=>length(lists[1]),"declared_measured"=>length(lists[2]),
            "declared_passed"=>length(lists[3]),"raw_hashes"=>hashes,"issues"=>localissues))
        append!(issues,["$directory: $issue" for issue in localissues])
        if valid
            for (target,ids) in zip((started,measured,passed),lists)
                append!(target,[expected[id]["global_index"] for id in ids])
            end
        end
    end
    counts=Dict{Int,Int}()
    for index in measured; counts[index]=get(counts,index,0)+1; end
    duplicates=sort([i for (i,count) in counts if count>1])
    isempty(duplicates) || push!(issues,"multiple supplied attempts measured the same global indices")
    qualified=Set(passed); missing=setdiff(Set(1:length(cases)),qualified)
    coverage=[Dict("shard"=>s["id"],"expected"=>length(planned[s["id"]]),
        "qualified"=>count(c->c["global_index"] in qualified,planned[s["id"]])) for s in manifest["shards"]]
    Dict("status"=>(!isempty(issues) ? "invalid" : isempty(missing) ? "complete" : "partial"),
        "manifest"=>abspath(manifest_path),"manifest_sha256"=>bytes2hex(sha256(read(manifest_path))),
        "auditor_sha256"=>bytes2hex(sha256(read(@__FILE__))),"source_sha256"=>manifest["source_sha256"],
        "measurement_condition"=>manifest["measurement_condition"],"interference_label"=>manifest["interference_label"],
        "expected_cases"=>length(cases),"distinct_started"=>length(unique(started)),"distinct_measured"=>length(unique(measured)),
        "distinct_qualified"=>length(qualified),"missing_cases"=>length(missing),"missing_index_ranges"=>compact_index_ranges(missing),
        "duplicate_measured_index_ranges"=>compact_index_ranges(duplicates),"incomplete_shards"=>[s["shard"] for s in coverage if s["qualified"]!=s["expected"]],
        "issues"=>issues,"archives"=>archives,"shard_coverage"=>coverage)
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)>=3 || error("usage: binary_rank_compact_audit.jl MANIFEST AUDIT_OUTPUT SHARD_DIRECTORY...")
    ispath(ARGS[2]) && error("use a fresh audit output path")
    report=compact_coverage_audit(ARGS[1],ARGS[3:end])
    open(io->TOML.print(io,report),ARGS[2],"w")
    println(report["status"],": ",report["distinct_qualified"],"/",report["expected_cases"]," qualified cases")
end
