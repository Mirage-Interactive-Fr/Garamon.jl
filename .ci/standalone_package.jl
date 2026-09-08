# Standalone package CI, explicitly distinct from a pinned multi-package cohort.
using Pkg, TOML
root=dirname(@__DIR__)
meta=TOML.parsefile(joinpath(root,"Project.toml"))
out=joinpath(root,"ci-evidence"); mkpath(out)
env=joinpath(out,"environment"); Pkg.activate(env)
Pkg.develop(PackageSpec(path=root))
Pkg.instantiate()
open(joinpath(out,"scope.toml"),"w") do io
    TOML.print(io,Dict("julia"=>string(VERSION),"package"=>meta["name"],
        "scope"=>"standalone package; other dependencies resolved by compat, not the development cohort"))
end
Pkg.test(meta["name"]; julia_args=["--threads=1"],allow_reresolve=false)
