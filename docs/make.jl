using Julia4BinarySTS
using Documenter

DocMeta.setdocmeta!(Julia4BinarySTS, :DocTestSetup, :(using Julia4BinarySTS); recursive=true)

makedocs(;
    modules=[Julia4BinarySTS],
    authors="lkampoli <campoli.lorenzo@gmail.com> and contributors",
    repo="https://github.com/lkampoli/Julia4BinarySTS.jl/blob/{commit}{path}#{line}",
    sitename="Julia4BinarySTS.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://lkampoli.github.io/Julia4BinarySTS.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/lkampoli/Julia4BinarySTS.jl",
    devbranch="main",
)
