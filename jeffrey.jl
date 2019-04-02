using Parameters, Statistics
using DataFrames, Combinatorics
import CSV, Distances, Distributions, Plots, LinearAlgebra

bids = CSV.File("FPSB_data.csv") |> DataFrame |> Matrix{Float64}

function Q1()
    quantiles = [quantile(bids[:,col], p) for p=[.25,.75], col=1:3]
    cases = [reverse(case) for case=vec(collect(Iterators.product(fill([1,2],3)...)))]
    function FU(quantiles)
        num_less = sum([all(row .<= quantiles) for row=eachrow(bids)])
        return num_less/size(bids)[1]
    end
    [FU(quantiles) for quantiles=[quantiles[CartesianIndex.(case,1:3)] for case=cases]]
end
