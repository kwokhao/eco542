using Statistics
using DataFrames, Combinatorics, KernelDensity
import CSV, Plots

bids = CSV.File("FPSB_data.csv") |> DataFrame |> Matrix{Float64}

function Q1()
    Fb(bid_vec) = mean([all(row .<= bid_vec) for row=eachrow(bids)])
    quantiles = [quantile(bids[:,col], p) for p=[.25,.75], col=1:3]
    cases = [reverse(case) for case=vec(collect(Iterators.product(fill([1,2],3)...)))]
    [Fb(quantiles) for quantiles=[quantiles[CartesianIndex.(case,1:3)] for case=cases]]
end

function Q2()
    m_i(i) = maximum(bids[:,i.!=1:3], dims=2)[:,1]
    Gb_i(bid, i) = mean(m_i(i) .<= bid)
    kde_vec = [kde(m_i(i), bandwidth=5) for i=1:3]
    g(bid, i) = pdf(kde_vec[i], bid)
    u = hcat([bids[:,i] + Gb_i.(bids[:,i], i)./g(bids[:,i], i) for i=1:3]...)
    f(bid, i) = pdf(kde(u[:,i][.!isnan.(u[:,i])], bandwidth=10), bid)
    Plots.plot(50:250, [f(50:250, i) for i=1:3], xlabel="bid", ylabel="f", label=[1,2,3], title="Affiliated Private Values")
end

function Q3()
    Fb_j(bid, j) = mean(bids[:,j] .<= bid)
    Gb_i(bid, i) = prod([Fb_j(bid, j) for j=(1:3)[i.!=1:3]])
    kde_vec = [kde(bids[:,j], bandwidth=5) for j=1:3]
    fb_j(bid, j) = pdf(kde_vec[j], bid)
    gb_i(bid, i) = prod([fb_j(bid, j) for j=(1:3)[i.!=1:3]])
    u = hcat([bids[:,i] + Gb_i.(bids[:,i], i)./gb_i.(bids[:,i], i) for i=1:3]...)
    f(bid, i) = pdf(kde(u[:,i][.!isnan.(u[:,i])], bandwidth=400), bid)
    Plots.plot(0:10000, [f(0:10000, i) for i=1:3], xlabel="bid", ylabel="f", label=[1,2,3], title="Independent Private Values")
end

Plots.png(Q2(), "Q1.2")
Plots.png(Q3(), "Q1.3")
