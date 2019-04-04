using Statistics, Distributions
using DataFrames, Combinatorics, KernelDensity, Dierckx
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
    u = zeros(1000,3)
    for i=1:3
        function G_hat_i(bid)
            m_i_mask = m_i(i) .<= bid
            sum(m_i_mask) < 3 && (return 0)
            mean(m_i_mask)*pdf(kde(bids[:,i][m_i_mask]), bid)
        end
        g_hat_kde = kde((bids[:,i], m_i(i)), bandwidth=(5,5))
        g_hat_i(bid) = pdf(g_hat_kde, bid, bid)
        u[:,i] = bids[:,i] + G_hat_i.(bids[:,i])./g_hat_i.(bids[:,i])
    end
    f_kde_vec = [kde(u[:,i][(.!isnan.(u[:,i]))]) for i=1:3]
    f(bid, i) = pdf(f_kde_vec[i], bid)
    Plots.plot(50:.1:300, [f(50:.1:300, i) for i=1:3], xlabel="bid", ylabel="f", label=[1,2,3], title="Affiliated Private Values")
end

function Q3()
    Fb_j(bid, j) = mean(bids[:,j] .<= bid)
    u = zeros(1000,3)
    for i=1:3
        Gb_i(bid) = prod([Fb_j(bid, j) for j=(1:3)[i.!=1:3]])
        m_samp = maximum(hcat([sample(bids[:,j],50000) for j=(1:3)[i!=1:3]]...), dims=2)[:,1]
        gb_kde = kde(m_samp)
        u[:,i] = bids[:,i] + Gb_i.(bids[:,i])./pdf(gb_kde, bids[:,i])
    end
    f(bid, i) = pdf(kde(u[:,i][.!isnan.(u[:,i])]), bid)
    Plots.plot(0:300, [f(0:300, i) for i=1:3], xlabel="bid", ylabel="f", label=[1,2,3], title="Independent Private Values")
end

function Q3_alt()
    m_i(i) = maximum(bids[:,i.!=1:3], dims=2)[:,1]
    Gb_i(bid, i) = mean(m_i(i) .<= bid)
    kde_vec = [kde(m_i(i)) for i=1:3]
    g(bid, i) = pdf(kde_vec[i], bid)
    u = hcat([bids[:,i] + Gb_i.(bids[:,i], i)./g(bids[:,i], i) for i=1:3]...)
    f(bid, i) = pdf(kde(u[:,i][.!isnan.(u[:,i])]), bid)
    Plots.plot(50:250, [f(50:250, i) for i=1:3], xlabel="bid", ylabel="f", label=[1,2,3], title="Independent Private Values: Alternative Method")
end

Plots.png(Q2(), "Q1.2")
Plots.png(Q3(), "Q1.3")
Plots.png(Q3_alt(), "Q1.3_alt")
