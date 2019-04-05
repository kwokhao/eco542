using Statistics, Distributions
using DataFrames, Combinatorics, KernelDensity, Dierckx, Roots
import CSV, Plots

bids = CSV.File("FPSB_data.csv") |> DataFrame |> Matrix{Float64}
v_range = 50:300

function Q11()
    Fb(bid_vec) = mean(all(bids .<= collect(bid_vec)', dims=2))
    Fb.(Iterators.product([quantile.(Ref(bids[:,col]), [.25,.75]) for col=1:3]...))
end

function Q12()
    f_est = zeros(length(v_range),3)
    for i=1:3
        m_i = maximum(bids[:,i.!=1:3], dims=2)[:,1]
        G_hat(bid) = (mask=m_i.<=bid; sum(mask)<3 ? 0 : mean(mask)*pdf(kde(bids[:,i][mask]), bid))
        g_kd = InterpKDE(kde((bids[:,i], m_i), kernel=Triweight))
        u = bids[:,i] + G_hat.(bids[:,i])./pdf.(Ref(g_kd), bids[:,i], bids[:,i])
        # Trim crazy and NaN values caused by tiny estimates of g and by G=g=0
        f_est[:,i] = pdf(kde(u[.!isnan.(u) .& (abs.(u) .<= 500)]), v_range)
    end
    Plots.plot(v_range, f_est, xlabel="bid", ylabel="f", label=[1,2,3], title="Affiliated Private Values")
end

function Q13()
    f_est = zeros(length(v_range),3)
    for i=1:3
        Gb_i(bid) = prod(mean(bids[:,i.!=1:3] .<= bid, dims=1))
        gb_kd = kde(max.([sample(bids[:,j],500000) for j=findall(i.!=1:3)]...), kernel=Triweight)
        u = bids[:,i] + Gb_i.(bids[:,i])./pdf(gb_kd, bids[:,i])
        # Trim NaN values caused by G=g=0
        f_est[:,i] = pdf(kde(u[.!isnan.(u)]), v_range)
    end
    Plots.plot(v_range, f_est, xlabel="bid", ylabel="f", label=[1,2,3], title="Independent Private Values")
end

Plots.png(Q12(), "Q1.2")
Plots.png(Q13(), "Q1.3")

price, winner, bidder_num = CSV.File("Ascending_data.csv") |> DataFrame |> Matrix{Float64} |> eachcol |> (x->collect.(x))

price_range = 100:300

function Q21()
    quantiles = [find_zero(q -> 4q^3-3q^4-mean(price.<=p), .5) for p=price_range]
    G_spl = Spline1D(price_range, quantiles)
    G(u) = evaluate(G_spl, u)
    # Draw a representative sample of values from the CDF G, then estimate the
    # PDF g by kernel density. This is basically soft differentiation.
    rep_samp = [find_zero(u -> q-G(u), 150) for q=0:.01:1]
    f_est = pdf(kde(rep_samp), price_range)
    Plots.plot(price_range, f_est, xlabel="u", ylabel="f(u)", legend=nothing, title="Ascending Auction")
end

Plots.png(Q21(), "Q2.1")
