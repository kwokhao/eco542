using Statistics, Distributions
using DataFrames, Combinatorics, KernelDensity
import CSV, Plots

bids = CSV.File("FPSB_data.csv") |> DataFrame |> Matrix{Float64}
v_range = 50:300

function Q1()
    Fb(bid_vec) = mean(all(bids .<= collect(bid_vec)', dims=2))
    Fb.(Iterators.product([quantile.(Ref(bids[:,col]), [.25,.75]) for col=1:3]...))
end

function Q2()
    f_est = zeros(length(v_range),3)
    for i=1:3
        m_i = maximum(bids[:,i.!=1:3], dims=2)[:,1]
        G_hat(bid) = (mask=m_i.<=bid; sum(mask)<3 ? 0 : mean(mask)*pdf(kde(bids[:,i][mask]), bid))
        g_kd = InterpKDE(kde((bids[:,i], m_i), kernel=Triweight))
        u = bids[:,i] + G_hat.(bids[:,i])./pdf.(Ref(g_kd), bids[:,i], bids[:,i])
        # I trim crazy and NaN values caused by tiny estimates of g and by G=g=0
        f_est[:,i] = pdf(kde(u[.!isnan.(u) .& (abs.(u) .<= 500)]), v_range)
    end
    Plots.plot(v_range, f_est, xlabel="bid", ylabel="f", label=[1,2,3], title="Affiliated Private Values")
end

function Q3()
    f_est = zeros(length(v_range),3)
    for i=1:3
        Gb_i(bid) = prod(mean(bids[:,i.!=1:3] .<= bid, dims=1))
        gb_kd = kde(max.([sample(bids[:,j],500000) for j=findall(i.!=1:3)]...), kernel=Triweight)
        u = bids[:,i] + Gb_i.(bids[:,i])./pdf(gb_kd, bids[:,i])
        # I trim NaN values caused by G=g=0
        f_est[:,i] = pdf(kde(u[.!isnan.(u)]), v_range)
    end
    Plots.plot(v_range, f_est, xlabel="bid", ylabel="f", label=[1,2,3], title="Independent Private Values")
end

Plots.png(Q2(), "Q1.2")
Plots.png(Q3(), "Q1.3")
