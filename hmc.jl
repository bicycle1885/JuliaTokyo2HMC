using Gadfly

include("util.jl")

#  u : potential energy function
# ∇u : gradient of the potential energy function
# x0 : initial state
#  N : the number of required samples
#  ϵ : step size
#  L : number of steps
function hmc(u::Function, ∇u::Function, x0::Vector{Float64}, N::Int, ϵ::Float64, L::Int)
    d = length(x0)
    # allocate sampels' holder
    samples = Array(typeof(x0), N)
    # set the current sate to the initail state
    x = x0
    for n in 1:N
        p = randn(d)
        h = u(x) + p ⋅ p / 2
        x̃ = x
        for l in 1:L
            p -= ϵ / 2 * ∇u(x̃)  # half step in momentum variable
            x̃ += ϵ * p          # full step in location variable
            p -= ϵ / 2 * ∇u(x̃)  # half step in momentum variable again
        end
        h̃ = u(x̃) + p ⋅ p / 2
        if randn() < min(1.0, exp(h - h̃))
            # accept the proposal
            x = x̃
        end
        samples[n] = x
    end
    samples
end

let
    # include functions
    include("norm.jl")
    L = 10
    for ϵ in [0.01, 0.05, 0.1, 0.5]
        srand(0)
        samples = hmc(ln_normal, ∇ln_normal, x0, 1000, ϵ, L)
        filename = string("hmc.", replace(string(ϵ), ".", ""), ".svg")
        plot_samples(filename, samples, "ϵ = $ϵ, L = $L")
     end
end
