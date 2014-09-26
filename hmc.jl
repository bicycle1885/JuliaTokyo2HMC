doc = """Hamiltonian Monte Carlo (HMC) Sampler.

Usage:
    hmc.jl [options] <# of samples>

Options:
   -h --help    Show this help.
   --plot       Plot samples.
"""
using Gadfly
using DocOpt

include("util.jl")

#  U : potential energy function
# ∇U : gradient of the potential energy function
# θ₀ : initial state
#  M : number of samples
#  ϵ : step size
#  L : number of steps
function hmc(U::Function, ∇U::Function, θ₀::Vector{Float64}, M::Int, ϵ::Float64, L::Int)
    d = length(θ₀)
    # allocate sampels' holder
    samples = Array(typeof(θ₀), M)
    # set the current sate to the initail state
    θ = θ₀
    for m in 1:M
        # sample momentum variable
        p = randn(d)
        H = U(θ) + p ⋅ p / 2
        θ̃ = θ
        for l in 1:L
            p -= ϵ / 2 * ∇U(θ̃)  # half step in momentum variable
            θ̃ += ϵ * p          # full step in location variable
            p -= ϵ / 2 * ∇U(θ̃)  # half step in momentum variable again
        end
        H̃ = U(θ̃) + p ⋅ p / 2
        if randn() < min(1.0, exp(H - H̃))
            # accept the proposal
            θ = θ̃
        end
        samples[m] = θ
        print_sample(θ)
    end
    samples
end

let
    args = docopt(doc)
    # include functions
    include("norm.jl")
    M = int(args["<# of samples>"])
    for L in [1, 5, 10, 25, 50, 100], ϵ in [0.01, 0.05, 0.1, 0.5]
        srand(0)
        title = "HMC (ϵ = $ϵ, L = $L)"
        println("# $title")
        samples = hmc(x -> -ln_normal(x), x -> -∇ln_normal(x), x₀, M, ϵ, L)
        if args["--plot"]
            plot_samples("$title.svg", samples, title)
        end
    end
end
