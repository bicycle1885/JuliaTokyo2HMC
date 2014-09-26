doc = """Not-U-Turn Sampler (NUTS).

Usage:
    nuts.jl [options] <# of samples>

Options:
   -h --help    Show this help.
   --plot       Plot samples.
"""
using Gadfly
using DocOpt

include("util.jl")

#  p: (unnormalized) probability density function
# θ₀: initial state
#  M: number of samples
#  ϵ: step size
function metropolis(p::Function, θ₀::Vector{Float64}, M::Int, ϵ::Float64)
    d = length(θ₀)
    # allocate samples' holder
    samples = Array(typeof(θ₀), M)
    # set the current state to the initial state
    θ = θ₀
    for m in 1:M
        # generate a candidate sample from
        # the proposal distribution (normal distribution)
        θ̃ = randn(d) * ϵ + θ
        if rand() < min(1.0, p(θ̃) / p(θ))
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
    for ϵ in [0.1, 0.5, 1.0, 2.0]
        srand(0)
        title = "Metropolis (ϵ = $ϵ)"
        println("# $title")
        samples = metropolis(normal, x₀, M, ϵ)
        if args["--plot"]
            plot_samples("$title.svg", samples, title)
        end
    end
end
