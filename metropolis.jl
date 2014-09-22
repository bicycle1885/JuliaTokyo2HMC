using Gadfly

include("util.jl")

# p:  (unnormalized) probability density function
# x0: initial state
# N:  the number of required samples
# ϵ:  step size
function metropolis(p::Function, x0::Vector{Float64}, N::Int, ϵ::Float64)
    d = length(x0)
    # allocate samples' holder
    samples = Array(typeof(x0), N)
    # set the current state to the initial state
    x = x0
    for n in 1:N
        # generate a candidate sample from
        # the proposal distribution (normal distribution)
        x̃ = randn(d) * ϵ .+ x
        if rand() < min(1.0, p(x̃) / p(x))
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
    for ϵ in [0.1, 0.5, 1.0, 2.0]
        srand(0)
        samples = metropolis(normal, x0, 1000, ϵ)
        @show ϵ
        @show mle(samples)
        filename = string("metropolis.", replace(string(ϵ), ".", ""), ".svg")
        plot_samples(filename, samples, "ϵ = $ϵ")
    end
end
