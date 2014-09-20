using Gadfly

# p: (unnormalized) probability density function
# x0: initial state
# N: the number of required samples
# ϵ: step size
function metropolis(p::Function, x0::Vector{Float64}, N::Int, ϵ::Float64)
    d = length(x0)
    # allocate samples' holder
    samples = Array(typeof(x0), N)
    # set the current state to the initial state
    x = x0
    for n in 1:N
        # generate a candidate sample from
        # the proposal distribution (normal distribution)
        x_star = randn(d) * ϵ .+ x
        if rand() < min(1.0, p(x_star) / p(x))
            # accept the proposal
            x = x_star
        end
        samples[n] = x
    end
    samples
end

# Maximum Likelihood Estimation
function mle(samples)
    n_samples = length(samples)
    d = length(first(samples))
    μ = zeros(d)
    for x in samples
        μ .+= x
    end
    μ ./= n_samples
    Σ = zeros(d, d)
    for x in samples
        Σ .+= (x - μ) * (x - μ)'
    end
    Σ ./= n_samples
    μ, Σ
end

let
    # mean
    μ = [0.0, 0.0]
    # covariance matrix
    Σ = [1.0 0.8; 0.8 1.0]
    # precision matrix
    Λ = inv(Σ)
    # unnormalized multivariate normal distribution
    normal = x -> exp((-0.5 * ((x .- μ)' * Λ * (x .- μ))))[1]
    # initial state
    x0 = [0.0, 0.0]
    for ϵ in [0.1, 0.5, 1.0, 2.0]
        srand(0)
        samples = metropolis(normal, x0, 1000, ϵ)
        @show ϵ
        @show mle(samples)
        is = Float64[]
        xs = Float64[]
        ys = Float64[]
        for (i, s) in enumerate(samples)
            push!(is, float64(i))
            push!(xs, s[1])
            push!(ys, s[2])
        end
        label = replace(string(ϵ), ".", "")
        draw(SVG("metropolis.$label.svg", 6inch, 4inch),
             plot(x=xs, y=ys, color=is, Geom.point,
             Guide.title("ϵ = $ϵ"), Guide.colorkey("Iteration")))
    end
end
