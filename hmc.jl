using Gadfly

# u:  potential energy function
# ∇u: gradient of the potential energy function
# x0: initial state
# N:  the number of required samples
# ϵ:  step size
# L:  number of steps
function hmc(u::Function, ∇u::Function, x0::Vector{Float64}, N::Int, ϵ::Float64, L::Int)
    d = length(x0)
    # allocate sampels' holder
    samples = Array(typeof(x0), N)
    # set the current sate to the initail state
    x = x0
    for n in 1:N
        p = randn(d)
        h = u(x) + 0.5 * p ⋅ p
        x_star = x
        for l in 1:L
            # half step in momentum variable
            p -= ϵ / 2 * ∇u(x_star)
            # full step in location variable
            x_star += ϵ * p
            # half step in momentum variable again
            p -= ϵ / 2 * ∇u(x_star)
        end
        h_star = u(x_star) + 0.5 * p ⋅ p
        if randn() < min(1.0, exp(h - h_star))
            # accept the proposal
            x = x_star
        end
        samples[n] = x
    end
    samples
end

let
    # mean
    μ = [0.0, 0.0]
    # covariance matrix
    Σ = [1.0 0.8; 0.8 1.0]
    # precision matrix
    Λ = inv(Σ)
    # unnormalized multivariate normal distribution
    ln_normal = x -> -0.5 * ((x .- μ)' * Λ * (x .- μ))[1]
    ∇ln_normal = x -> Λ * (x -μ)
    L = 10
    # initial state
    x0 = [0.0, 0.0]
    for ϵ in [0.01, 0.05, 0.1, 0.5]
        srand(0)
        samples = hmc(ln_normal, ∇ln_normal, x0, 1000, ϵ, L)
        is = Float64[]
        xs = Float64[]
        ys = Float64[]
        for (i, s) in enumerate(samples)
            push!(is, float64(i))
            push!(xs, s[1])
            push!(ys, s[2])
        end
        label = replace(string(ϵ), ".", "")
        draw(SVG("hmc.$label.svg", 6inch, 4inch),
             plot(x=xs, y=ys, color=is, Geom.point,
             Guide.title("ϵ = $ϵ"), Guide.colorkey("Iteration")))
     end
end
