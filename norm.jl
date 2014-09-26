# bivariate correlated normal distribution

# mean
μ = [0.0, 0.0]
# covariance matrix
Σ = [1.0 0.8;
     0.8 1.0]
# precision matrix
Λ = inv(Σ)
d = length(μ)
# unnormalized multivariate normal distribution
normal = x -> exp(-0.5 * ((x - μ)' * Λ * (x - μ))[1])
# unnormalized multivariate normal distribution
ln_normal = x -> -((x - μ)' * Λ * (x - μ))[1] / 2
# gradient of unnormalized multivariate normal distribution
∇ln_normal = x -> -(Λ * (x -μ))
# initial state
x₀ = [0.0, 0.0]

