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

const Δmax = 1000.0

# Algorithm 2: Naive No-U-Turn Sampler

#  L: logarithm of the joint density θ
# ∇L: gradient of L
# θ₀: initial state
#  M: number of samples
#  ϵ: step size
function nuts(L::Function, ∇L::Function, θ₀::Vector{Float64}, M::Int, ϵ::Float64)
    d = length(θ₀)
    samples = Array(typeof(θ₀), M)
    θ = θ₀
    for m in 1:M
        r₀ = randn(d)
        u = rand() * exp(L(θ) - r₀ ⋅ r₀ / 2)
        θ⁻ = θ⁺ = θ
        r⁻ = r⁺ = r₀
        C = Set([(θ, r₀)])
        j = 0
        s = 1
        while s == 1
            v = randbool() ? -1 : 1
            #@show j, v, u, length(C)
            if v == -1
                θ⁻, r⁻, _, _, C′, s′ = build_tree(L, ∇L, θ⁻, r⁻, u, v, j, ϵ)
            else
                _, _, θ⁺, r⁺, C′, s′ = build_tree(L, ∇L, θ⁺, r⁺, u, v, j, ϵ)
            end
            if s′ == 1
                C = C ∪ C′
            end
            s = s′ * ((θ⁺ - θ⁻) ⋅ r⁻ ≥ 0) * ((θ⁺ - θ⁻) ⋅ r⁺ ≥ 0)
            j += 1
        end
        θ, _ = rand(C)
        samples[m] = θ
        print_sample(θ)
    end
    samples
end

function build_tree(L::Function, ∇L::Function, θ::Vector{Float64}, r::Vector{Float64}, u::Float64, v::Int, j::Int, ϵ::Float64)
    if j == 0
        θ′, r′ = leapfrog(∇L, θ, r, v * ϵ)
        C′ = u ≤ exp(L(θ′) - r′ ⋅ r′ / 2) ? Set([(θ′, r′)]) : Set([])
        s′ = int(L(θ′) - r′ ⋅ r′ / 2 > log(u) - Δmax)
        return θ′, r′, θ′, r′, C′, s′
    else
        θ⁻, r⁻, θ⁺, r⁺, C′, s′ = build_tree(L, ∇L, θ, r, u, v, j - 1, ϵ)
        if v == -1
            θ⁻, r⁻, _, _, C″, s″ = build_tree(L, ∇L, θ⁻, r⁻, u, v, j - 1, ϵ)
        else
            _, _, θ⁺, r⁺, C″, s″ = build_tree(L, ∇L, θ⁺, r⁺, u, v, j - 1, ϵ)
        end
        s′ = s′ * s″ * ((θ⁺ - θ⁻) ⋅ r⁻ ≥ 0) * ((θ⁺ - θ⁻) ⋅ r⁺ ≥ 0)
        C′ = C′ ∪ C″
        return θ⁻, r⁻, θ⁺, r⁺, C′, s′
    end
end

function leapfrog(∇L::Function, θ::Vector{Float64}, r::Vector{Float64}, ϵ::Float64)
    r̃ = r + ϵ/2 * ∇L(θ)
    θ̃ = θ + ϵ * r̃
    r̃ = r̃ + ϵ/2 * ∇L(θ̃)
    return θ̃, r̃
end

function Base.rand(set::Set)
    n = length(set)
    t = rand(1:n)
    i = 0
    for elm in set
        i += 1
        if i == t
            return elm
        end
    end
    @assert false
end

let
    args = docopt(doc)
    # include functions
    include("norm.jl")
    M = int(args["<# of samples>"])
    for ϵ in [0.01, 0.05, 0.1, 0.5]
        srand(0)
        title = "NUTS (ϵ = $ϵ)"
        println("# $title")
        samples = nuts(ln_normal, ∇ln_normal, x₀, M, ϵ)
        if args["--plot"]
            plot_samples("$title.svg", samples, title)
        end
     end
end
