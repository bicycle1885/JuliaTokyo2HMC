using Gadfly


function hmc(u, grad_u, ϵ, L, current_q)
    q = current_q
    p = randn(length(q))
    current_p = p

    p = p - ϵ .* grad_u(q) / 2

    for i in 1:L
        q += ϵ .* p
        if i != L
            p -= ϵ .* grad_u(q)
        end
    end

    p -= ϵ .* grad_u(q) / 2
    p = -p

    current_u = u(current_q)
    current_k = sum(current_p.^2) ./ 2
    proposed_u = u(q)
    proposed_k = sum(p.^2) ./ 2

    if rand() < exp(current_u-proposed_u+current_k-proposed_k)
        return q
    else
        return current_q
    end
end

let
    u = x -> (x[1] - 1.0)^2
    grad_u = x -> 2x[1]

    q = [0.0]
    samples = Float64[]
    burnin = 100
    n_samples = 1000
    for i in 1:burnin+n_samples
        q = hmc(u, grad_u, 0.005, 100, q)
        if i > burnin
            push!(samples, q[1])
        end
    end

    draw(PNG("samples.png", 6inch, 4inch), plot(x=samples, Geom.density))
    draw(PNG("trajectory.png", 6inch, 4inch), plot(x=1:n_samples, y=samples, Geom.line))
end
