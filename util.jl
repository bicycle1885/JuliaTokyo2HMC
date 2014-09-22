# maximum likelihood estimation
function mle(samples)
    n_samples = length(samples)
    d = length(first(samples))
    μ̂ = zeros(d)
    for x in samples
        μ̂ .+= x
    end
    μ̂ ./= n_samples
    Σ̂ = zeros(d, d)
    for x in samples
        Σ̂ .+= (x - μ̂) * (x - μ̂)'
    end
    Σ̂ ./= n_samples
    μ̂, Σ̂
end

function plot_samples(filename, samples, title)
    ns = Float64[]
    xs = Float64[]
    ys = Float64[]
    for (i, s) in enumerate(samples)
        push!(ns, float64(i))
        push!(xs, s[1])
        push!(ys, s[2])
    end
    draw(SVG(filename, 6inch, 4inch),
         plot(x=xs, y=ys, color=ns, Geom.point,
              Guide.title(title), Guide.colorkey("Iteration")))
end
