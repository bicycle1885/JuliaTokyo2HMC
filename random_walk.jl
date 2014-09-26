# Gaussian random walk
using Gadfly

let
    srand(0)
    M = 99
    x = 0.0
    xs = [x]
    println(x)
    for m in 1:M
        x += randn()
        push!(xs, x)
        println(x)
    end
    draw(SVG("random_walk.svg", 6inch, 2inch),
        plot(x=0:M, y=xs, Geom.line, Guide.xlabel("m"), Guide.ylabel("θ")))
end
