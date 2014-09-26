.PHONY: all clean install

# number of samples
M=500

all: metropolis.log hmc.log nuts.log random_walk.log

metropolis.log:
	julia metropolis.jl --plot $M > metropolis.log

hmc.log:
	julia hmc.jl --plot $M > hmc.log

nuts.log:
	julia nuts.jl --plot $M > nuts.log

random_walk.log:
	julia random_walk.jl > random_walk.log

clean:
	rm *.log
	rm *.svg

install:
	cp *.svg images/
