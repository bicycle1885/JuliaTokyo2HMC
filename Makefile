.PHONY: all clean

# number of samples
M=500

all: metropolis.log hmc.log nuts.log

metropolis.log:
	julia metropolis.jl --plot $M > metropolis.log

hmc.log:
	julia hmc.jl --plot $M > hmc.log

nuts.log:
	julia nuts.jl --plot $M > nuts.log

clean:
	rm *.log
