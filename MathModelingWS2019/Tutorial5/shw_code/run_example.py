#!/usr/bin/env python3
import numpy as np
import importlib
import writer
import simulation
import models.shw as model
import plotting

example_to_run = "bouchut_1"

class froude_no:
	def __call__(self, state, time):
		
		f = state[1] / np.sqrt(ex.gravity * state[0])
		print(f)

# parse input and/or load data from example

ex = importlib.import_module("examples." + example_to_run) 
num_grid = ex.num_grid
num_time = ex.num_time 
print("Running " + example_to_run +" with " + str(num_grid) + " grid points and " + str(num_time) + " snapshots...")

# spatial grid
grid  = ex.grid(num_grid)
 
# output times
time  = ex.time(num_time)

# initial condition 
ic    = ex.initial(grid)

# setup solver
solver = ex.solver(grid)

# initialize output
plotter = ex.plotter(grid) 
writer  = writer.hdf5("output.hdf5", time.size, model.idx.size, grid.size)

# setup simulation
sim = simulation.simulation(time, ic, solver, plotter, [writer, froude_no()])

#start simulation 
sim.start()
