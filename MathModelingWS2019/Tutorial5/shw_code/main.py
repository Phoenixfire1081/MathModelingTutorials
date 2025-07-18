#!/usr/bin/env python3
import numpy as np
import argparse
import importlib

import writer
import simulation
import models.shw as model
import plotting

default_example = "bouchut_1"
# read commandline arguments
def parse_input():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("example",  type=str, nargs="?", default=default_example, 
                        help="The name of the example which is started.")
    parser.add_argument("num_grid", type=int, nargs="?", default=-1, 
                        help="Number of grid points (not cells) used for spatial discretization.")
    parser.add_argument("num_time", type=int, nargs="?", default=-1, 
                        help="Number of times the solution should be displayed or written to output.")
    return parser.parse_args()

class froude_no:
	def __call__(self, state, time):
		print(state[0])

if __name__ == "__main__":
	
    # parse input and/or load data from example
    args = parse_input()
    ex = importlib.import_module("examples." + args.example) 
    num_grid = args.num_grid if args.num_grid>0 else ex.num_grid
    num_time = args.num_time if args.num_time>0 else ex.num_time 
    print("Running " + args.example +" with " + str(num_grid) + " grid points and " + str(num_time) + " snapshots...")

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
    sim = simulation.simulation(time, ic, solver, plotter, [writer])
    #start simulation 
    sim.start()
