# Dependencies
- Python 3.7
- Pythonpakete.
    - numpy 
    - matplotlib
    - h5py (hdf5 needed)

# Usage

## How to run
```main.py``` implements a simple simulation with graphical and hdf5 output.
The example simulated can be changed in ```main.py ``` by changing NAME in the line
```
9: import examples.NAME as ex
```

```bash
> python3 main.py [num_cells] [num_time_interval]
```


## Adding diagnostics

You can write your own output by introducing an additional class 
```
class OWN_CLASS:
    def __call__(self, state, time):
        ...
```
and hooking the instanciated object to the diagnostics list 

```
o = OWN_CLASS(...)
sim = simulation.simulation(grid, time, ic, solver, plotter, [writer, o])
```
