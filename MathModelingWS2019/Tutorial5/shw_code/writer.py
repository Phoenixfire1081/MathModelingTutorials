import h5py

class hdf5:
    """ Very simple hdf5 output writer class. """

    def __init__(self, file_name, num_snapshots, 
                 num_variables, num_gridpoints, idx=slice(None)):
        self.file = h5py.File(file_name, "w")
        self.dset_data = self.file.create_dataset(
            "data", (num_snapshots, num_variables, num_gridpoints-1), dtype="f8")
        self.dset_time = self.file.create_dataset(
            "time", (num_snapshots,), dtype="f8")
        self.counter = 0
        self.idx = idx

    def __call__(self, state, time):
        self.dset_data[self.counter, :, :] = state[self.idx]
        self.dset_time[self.counter] = time
        self.counter += 1
