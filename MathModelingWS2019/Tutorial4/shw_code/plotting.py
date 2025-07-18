import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numerics.reconstruct as reconstruct
from abc import ABC, abstractmethod, abstractproperty


class base(ABC):
    def __init__(self, fig_id=1):
        self.fig = plt.figure(fig_id)

    def __call__(self, state, time):
        return self.draw(state, time)

    @abstractmethod
    def draw(self, state, time):
        pass


class empty(base):
    def draw(self, state, time):
        pass


class fvm(base):
    def __init__(self, grid,  content,  **kwargs):
        if not isinstance(content, (list,)):
            content = list(content)
        if "gui" in kwargs:
            self.fig, self.ax = plt.subplots(**kwargs["gui"])
        else:
            self.fig, self.ax = plt.subplots(len(content))
        self.ax = self.ax.ravel()
        self.fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.975])
        self.grid = grid
        self.content = content
        if "colors" in kwargs:
            self.colors = kwargs["colors"]
        else:
            self.colors = ["blue", "black", "red"]
        if "labels" in kwargs:
            self.labels = kwargs["labels"]
        else:
            self.labels = None
        if "bounds" in kwargs:
            self.bounds = kwargs["bounds"]
        else:
            self.bounds = []

        self.init(**kwargs)

    def init(self, **kwargs):
        pass

    def pre_proc(self, state, time):
        return state, state

    def draw(self, state, time):
        left, right = self.pre_proc(state, time)
        col = 0  # counter for colors
        self.fig.suptitle("Solution at time = %.3f" % time)
        for plot_idx, idx in enumerate(self.content):
            ax = self.ax[plot_idx]
            if isinstance(idx, (list,)):
                ax.clear()
                for i in idx:
                    curr_plot = self.draw_piecewise_linear_field(
                        ax, left[i], right[i])
                    curr_plot.set_color(self.colors[col])
                    col = self.next_color_idx(col)
                    self.rescale(ax, plot_idx)
            else:
                ax.clear()
                curr_plot = self.draw_piecewise_linear_field(
                    ax, left[idx], right[idx])
                curr_plot.set_color(self.colors[col])
                col = self.next_color_idx(col)
                self.rescale(ax, plot_idx)
            if self.labels is not None:
                self.ax[plot_idx].set_ylabel(self.labels[plot_idx])
            self.ax[plot_idx].set_xlabel("x")
        self.fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.975])
        return self.ax[0]

    def draw_piecewise_linear_field(self, ax, value_l, value_r):
        curr_plot = ax.add_collection(self.lines_collection(value_l, value_r))
        return curr_plot

    def lines_collection(self, left, right):
        segs = np.zeros(shape=(self.grid.size-1, 2, 2))
        segs[:, 0, 0] = self.grid[0:-1]
        segs[:, 0, 1] = left
        segs[:, 1, 0] = self.grid[1:]
        segs[:, 1, 1] = right
        return mc.LineCollection(segs)

    def rescale(self, ax, plot_idx):
        if len(self.bounds) > plot_idx and self.bounds[plot_idx] is not None:
            ax.set_xlim([self.grid[0],self.grid[-1]])
            ax.set_ylim(self.bounds[plot_idx])
        else:
            ax.autoscale_view()

    def next_color_idx(self, idx):
        return min(idx + 1, len(self.colors) - 1)


# class plotter:
#    def __init__(self,x, conservative=True):
#       self.fig, self.ax = plt.subplots(3,1)
#       self.fig.subplots_adjust(hspace = 0.25)
#       self.x = x
#       self.h,  = self.ax[0].plot([],[],'b-')
#       self.hu, = self.ax[1].plot([],[],'r:')
#       self.hv, = self.ax[2].plot([],[],'g:')
#       self.z,  = self.ax[0].plot([],[],'k-')
#       self.conservative = conservative
#       self.set_labels(conservative)
#       self.ax[0].set_xlim(np.min(x),np.max(x))
##       self.time_text = self.ax[0].text(0.02, 0.95, '', transform=self.ax[0].transAxes)
#
#
#    def set_labels(self,conservative):
#       self.ax[0].set_ylabel("height")
#       if conservative:
#           self.ax[1].set_ylabel("momentum in x")
#           self.ax[2].set_ylabel("momentum in y")
#       else:
#           self.ax[1].set_ylabel("velocity in x")
#           self.ax[2].set_ylabel("velocity in y")
#       self.ax[2].set_xlabel("x")
#
#    def __call__(self,data,time):
#       x = 0.5*(self.x[1:]+self.x[0:-1])
#       self.h.set_data(x,data[0,:]+data[3,:])
#       if self.conservative:
#           self.hu.set_data(x,data[1,:])
#           self.hv.set_data(x,data[2,:])
#       else:
#           self.hu.set_data(x,data[1,:]/data[0,:])
#           self.hv.set_data(x,data[2,:]/data[0,:])
#       self.z.set_data(x,data[3,:])
##       self.time_text.set_text("time = %.3f"%time)
#       self.ax[0].set_title("time = %.3f"%time)
#       for i in range(3):
#            self.ax[i].relim()
#            self.ax[i].autoscale_view()
#       return self.h, self.hu, self.hv, self.z
