import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import pdb
class simulation:
    """
    This class provides the infrastruture to run a graphical simulation
    using matplotlib.
    """
    def __init__(self, snapshots, initial, solver, plotter, diagnostics = None):
        """ 
        Constructs instance of simulation.
        
        
        Pause on spacebar.
        Arguments:
            snapshots (numpy.array): Point in time, the simulation 
                                     evaluates, shows and/or writes 
                                     the approximate solution.

            inital (numpy.array):    State array for all cells at 
                                     initial time. 

            solver (Solver):         Solver object. 

            plotter (plotting.base): Plotting object, i.e. plotting.fvm

            diagnostics (List):      List of diagnostics, evaluated at 
                                     every snapshots.

        Returns:
            (Object):
        """
        self.current_state = initial 
        self.current_time = 0 
        self.solver =  solver
        self.plotter = plotter 
        self.anim = animation.FuncAnimation(self.plotter.fig, self, snapshots, init_func=self.init, repeat=False, blit=False)#,blit=True)
        # self.plotter.fig.canvas.mpl_connect('key_press_event', self.pause)
        self.running = True 
        if type(diagnostics) is not list:
            self.diagnostics = [diagnostics]
        else:
            self.diagnostics = diagnostics

    def init(self):
        self.current_time = 0 
        self.running = True 

    # def pause(self,event):    
        # if event.key == " " :
            # if self.running:
                # self.anim.event_source.stop()
                # self.running = False
            # else:
                # self.anim.event_source.start()
                # self.running = True
            
    def __call__(self, next_time):
        snapshot_interval = next_time-self.current_time
        self.current_state, snapshot_interval = self.solver(self.current_state, self.current_time, snapshot_interval)
        self.current_time += snapshot_interval
        for d in self.diagnostics:
            if d is not None:
                d(self.current_state,self.current_time)
        return self.plotter(self.current_state, self.current_time)

    def start(self):
        plt.show()
    
        


