import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import scipy.integrate as integrate


# The expected signature of 'func' and 'init_func' is very simple to keep
# 'FuncAnimation' out of your book keeping and plotting logic, but this means
# that the callable objects you pass in must know what artists they should be
# working on. There are several approaches to handling this, of varying
# complexity and encapsulation.

# The simplest approach, which works quite well in the case of a script, is to
# define the artist at a global scope and let Python sort things out.

# The second method is to use 'functools.partial' to bind artists to function.
# A third method is to use *closures* to build up the required artists and
# functions. A fourth method is to create a *class*.

# FuncAnimation arguments:
# fig -- Used to get draw, resize and any other needed events.
# func -- Callable, function to call at each frame.
# frames=None -- Passed to func, iterable/integer/generator_func/None
# init_func=None -- Called once before the first frame
# fargs=None -- Tuple of additional arguments to pass to each call to func
# save_count=None -- Number of values from frames to cache
# interval=200 -- Delay between frames in millisec
# repeat_delay=None -- Delay in millisec before repeating
# repeat=True -- Repeat when the sequence of frames is complete if True
# blit=False -- init_func and func must return an iterable of artists to be redrawn

class Test:
    def __init__(self, ax):
        self.ax = ax
        self.xdata = []
        self.ydata = []
        self.ln, = plt.plot([], [], 'ro', animated=True)

    def init(self):
        self.ax.set_xlim(0, 2 * np.pi)
        self.ax.set_ylim(-1, 1)
        return self.ln,

    def __call__(self, frame):
        self.xdata.append(frame)
        self.ydata.append(np.sin(frame))
        self.ln.set_data(self.xdata, self.ydata)
        return self.ln,


class Test2:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.xdata = []
        self.ydata = []
        self.ln, = plt.plot([], [], 'ro', animated=True)

    def init_func(self):
        self.ax.set_xlim(0, 2 * np.pi)
        self.ax.set_ylim(-1, 1)
        return self.ln,

    def func(self, frame):
        self.xdata.append(frame)
        self.ydata.append(np.sin(frame))
        self.ln.set_data(self.xdata, self.ydata)
        return self.ln,

    def __call__(self):
        out = FuncAnimation(self.fig, self.func, frames=np.linspace(0, 2 * np.pi, 128),
                            init_func=self.init_func, blit=True, interval=40)
        plt.show()


class DoublePendulum:
    def __init__(self):
        self.G = 9.8  # acceleration due to gravity, in m/s^2
        self.L1 = 1.0  # length of pendulum 1 in m
        self.L2 = 1.0  # length of pendulum 2 in m
        self.M1 = 1.0  # mass of pendulum 1 in kg
        self.M2 = 1.0  # mass of pendulum 2 in kg

        # create a time array from 0..100 sampled at 0.05 second steps
        self.dt = 0.05
        t = np.arange(0.0, 20, self.dt)

        # th1 and th2 are the initial angles (degrees)
        # w10 and w20 are the initial angular velocities (degrees per second)
        self.th1 = 120.0
        self.w1 = 0.0
        self.th2 = -10.0
        self.w2 = 0.0

        # initial state
        state = np.radians([self.th1, self.w1, self.th2, self.w2])

        # integrate your ODE using scipy.integrate.
        self.y = integrate.odeint(self.derivs, state, t)

        self.x1 = self.L1 * np.sin(self.y[:, 0])
        self.y1 = -self.L1 * np.cos(self.y[:, 0])

        self.x2 = self.L2 * np.sin(self.y[:, 2]) + self.x1
        self.y2 = -self.L2 * np.cos(self.y[:, 2]) + self.y1

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
        self.ax.set_aspect('equal')
        self.ax.grid()

        self.line, = self.ax.plot([], [], 'o-', lw=2)
        self.time_template = 'time = %.1fs'
        self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)
        self.anim = None

    def init_func(self):
        self.line.set_data([], [])
        self.time_text.set_text('')
        return self.line, self.time_text

    def func(self, i):
        thisx = [0, self.x1[i], self.x2[i]]
        thisy = [0, self.y1[i], self.y2[i]]
        self.line.set_data(thisx, thisy)
        self.time_text.set_text(self.time_template % (i * self.dt))
        return self.line, self.time_text

    def update(self):
        for i in np.arange(1, len(self.y)):
            # The input to 'func'
            yield i

    def run(self):
        # self.anim = FuncAnimation(self.fig, self.func, np.arange(1, len(self.y)),
        #                           interval=25, blit=True, init_func=self.init_func)
        self.anim = FuncAnimation(self.fig, self.func, self.update,
                                  interval=25, blit=True, init_func=self.init_func)
        plt.show()

    def derivs(self, state, t):
        dydx = np.zeros_like(state)
        dydx[0] = state[1]

        del_ = state[2] - state[0]
        den1 = (self.M1 + self.M2) * self.L1 - self.M2 * self.L1 * np.cos(del_) * np.cos(del_)
        dydx[1] = (self.M2 * self.L1 * state[1] * state[1] * np.sin(del_) * np.cos(del_) +
                   self.M2 * self.G * np.sin(state[2]) * np.cos(del_) +
                   self.M2 * self.L2 * state[3] * state[3] * np.sin(del_) -
                   (self.M1 + self.M2) * self.G * np.sin(state[0])) / den1

        dydx[2] = state[3]

        den2 = (self.L2 / self.L1) * den1
        dydx[3] = (-self.M2 * self.L2 * state[3] * state[3] * np.sin(del_) * np.cos(del_) +
                   (self.M1 + self.M2) * self.G * np.sin(state[0]) * np.cos(del_) -
                   (self.M1 + self.M2) * self.L1 * state[1] * state[1] * np.sin(del_) -
                   (self.M1 + self.M2) * self.G * np.sin(state[2])) / den2

        return dydx

def beta_pdf(x, a, b):
    return (x**(a-1) * (1-x)**(b-1) * math.gamma(a + b)
            / (math.gamma(a) * math.gamma(b)))


class UpdateDist(object):
    """Bayes update"""
    def __init__(self, ax, prob=0.5):
        self.success = 0
        self.prob = prob
        self.line, = ax.plot([], [], 'k-')
        self.x = np.linspace(0, 1, 200)
        self.ax = ax

        # Set up plot parameters
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 15)
        self.ax.grid(True)

        # This vertical line represents the theoretical value, to
        # which the plotted distribution should converge.
        self.ax.axvline(prob, linestyle='--', color='black')

    def init(self):
        self.success = 0
        self.line.set_data([], [])
        return self.line,

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        if i == 0:
            return self.init()

        # Choose success based on exceed a threshold with a uniform pick
        if np.random.rand(1,) < self.prob:
            self.success += 1
        y = beta_pdf(self.x, self.success + 1, (i - self.success) + 1)
        self.line.set_data(self.x, y)
        return self.line,


class Scope:
    def __init__(self, ax, maxt=2, dt=0.02):
        self.ax = ax
        self.dt = dt
        self.maxt = maxt
        self.tdata = [0]
        self.ydata = [0]
        self.line = Line2D(self.tdata, self.ydata)
        self.ax.add_line(self.line)
        self.ax.set_ylim(-.1, 1.1)
        self.ax.set_xlim(0, self.maxt)

    def update(self, y):
        lastt = self.tdata[-1]
        if lastt > self.tdata[0] + self.maxt:  # reset the arrays
            self.tdata = [self.tdata[-1]]
            self.ydata = [self.ydata[-1]]
            self.ax.set_xlim(self.tdata[0], self.tdata[0] + self.maxt)
            self.ax.figure.canvas.draw()

        t = self.tdata[-1] + self.dt
        self.tdata.append(t)
        self.ydata.append(y)
        self.line.set_data(self.tdata, self.ydata)
        return self.line,


def emitter(p=0.03):
    """return a random value with probability p, else 0"""
    while True:
        v = np.random.rand(1)
        if v > p:
            yield 0.
        else:
            yield np.random.rand(1)


if __name__ == '__main__':
    #
    # Script example:
    #

    # fig, ax = plt.subplots()
    # xdata, ydata = [], []
    # ln, = plt.plot([], [], 'ro', animated=True)
    #
    #
    # def init():
    #     """init_func"""
    #     ax.set_xlim(0, 2*np.pi)
    #     ax.set_ylim(-1, 1)
    #     return ln,
    #
    #
    # def update(frame):
    #     """func"""
    #     xdata.append(frame)
    #     ydata.append(np.sin(frame))
    #     ln.set_data(xdata, ydata)
    #     return ln,
    #
    #
    # ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
    #                     init_func=init, blit=True, interval=50)
    # plt.show()

    #
    # Class example1:
    #

    # fig, ax = plt.subplots()
    # test = Test(ax)
    # anim = FuncAnimation(fig, test, frames=np.linspace(0, 2*np.pi, 128),
    #                      init_func=test.init, blit=True, interval=40)
    # plt.show()

    # anim = Test2()
    # anim()

    anim = DoublePendulum()
    anim.run()

    #
    # Class example2:
    #

    # np.random.seed(19680801)  # Fixing random state for reproducibility
    #
    # fig, ax = plt.subplots()
    # ud = UpdateDist(ax, prob=0.7)
    # anim = FuncAnimation(fig, ud, frames=np.arange(200), init_func=ud.init,
    #                      interval=50, blit=True, repeat=False)
    # plt.show()

    #
    # Class example3, with generator function passed to the frames argument
    #

    # np.random.seed(19680801)  # Fixing random state for reproducibility
    #
    # fig, ax = plt.subplots()
    # scope = Scope(ax)
    #
    # # pass a generator in "emitter" to produce data for the update func
    # anim = FuncAnimation(fig, scope.update, emitter, interval=10,
    #                      blit=True)
    # plt.show()



