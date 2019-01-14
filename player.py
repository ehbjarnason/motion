import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import numpy as np


class AnimationDescriptor:
    def __init__(self):
        self.fig = Figure(figsize=(12, 6), dpi=72)
        self.frames = None
        self.fargs = None
        self.save_count = None
        self.interval = 200
        self.repeat = False
        self.repeat_delay = None
        self.blit = False


class AnimationTest(AnimationDescriptor):
    def __init__(self):
        AnimationDescriptor.__init__(self)
        self.frames = np.linspace(0, 2 * np.pi, 128)
        self.ax = self.fig.gca()
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


class PlayerApp(tk.Tk, FuncAnimation):
    def __init__(self, anim, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        FuncAnimation.__init__(self, anim.fig, anim, frames=anim.frames,
                               init_func=anim.init, fargs=anim.fargs,
                               save_count=anim.save_count, interval=anim.interval,
                               repeat=anim.repeat, repeat_delay=anim.repeat_delay,
                               blit=False)

        # container = tk.Frame(self)
        # container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # container.grid_rowconfigure(0, weight=1)
        # container.grid_columnconfigure(0, weight=1)

        # self.fig = Figure(figsize=(12, 6), dpi=72)
        # figsize: with, height in inches (25.4mm)
        # dpi: number of pixels per inch

        # self.frames = {}

        # menu = MenuFrame(container, self)
        # self.frames[MenuFrame] = menu
        # menu.grid(row=0, column=0, sticky="nsew")

        # frame = AnimationFrame(container, self, fig)
        # self.frames[AnimationFrame] = frame
        # frame.grid(row=0, column=0, sticky="nsew")

        frame_anim = tk.Frame(self)

        # player = FuncAnimationPlayer(frame_anim, self.fig, None)
        # self.frames[FuncAnimationPlayer] = frame_anim

        frame_anim.grid(row=0, column=0, sticky="nsew")

        # self.show_frame(MenuFrame)
        # self.show_frame(AnimationFrame)
        # self.show_frame(FuncAnimationPlayer)

        canvas = FigureCanvasTkAgg(anim.fig, frame_anim)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, frame_anim)
        toolbar.update()

        self.scale = ttk.Scale(frame_anim, orient=tk.HORIZONTAL)
        self.scale.pack(fill=tk.BOTH)
        self.button_play = ttk.Button(frame_anim, text="Play", command=self.on_button_play)
        self.button_play.pack(side=tk.LEFT)
        self.button_stop = ttk.Button(frame_anim, text="Stop")
        self.button_stop.pack(side=tk.LEFT)
        self.button_back = ttk.Button(frame_anim, text="< Back")
        self.button_back.pack(side=tk.LEFT)
        self.button_forward = ttk.Button(frame_anim, text="Forward >")
        self.button_forward.pack(side=tk.LEFT)
        self.label_step = ttk.Label(frame_anim, text='0.000/0.000 sec, 0/0 frames')
        self.label_step.pack(side=tk.LEFT, padx=5)
        self.label_stepsize = ttk.Label(frame_anim, text="Step size: Frames: ")
        self.label_stepsize.pack(side=tk.LEFT)
        self.entryvar_steps_frames = tk.StringVar()
        self.entryvar_steps_frames.set("0")
        self.entry_stepsize_frame = ttk.Entry(frame_anim, width=6,
                                              textvariable=self.entryvar_steps_frames)
        self.entry_stepsize_frame.pack(side=tk.LEFT)
        self.label_stepsize_sec = ttk.Label(frame_anim, text=" Sec: ")
        self.label_stepsize_sec.pack(side=tk.LEFT)
        self.entryvar_steps_sec = tk.StringVar()
        self.entryvar_steps_sec.set("0.000")
        self.entry_stepsize_sec = ttk.Entry(frame_anim, width=6,
                                            textvariable=self.entryvar_steps_sec)
        self.entry_stepsize_sec.pack(side=tk.LEFT)

    def on_button_play(self):
        if self.button_play.instate(("!selected",)):
            self.button_play.state(("selected",))
            self.button_play.config(text="Pause")
        else:
            self.button_play.state(("!selected",))
            self.button_play.config(text="Play")

# class MenuFrame(tk.Frame):
#     def __init__(self, parent, controller):
#         tk.Frame.__init__(self, parent)


class FuncAnimationPlayer(FuncAnimation):
    def __init__(self, parent, figure, func=None, frames=None, init_func=None, fargs=None,
                 save_count=None, **kwargs):
        self.fig = figure
        self.func = func
        self.frames = frames

        if self.func is not None:
            FuncAnimation.__init__(self, self.fig, self.func, frames=self.play(),
                                   init_func=init_func, fargs=fargs,
                                   save_count=save_count, **kwargs)

        canvas = FigureCanvasTkAgg(figure, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()

        self.scale = ttk.Scale(parent, orient=tk.HORIZONTAL)
        self.scale.pack(fill=tk.BOTH)
        self.button_play = ttk.Button(parent, text="Play", command=self.on_button_play)
        self.button_play.pack(side=tk.LEFT)
        self.button_stop = ttk.Button(parent, text="Stop")
        self.button_stop.pack(side=tk.LEFT)
        self.button_back = ttk.Button(parent, text="< Back")
        self.button_back.pack(side=tk.LEFT)
        self.button_forward = ttk.Button(parent, text="Forward >")
        self.button_forward.pack(side=tk.LEFT)
        self.label_step = ttk.Label(parent, text='0.000/0.000 sec, 0/0 frames')
        self.label_step.pack(side=tk.LEFT, padx=5)
        self.label_stepsize = ttk.Label(parent, text="Step size: Frames: ")
        self.label_stepsize.pack(side=tk.LEFT)
        self.entryvar_steps_frames = tk.StringVar()
        self.entryvar_steps_frames.set("0")
        self. entry_stepsize_frame = ttk.Entry(parent, width=6,
                                               textvariable=self.entryvar_steps_frames)
        self.entry_stepsize_frame.pack(side=tk.LEFT)
        self.label_stepsize_sec = ttk.Label(parent, text=" Sec: ")
        self.label_stepsize_sec.pack(side=tk.LEFT)
        self.entryvar_steps_sec = tk.StringVar()
        self.entryvar_steps_sec.set("0.000")
        self.entry_stepsize_sec = ttk.Entry(parent, width=6,
                                            textvariable=self.entryvar_steps_sec)
        self.entry_stepsize_sec.pack(side=tk.LEFT)

    def play(self):
        pass

    def on_button_play(self):
        if self.button_play.instate(("!selected",)):
            self.button_play.state(("selected",))
            self.button_play.config(text="Pause")
        else:
            self.button_play.state(("!selected",))
            self.button_play.config(text="Play")


class AnimationFrame(tk.Frame):
    def __init__(self, parent, controller, figure):
        tk.Frame.__init__(self, parent)

        # f = Figure(figsize=(12, 6), dpi=100)
        # a = f.add_subplot(111)

        canvas = FigureCanvasTkAgg(figure, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()

        scale_frame = ttk.Frame(self)
        scale = ttk.Scale(self, orient=tk.HORIZONTAL)
        scale.pack(fill=tk.BOTH)
        scale_frame.pack()

        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(side=tk.LEFT)

        button_play = ttk.Button(toolbar_frame, text="Play",
                                 command=lambda *args: self.toggle_play_pause(button_play))
        button_play.pack(side=tk.LEFT)
        button_stop = ttk.Button(toolbar_frame, text="Stop")
        button_stop.pack(side=tk.LEFT)
        button_back = ttk.Button(toolbar_frame, text="< Back")
        button_back.pack(side=tk.LEFT)
        button_forward = ttk.Button(toolbar_frame, text="Forward >")
        button_forward.pack(side=tk.LEFT)
        label_step = ttk.Label(toolbar_frame, text='0.000/0.000 sec, 0/0 frames')
        label_step.pack(side=tk.LEFT, padx=5)
        label_stepsize = ttk.Label(toolbar_frame, text="Step size: Frames: ")
        label_stepsize.pack(side=tk.LEFT)
        self.entryvar_steps_frames = tk.StringVar()
        self.entryvar_steps_frames.set("0")
        entry_stepsize_frame = ttk.Entry(toolbar_frame, width=6,
                                         textvariable=self.entryvar_steps_frames)
        entry_stepsize_frame.pack(side=tk.LEFT)
        label_stepsize_sec = ttk.Label(toolbar_frame, text=" Sec: ")
        label_stepsize_sec.pack(side=tk.LEFT)
        self.entryvar_steps_sec = tk.StringVar()
        self.entryvar_steps_sec.set("0.000")
        entry_stepsize_sec = ttk.Entry(toolbar_frame, width=6,
                                       textvariable=self.entryvar_steps_sec)
        entry_stepsize_sec.pack(side=tk.LEFT)

    def toggle_play_pause(self, button):
        if button.instate(("!selected",)):
            button.state(("selected",))
            button.config(text="Pause")
        else:
            button.state(("!selected",))
            button.config(text="Play")


class Player(tk.Tk, FuncAnimation):
    def __init__(self, fig, func, frames=None, init_func=None, fargs=None,
                 save_count=None, mini=0, maxi=100, pos=(0.125, 0.92), *args, **kwargs):
        self.i = 0
        self.min = mini
        self.max = maxi
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.setup(pos)
        tk.Tk.__init__(self, *args, **kwargs)
        FuncAnimation.__init__(self, self.fig, self.func, frames=self.play(),
                               init_func=init_func, fargs=fargs,
                               save_count=save_count, **kwargs)

    def play(self):
        while self.runs:
            self.i = self.i + self.forwards - (not self.forwards)
            if self.min < self.i < self.max:
                yield self.i
            else:
                self.stop()
                yield self.i

    def start(self):
        self.runs = True
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()

    def forward(self, event=None):
        self.forwards = True
        self.start()

    def backward(self, event=None):
        self.forwards = False
        self.start()

    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()

    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.min < self.i < self.max:
            self.i = self.i + self.forwards - (not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i += 1
        elif self.i == self.max and not self.forwards:
            self.i -= 1
        self.func(self.i)
        # self.slider.set_val(self.i)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        # playerax = self.fig.add_axes([pos[0], pos[1], 0.22, 0.04])
        # divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        # bax = divider.append_axes("right", size="80%", pad=0.05)
        # sax = divider.append_axes("right", size="80%", pad=0.05)
        # fax = divider.append_axes("right", size="80%", pad=0.05)
        # ofax = divider.append_axes("right", size="100%", pad=0.05)
        # sliderax = divider.append_axes("right", size="500%", pad=0.07)
        # self.button_oneback = matplotlib.widgets.Button(playerax, label=u'$\u29CF$')
        # self.button_back = matplotlib.widgets.Button(bax, label=u'$\u25C0$')
        # self.button_stop = matplotlib.widgets.Button(sax, label=u'$\u25A0$')
        # self.button_forward = matplotlib.widgets.Button(fax, label=u'$\u25B6$')
        # self.button_oneforward = matplotlib.widgets.Button(ofax, label=u'$\u29D0$')
        # self.button_oneback.on_clicked(self.onebackward)
        # self.button_back.on_clicked(self.backward)
        # self.button_stop.on_clicked(self.stop)
        # self.button_forward.on_clicked(self.forward)
        # self.button_oneforward.on_clicked(self.oneforward)
        # self.slider = matplotlib.widgets.Slider(sliderax, '',
        #                                         self.min, self.max, valinit=self.i)

        frame_anim = tk.Frame(self)
        frame_anim.grid(row=0, column=0, sticky="nsew")

        canvas = FigureCanvasTkAgg(self.fig, frame_anim)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, frame_anim)
        toolbar.update()

        self.scale = ttk.Scale(frame_anim, orient=tk.HORIZONTAL)
        self.scale.pack(fill=tk.BOTH)
        self.button_play = ttk.Button(frame_anim, text="Play", command=self.start)
        self.button_play.pack(side=tk.LEFT)
        self.button_stop = ttk.Button(frame_anim, text="Stop", command=self.stop)
        self.button_stop.pack(side=tk.LEFT)
        self.button_back = ttk.Button(frame_anim, text="< Back", command=self.onebackward)
        self.button_back.pack(side=tk.LEFT)
        self.button_forward = ttk.Button(frame_anim, text="Forward >", command=self.oneforward)
        self.button_forward.pack(side=tk.LEFT)
        self.label_step = ttk.Label(frame_anim, text='0.000/0.000 sec, 0/0 frames')
        self.label_step.pack(side=tk.LEFT, padx=5)
        self.label_stepsize = ttk.Label(frame_anim, text="Step size: Frames: ")
        self.label_stepsize.pack(side=tk.LEFT)
        self.entryvar_steps_frames = tk.StringVar()
        self.entryvar_steps_frames.set("0")
        self.entry_stepsize_frame = ttk.Entry(frame_anim, width=6,
                                              textvariable=self.entryvar_steps_frames)
        self.entry_stepsize_frame.pack(side=tk.LEFT)
        self.label_stepsize_sec = ttk.Label(frame_anim, text=" Sec: ")
        self.label_stepsize_sec.pack(side=tk.LEFT)
        self.entryvar_steps_sec = tk.StringVar()
        self.entryvar_steps_sec.set("0.000")
        self.entry_stepsize_sec = ttk.Entry(frame_anim, width=6,
                                            textvariable=self.entryvar_steps_sec)
        self.entry_stepsize_sec.pack(side=tk.LEFT)

    # def set_pos(self, i):
    #     # self.i = int(self.slider.val)
    #     self.func(self.i)

    # def update(self, i):
    #     # self.slider.set_val(i)
    #     pass


if __name__ == '__main__':
    # test = AnimationTest()
    # app = PlayerApp(test)
    # app.mainloop()

    fig, ax = plt.subplots()
    x = np.linspace(0, 6 * np.pi, num=100)
    y = np.sin(x)
    ax.plot(x, y)
    point, = ax.plot([], [], marker="o", color="crimson", ms=15)

    def update(i):
        point.set_data(x[i], y[i])

    ani = Player(fig, update, maxi=len(y) - 1)
    ani.mainloop()
