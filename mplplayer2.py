import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets

# pause = False
#
#
# def sim_data():
#     t_max = 10.0
#     dt = 0.05
#     x = 0.0
#     t = 0.0
#     while t < t_max:
#         if not pause:
#             x = np.sin(np.pi * t)
#             t = t + dt
#         yield x, t
#
#
# def on_click(event):
#     global pause
#     pause ^= True
#
#
# def sim_points(sim_data):
#     x, t = sim_data[0], sim_data[1]
#     time_text.set_text(time_template % t)
#     line.set_data(t, x)
#     return line, time_text
#
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# line, = ax.plot([], [], 'bo', ms=10)
# ax.set_ylim(-1, 1)
# ax.set_xlim(0, 10)
#
# time_template = 'Time = %.1f s'
# time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
# fig.canvas.mpl_connect('button_press_event', on_click)
# ani = animation.FuncAnimation(fig, sim_points, sim_data, blit=False, interval=10,
#                               repeat=True)
# plt.show()


class Player:
    def __init__(self, d, fargs=None, save_count=None, interval=200,
                 repeat=None, repeat_delay=None, blit=None, button_pos=(0.125, 0.92)):
        self.d = d
        self.pause = True

        self.fargs = fargs
        self.save_count = save_count
        self.interval = interval
        self.repeat_delay = repeat_delay
        self.repeat = repeat
        self.blit = blit

        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'bo', ms=10)
        self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)
        self.anim = None

        play_ax = self.fig.add_axes([button_pos[0], button_pos[1], 0.16, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(play_ax)
        pause_ax = divider.append_axes("right", size="100%", pad=0.05)
        self.button_play = matplotlib.widgets.Button(play_ax, label=u'$\u25B6$')
        self.button_pause = matplotlib.widgets.Button(pause_ax, label=u'$\u258B\u258B$')
        self.button_play.on_clicked(self.on_play)
        self.button_pause.on_clicked(self.on_pause)

        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def func(self, p):
        x, t = p[0], p[1]
        self.time_text.set_text('Time = %.1f s' % t)
        self.line.set_data(t, x)
        return self.line, self.time_text

    def init_func(self):
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, 10)
        return self.line,

    def update(self):
        for i in range(len(self.d)):
            yield self.d[i]

    def run(self):
        self.anim = FuncAnimation(
            self.fig, self.func, frames=self.update, init_func=self.init_func,
            fargs=self.fargs, save_count=self.save_count,
            interval=self.interval, repeat=self.repeat,
            repeat_delay=self.repeat_delay, blit=self.blit)
        plt.show()

    def on_play(self, event):
        self.anim.event_source.start()

    def on_pause(self, event):
        self.anim.event_source.stop()

    def on_key_press(self, event):
        if event.key == 'k':
            print('pressed', event.key)
            if self.pause:
                print('pause', self.pause)
                self.pause ^= True
                self.anim.event_source.stop()
            else:
                print('pause', self.pause)
                self.pause ^= True
                self.anim.event_source.start()




if __name__ == '__main__':
    t_max = 10.0
    dt = 0.05
    x = 0.0
    t = 0.0
    d = []
    while t < t_max:
        x = np.sin(np.pi * t)
        t += dt
        d.append([x, t])

    p = Player(d, blit=True)
    p.run()
