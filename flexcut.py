"""
Purpose for simulation:

Given parameters:
    product size
    product belt speed
    product cut patterns
    allowable cut window area

    motion paths between cut patterns (if they are more than one)
    motion paths between products
    nozzle resting (ready) position
    type of x,y cam profiles, the nozzle-drive uses

Returns:
    The required x,y position/speed/acceleration/jerk curves for the nozzle-drive to cut the given cut patterns

These results scan then be used in motor specificatin software, like the Lenze Drive Solution Designer, to choose
    proper motors (along with the nozzle-drive mechanism inertia and friction).
"""


import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.text as text
import matplotlib.lines as lines
import matplotlib.transforms as transforms
import matplotlib.collections as collections

import camprofiles


def polyline_cut2(speed, runtime, fps, piece, cutpath, nozzlevelmax, doanim=False, iscut=None):
    # cutpath = [[x, ..., x], [y, ..., y]] with respect to piece-lower-left-corner
    nozzle_window_pos = (0, -.5 * piece[1])  # Position of the cut window lower-left corner
    fig_size = (7, 3)  # (12, 5)  # inches, default 6.4, 4.8
    fig_dpi = 200  # default 100
    anim_delay = .1  # ms
    piece_pos_start = -1.1 * piece[0] - nozzle_window_pos[0]  # The piece travels in the x-direction
    time_steps = np.linspace(0, runtime, int(fps * runtime))  # Shape (N, 1)
    piece_pos_steps = speed * time_steps + piece_pos_start  # Shape (N, 1)

    cutpath = np.asarray(cutpath)
    cutpath_pos_steps = piece_pos_steps + cutpath[0, 0]
    cutpath[0] = piece_pos_start + cutpath[0]
    cutpath[1] = -.5 * piece[1] + cutpath[1]

    # plt.title('piece cutting paths')
    # plt.plot(cutpath[0], cutpath[1])
    # plt.show()
    # plt.title('')
    # Nozzle path

    # Devite the nozzle cut x compoent in equal intervals over the window length
    # nozzle_path_x = nozzle_window_pos[0] + np.linspace(0, 1, cutpath[0].size) * window[0]

    nozzle_path_y = cutpath[1]

    # Find the smallest possible x-component-length for each nozzle-cut-path when "nozzlevelmax" is utilized.
    # Each cut-path consists of a straght line between two points.
    nozzle_path_x = nozzle_window_pos[0] * np.ones(cutpath[0].size)
    for i, x1, x2, y1, y2 in zip(range(1, cutpath[0].size), cutpath[0, :-1], cutpath[0, 1:], cutpath[1, :-1],
                                 cutpath[1, 1:]):
        if x1 == x2 and y1 != y2:
            # Side cut
            # Find required x-length at max-nozzle-velocity-y-component.
            nozzle_path_x[i] = speed / nozzlevelmax[1] * np.abs(y2 - y1) + nozzle_path_x[i - 1]

        elif x1 > x2 and y1 != y2:
            # Angular back cut

            # The most prefered case is to move the nozzle backwards as fast as possible
            tn = (x2 - x1) / (-nozzlevelmax[0] - speed)
            nozzle_path_x[i] = -nozzlevelmax[0] * tn + nozzle_path_x[i - 1]

            # vnx = -nozzlevelmax[0]
            # vny = (y2 - y1) / tn
            # print(1, 'vnx', vnx, 'vny', vny)

            if np.abs((y2 - y1) / tn) > nozzlevelmax[1]:
                # Slow down the y-speed-component to max-nozzle-velocity-y-component
                if y2 > y1:
                    # Upwards cut
                    vny = nozzlevelmax[1]
                else:
                    # y2 < y1, downwards cut
                    vny = -nozzlevelmax[1]

                nozzle_path_x[i] = speed / vny * (y2 - y1) + (x2 - x1) + nozzle_path_x[i - 1]

                # tn = ((nozzle_path_x[i] - nozzle_path_x[i - 1]) - (x2 - x1)) / speed
                # vnx = (nozzle_path_x[i] - nozzle_path_x[i - 1]) / tn
                # print(11, 'vnx', vnx, 'vny', vny)

            if nozzle_path_x[i] < nozzle_window_pos[0]:
                # The nozzle path crosses the nozzle window start line
                nozzle_path_x[i] = nozzle_window_pos[0]

                # tn = ((nozzle_path_x[i] - nozzle_path_x[i - 1]) - (x2 - x1)) / speed
                # vnx = (nozzle_path_x[i] - nozzle_path_x[i - 1]) / tn
                # vny = (y2 - y1) / tn
                # print(2, 'vnx', vnx, 'vny', vny)

        elif x1 > x2 and y1 == y2:
            # Longitudinal back cut
            nozzle_run_time = (x1 - x2) / (nozzlevelmax[0] + speed)
            nozzle_path_x[i] = -nozzle_run_time * nozzlevelmax[0] + nozzle_path_x[i - 1]

            if nozzle_path_x[i] < nozzle_window_pos[0]:
                nozzle_path_x[i] = nozzle_window_pos[0]

            # Don't move the nozzle during the cut
            # nozzle_path_x[i] = nozzle_path_x[i-1]

        elif x1 < x2 and y1 != y2:
            # Angular forward cut
            if speed < nozzlevelmax[0]:
                # Set nozzle-x-velocity-component to max-nozzle-x-velocity-component
                nozzle_run_time = (x2 - x1) / (nozzlevelmax[0] - speed)
                nozzle_path_x[i] = nozzle_run_time * nozzlevelmax[0] + nozzle_path_x[i - 1]

                if np.abs(y2 - y1) / nozzle_run_time > nozzlevelmax[1]:
                    # Set nozzle-y-velocity-component to max-nozzle-y-velocity-component
                    nozzle_path_x[i] = speed / nozzlevelmax[1] * np.abs(y2 - y1) + (x2 - x1) + nozzle_path_x[i - 1]
                    nozzle_run_time = np.abs(y2 - y1) / nozzlevelmax[1]

                    if (nozzle_path_x[i] - nozzle_path_x[i - 1]) / nozzle_run_time > nozzlevelmax[0]:
                        print(f'Error in cut path nr. {i}, for angular forward cut,'
                              f'x1, y1 = {x1}, {y1}; x2, y2 = {x2}, {y2}.')
                        print(f'Max x- and y-velocity-components have been exceeted')
                        return
            else:
                print(f'Error in cut path nr. {i}, for angular forward cut, the belt speed {speed} is smaller than'
                      f'max-nozzle-x-velocity-component {nozzlevelmax[0]}')
                return

        elif x1 < x2 and y1 == y2:
            # Longitudinal forward cut
            if speed < nozzlevelmax[0]:
                # Set nozzle-x-velocity-component to max-nozzle-x-velocity-component
                nozzle_path_x[i] = 1 / (1 - speed / nozzlevelmax[0]) * (x2 - x1) + nozzle_path_x[i - 1]
            else:
                print(f'Error in cut path nr. {i}, for longitudinal forward cut,'
                      f' x1, y1 = {x1}, {y1}; x2, y2 = {x2}, {y2}.')
                print(f'Nozzle-max-x-velocity {nozzlevelmax[0]}, is smaller than the belt-speed {speed}.')
                return
        else:
            print(f'Error in cut path. x1, y1 = {x1}, {y1}; x2, y2 = {x2}, {y2}.')
            return

    nozzle_window_length = np.max(nozzle_path_x) - nozzle_window_pos[0]

    # plt.plot(cutpath[0], cutpath[1])
    # plt.plot(nozzle_path_x, nozzle_path_y)
    # plt.show()
    # return

    # Nozzle steps array, reprecenting a dot during animation
    # Initialize the nozzle_steps array. All time-steps are in the idle position at nozzle_path[0, 0].
    nozzle_steps = np.array([nozzle_path_x[0], nozzle_path_y[0]]) * np.ones((time_steps.size, 2))  # Shape (N, 2)

    # Time points between cut lines
    ind = np.zeros(nozzle_path_x.size, dtype=np.int)
    time_points = np.zeros(nozzle_path_x.size)
    cutpath_x_diff = np.block([0, np.diff(cutpath[0])])
    cutpath_x_steps = cutpath_pos_steps
    for i in range(ind.size):
        cutpath_x_steps += cutpath_x_diff[i]
        ind[i] = np.where(cutpath_x_steps > nozzle_path_x[i])[0][0]
        time_points[i] = time_steps[ind[i]]

    # i0 = np.where(cutpath_pos_steps >= nozzle_path_x[0])[0][0]
    # i1 = np.where(cutpath_pos_steps + cutpath[0, 1] - cutpath[0, 0] >= nozzle_path_x[1])[0][0]
    # i2 = np.where(cutpath_pos_steps + cutpath[0, 2] - cutpath[0, 1] >= nozzle_path_x[2])[0][0]
    # tmp = np.array([time_steps[i0], time_steps[i1], time_steps[i2]])
    # print('ind', ind)

    # Time interval for each line
    nozzle_run_time = np.diff(time_points)

    for i in range(nozzle_run_time.size):
        if nozzle_run_time[i] <= 0:
            print(f'Error in cut path nr. {i}, the run-time is negative or zero: {nozzle_run_time[i]}.')
            return

    # x and y cut velocity components for each line, shape (2, N)
    nozzle_vel = np.array([np.diff(nozzle_path_x) / nozzle_run_time, np.diff(cutpath[1]) / nozzle_run_time])
    # print('runtime', nozzle_run_time)
    # print('vel', nozzle_vel)
    # plt.plot(nozzle_vel[0])
    # plt.show()
    # print('nozzle_path_x', nozzle_path_x)
    # print('nozzle_path_y', nozzle_path_y)
    # return

    for i in range(nozzle_run_time.size):
        # nozzle steps in x and y directions
        # nozzle_cut_steps = [vx * dt * [0, 1, 2, ...], vy * dt * [0, 1, 2, ...]].T, shape (2, N)
        #                  = [[vx * dt * 0, vy * dt * 0], [vx * dt * 1, vy * dt * 1], ...], shape (N, 2)
        #                  = [[x,y], [x,y], ... ], shape (N, 2)
        # nozzle_cut_steps = (np.array([nozzle_vel[:, i]]).T * time_steps[1] * np.arange(ind[i + 1] - ind[i])).T

        num_points = ind[i + 1] - ind[i]

        # print('\ni', i)
        # print('num_points = indi+1 - indi', ind[i + 1] - ind[i])
        # print('dist = nozzle_path_x[i + 1] - nozzle_path_x[i]', nozzle_path_x[i + 1] - nozzle_path_x[i])
        # print('time = time_steps[1] * num_points', time_steps[1] * num_points)

        dir_x = 1 if nozzle_path_x[i + 1] - nozzle_path_x[i] >= 0 else -1
        dir_y = 1 if nozzle_path_y[i + 1] - nozzle_path_y[i] >= 0 else -1

        quad_x = camprofiles.Quadratic(dist=nozzle_path_x[i + 1] - nozzle_path_x[i], time=nozzle_run_time[i],
                                       vmax=dir_x * nozzlevelmax[0] * 2, num=num_points, name=f'quad x{i}')
        quad_y = camprofiles.Quadratic(dist=nozzle_path_y[i + 1] - nozzle_path_y[i], time=nozzle_run_time[i],
                                       vmax=dir_y * nozzlevelmax[0] * 2, num=num_points, name=f'quad y{i}')
        print(quad_x)
        print(quad_y)
        # camprofiles.plot_cam((quad_x, quad_y))
        # plt.show()
        # return

        # nozzle_cut_steps_x = quad_x.pos + nozzle_path_x[i]
        # nozzle_cut_steps_y = quad_y.pos + nozzle_path_y[i]
        # print('quad_x.pos', quad_x.pos[-5:])
        # print('nozzle_cut_steps[:,0]', nozzle_cut_steps[-5:, 0])
        nozzle_cut_steps = (np.array([quad_x.pos, quad_y.pos])).T  # Shape (N, 2)
        # print(nozzle_cut_steps.shape)
        # print(nozzle_cut_steps_tmp.shape)

        # Translate the nozzle to the beginning of the cut window and the initial vertical position of the cut path
        nozzle_cut_steps[:, 0] += nozzle_path_x[i]
        nozzle_cut_steps[:, 1] += nozzle_path_y[i]

        # Add the "nozzle cut steps array" into the initial "nozzle steps" array at the nozzle-run-time-interval no. i
        nozzle_steps[ind[i]:ind[i + 1]] = nozzle_cut_steps

    # Stop the nozzle at a x,y point, at the end of the last nozzle-path-segment
    nozzle_steps[ind[-1]:] = nozzle_steps[ind[-1] - 1] * np.ones((time_steps.size - ind[-1], 2))

    nozzle_pos_steps = nozzle_steps.T  # Transform to create a (2, N) shape array, for plotting
    nozzle_vel_steps = np.gradient(nozzle_pos_steps, time_steps[1], axis=1)  # Shape (2, N)

    # return

    # Graphics

    # fig, ax0 = plt.subplots(figsize=fig_size, dpi=fig_dpi, constrained_layout=True)
    fig = plt.figure(figsize=fig_size, dpi=fig_dpi, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[.5, .5], height_ratios=[.6, .4])
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    tick_kw = dict(labelsize=7)
    ax0.tick_params(**tick_kw)
    ax1.tick_params(**tick_kw)
    ax2.tick_params(**tick_kw)

    ax0.set_title(f'Belt speed: {speed} mm/s, window length: {nozzle_window_length:.1f} mm', fontsize=7)
    ax0.set_xlim(1.1 * piece_pos_start, 1.1 * piece_pos_steps[-1] + piece[0])
    ax0.set_ylim(-1.3 * piece[1] / 2, 1.5 * piece[1] / 2)
    ax0.set_aspect(1)
    xlim, ylim = ax0.get_xlim(), ax0.get_ylim()

    label_text = text.Text(xlim[0] + 10, ylim[1] - 30, f'Time: {time_steps[0]:.2f}s')
    label_text.set_fontsize(7)
    ax0.add_artist(label_text)

    # The cut window
    # win_rect = patches.Rectangle(nozzle_window_pos, window[0], window[1],
    #                              ec='grey', fill=False, ls=':')
    # ax0.add_patch(win_rect)
    win_start = lines.Line2D([nozzle_window_pos[0], nozzle_window_pos[0]], ylim,
                             linestyle='dotted', linewidth=.5)
    ax0.add_line(win_start)
    win_end = lines.Line2D([nozzle_window_length, nozzle_window_length], ylim,
                           linestyle='dotted', linewidth=.5)
    ax0.add_line(win_end)

    # The piece
    piece_rect = patches.Rectangle((piece_pos_start, -.5 * piece[1]), piece[0], piece[1],
                                   fill=True, ec=None, fc='lightgrey', alpha=.4)
    ax0.add_patch(piece_rect)

    # The intented cut path on the piece during animation
    if iscut is not None:
        cut_lines = []
        linestyles = []
        linewidths = []
        for c, x1, x2, y1, y2 in zip(iscut, cutpath[0, :-1], cutpath[0, 1:], cutpath[1, :-1], cutpath[1, 1:]):
            if c:
                linestyles.append('solid')
                linewidths.append(1.)
            else:
                linestyles.append('dotted')
                linewidths.append(.5)
            cut_lines.append([(x1, y1), (x2, y2)])
        print('collection')
        piece_cut = collections.LineCollection(cut_lines, color='lightgrey', linewidths=linewidths,
                                               linestyles=linestyles)
    else:
        piece_cut = lines.Line2D(cutpath[0], cutpath[1], color='lightgrey', linewidth=1.)
    ax0.add_artist(piece_cut)

    # The nozzle

    # Nozzle movement path, within the cut window
    if iscut is not None:
        nozzle_path_lines = []
        linestyles = []
        linewidths = []
        for c, x1, x2, y1, y2 in zip(iscut, nozzle_path_x[:-1], nozzle_path_x[1:], nozzle_path_y[:-1],
                                     nozzle_path_y[1:]):
            if c:
                linestyles.append('solid')
                linewidths.append(1.)
            else:
                linestyles.append('dotted')
                linewidths.append(.5)
            nozzle_path_lines.append([(x1, y1), (x2, y2)])
        nozzle_path = collections.LineCollection(nozzle_path_lines, color='lightblue', linestyles=linestyles,
                                                 linewidths=linewidths, alpha=.5)
    else:
        nozzle_path = lines.Line2D(nozzle_path_x, nozzle_path_y, color='lightblue', linewidth=1, alpha=.5)
    ax0.add_artist(nozzle_path)

    # Nozzle dot, during animation
    nozzle_dot = lines.Line2D([nozzle_path_x[0]], [nozzle_path_y[0]], marker='x', color='lightblue')
    ax0.add_line(nozzle_dot)

    # A path showing the part of the cut path, which has been cut during each animation step
    nozzle_cut = lines.Line2D([], [], color='black', zorder=10)
    ax0.add_line(nozzle_cut)

    def cam_curve(dim, ax, pos, vel):
        ax.annotate(dim, xy=(.01, .975), xycoords='axes  fraction', horizontalalignment='left',
                    verticalalignment='top', color='black', fontsize=10)
        ax.grid(True)
        ax_1 = ax.twinx()
        ax_1.spines['left'].set_position(('outward', 30))

        ax_1.set_frame_on(True)
        ax_1.patch.set_visible(False)
        for spine in ax_1.spines.values():
            spine.set_visible(False)
        ax_1.yaxis.set_ticks_position('left')
        ax_1.yaxis.set_label_position('left')
        ax_1.spines['left'].set_visible(True)

        p2, = ax_1.plot(time_steps, vel, color='red', label='v', alpha=0.5, linewidth=1.)
        p1, = ax.plot(time_steps, pos, color='blue', label='x', linewidth=1.)

        ax.yaxis.label.set_color(p1.get_color())
        ax_1.yaxis.label.set_color(p2.get_color())
        ax.tick_params(axis='y', colors=p1.get_color())
        ax_1.tick_params(axis='y', colors=p2.get_color(), **tick_kw)

        drag_line = lines.Line2D([time_steps[0], time_steps[0]], ax.get_ylim(), color='black')
        ax.add_line(drag_line)

        return drag_line

    drag_line_x = cam_curve('x', ax1, nozzle_pos_steps[0], nozzle_vel_steps[0])
    drag_line_y = cam_curve('y', ax2, nozzle_pos_steps[1], nozzle_vel_steps[1])

    # plt.show()
    # return

    class Draggable:
        def __init__(self):
            self.picked = False
            self.ax = None
            fig.canvas.mpl_connect('button_press_event', self.on_button_press)
            fig.canvas.mpl_connect('button_release_event', self.on_button_release)
            fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        def on_button_press(self, event):
            self.picked, attrd = drag_line_x.contains(event)
            self.ax = event.inaxes
            if not self.picked:
                self.picked, attrd = drag_line_y.contains(event)

        def on_button_release(self, event):
            self.picked = False
            self.ax = None

        def on_mouse_move(self, event):
            if self.picked and event.inaxes is self.ax:
                # print('move', self.picked, event)
                try:
                    i = np.where(time_steps >= event.xdata)[0][0]
                except IndexError:
                    return
                # print(i)

                drag_line_x.set_xdata([time_steps[i], time_steps[i]])
                drag_line_y.set_xdata([time_steps[i], time_steps[i]])

                label_text.set_text(f'Time: {time_steps[i]:.2f}s')
                piece_rect.set_x(piece_pos_steps[i])
                piece_cut.set_transform(transforms.Affine2D().translate(
                    cutpath_pos_steps[i] - cutpath_pos_steps[0], 0) + ax0.transData)
                nozzle_dot.set_xdata(nozzle_steps[i, 0])
                nozzle_dot.set_ydata(nozzle_steps[i, 1])

                fig.canvas.blit(ax0.bbox)
                fig.canvas.blit(ax1.bbox)
                fig.canvas.blit(ax2.bbox)
                fig.canvas.draw_idle()

    # For animation

    def anim_data_gen():
        for t, p, c, n in zip(time_steps, piece_pos_steps, cutpath_pos_steps - cutpath_pos_steps[0], nozzle_steps):
            yield t, p, c, n[0], n[1]

    def anim_update(data):
        t, p, c, nx, ny = data
        label_text.set_text(f'Time: {t:.2f}s')

        # The piece
        piece_rect.set_x(p)
        piece_cut.set_transform(transforms.Affine2D().translate(c, 0) + ax0.transData)

        # The position of the nozzle-dot
        nozzle_dot.set_xdata(nx)
        nozzle_dot.set_ydata(ny)

        drag_line_x.set_xdata([t, t])
        drag_line_y.set_xdata([t, t])

        # The cut
        # if t1 <= t <= t2:
        #     nozzle_cut.set_data([[piece_cut.get_xdata()[0], nx], [piece_cut.get_ydata()[0], ny]])
        #
        # if t >= t2:
        #     nozzle_cut.set_xdata([p + cutstart[0], p + cutstart[0] + cut[0]])

        return label_text, piece_rect, piece_cut, nozzle_dot, nozzle_cut, drag_line_x, drag_line_y

    if doanim:
        piece_rect.set_animated(doanim)
        piece_cut.set_animated(doanim)
        nozzle_dot.set_animated(doanim)
        nozzle_cut.set_animated(doanim)
        drag_line_x.set_animated(doanim)
        drag_line_y.set_animated(doanim)

        anim = animation.FuncAnimation(fig=fig, func=anim_update, frames=anim_data_gen, interval=anim_delay, blit=True,
                                       repeat=False, save_count=int(runtime * fps))
        # anim.save('angular_cut_1.mp4', fps=200)
        plt.show()

    else:
        draggable = Draggable()
        plt.show()


def polyline_cut(speed, runtime, fps, piece, cutpath, nozzlevelmax, doanim=False, iscut=None):
    # cutpath = [[x, ..., x], [y, ..., y]] with respect to piece-lower-left-corner
    nozzle_window_pos = (0, -.5 * piece[1])  # Position of the cut window lower-left corner
    fig_size = (7, 3)  # (12, 5)  # inches, default 6.4, 4.8
    fig_dpi = 200  # default 100
    anim_delay = .1  # ms
    piece_pos_start = -1.1 * piece[0] - nozzle_window_pos[0]  # The piece travels in the x-direction
    time_steps = np.linspace(0, runtime, int(fps * runtime))  # Shape (N, 1)
    piece_pos_steps = speed * time_steps + piece_pos_start  # Shape (N, 1)

    cutpath = np.asarray(cutpath)
    cutpath_pos_steps = piece_pos_steps + cutpath[0, 0]
    cutpath[0] = piece_pos_start + cutpath[0]
    cutpath[1] = -.5 * piece[1] + cutpath[1]

    # plt.title('piece cutting paths')
    # plt.plot(cutpath[0], cutpath[1])
    # plt.show()
    # plt.title('')
    # Nozzle path

    # Devite the nozzle cut x compoent in equal intervals over the window length
    # nozzle_path_x = nozzle_window_pos[0] + np.linspace(0, 1, cutpath[0].size) * window[0]

    nozzle_path_y = cutpath[1]

    # Find the smallest possible x-component-length for each nozzle-cut-path when "nozzlevelmax" is utilized.
    # Each cut-path consists of a straght line between two points.
    nozzle_path_x = nozzle_window_pos[0] * np.ones(cutpath[0].size)
    for i, x1, x2, y1, y2 in zip(range(1, cutpath[0].size), cutpath[0, :-1], cutpath[0, 1:], cutpath[1, :-1], cutpath[1, 1:]):
        if x1 == x2 and y1 != y2:
            # Side cut
            # Find required x-length at max-nozzle-velocity-y-component.
            nozzle_path_x[i] = speed / nozzlevelmax[1] * np.abs(y2 - y1) + nozzle_path_x[i-1]

        elif x1 > x2 and y1 != y2:
            # Angular back cut

            # The most prefered case is to move the nozzle backwards as fast as possible
            tn = (x2 - x1) / (-nozzlevelmax[0] - speed)
            nozzle_path_x[i] = -nozzlevelmax[0] * tn + nozzle_path_x[i - 1]

            # vnx = -nozzlevelmax[0]
            # vny = (y2 - y1) / tn
            # print(1, 'vnx', vnx, 'vny', vny)

            if np.abs((y2 - y1) / tn) > nozzlevelmax[1]:
                # Slow down the y-speed-component to max-nozzle-velocity-y-component
                if y2 > y1:
                    # Upwards cut
                    vny = nozzlevelmax[1]
                else:
                    # y2 < y1, downwards cut
                    vny = -nozzlevelmax[1]

                nozzle_path_x[i] = speed / vny * (y2 - y1) + (x2 - x1) + nozzle_path_x[i - 1]

                # tn = ((nozzle_path_x[i] - nozzle_path_x[i - 1]) - (x2 - x1)) / speed
                # vnx = (nozzle_path_x[i] - nozzle_path_x[i - 1]) / tn
                # print(11, 'vnx', vnx, 'vny', vny)

            if nozzle_path_x[i] < nozzle_window_pos[0]:
                # The nozzle path crosses the nozzle window start line
                nozzle_path_x[i] = nozzle_window_pos[0]

                # tn = ((nozzle_path_x[i] - nozzle_path_x[i - 1]) - (x2 - x1)) / speed
                # vnx = (nozzle_path_x[i] - nozzle_path_x[i - 1]) / tn
                # vny = (y2 - y1) / tn
                # print(2, 'vnx', vnx, 'vny', vny)

        elif x1 > x2 and y1 == y2:
            # Longitudinal back cut
            nozzle_run_time = (x1 - x2) / (nozzlevelmax[0] + speed)
            nozzle_path_x[i] = -nozzle_run_time * nozzlevelmax[0] + nozzle_path_x[i - 1]

            if nozzle_path_x[i] < nozzle_window_pos[0]:
                nozzle_path_x[i] = nozzle_window_pos[0]

            # Don't move the nozzle during the cut
            # nozzle_path_x[i] = nozzle_path_x[i-1]

        elif x1 < x2 and y1 != y2:
            # Angular forward cut
            if speed < nozzlevelmax[0]:
                # Set nozzle-x-velocity-component to max-nozzle-x-velocity-component
                nozzle_run_time = (x2 - x1) / (nozzlevelmax[0] - speed)
                nozzle_path_x[i] = nozzle_run_time * nozzlevelmax[0] + nozzle_path_x[i-1]

                if np.abs(y2 - y1) / nozzle_run_time > nozzlevelmax[1]:
                    # Set nozzle-y-velocity-component to max-nozzle-y-velocity-component
                    nozzle_path_x[i] = speed / nozzlevelmax[1] * np.abs(y2 - y1) + (x2 - x1) + nozzle_path_x[i-1]
                    nozzle_run_time = np.abs(y2 - y1) / nozzlevelmax[1]

                    if (nozzle_path_x[i] - nozzle_path_x[i-1]) / nozzle_run_time > nozzlevelmax[0]:
                        print(f'Error in cut path nr. {i}, for angular forward cut,'
                              f'x1, y1 = {x1}, {y1}; x2, y2 = {x2}, {y2}.')
                        print(f'Max x- and y-velocity-components have been exceeted')
                        return
            else:
                print(f'Error in cut path nr. {i}, for angular forward cut, the belt speed {speed} is smaller than'
                      f'max-nozzle-x-velocity-component {nozzlevelmax[0]}')
                return

        elif x1 < x2 and y1 == y2:
            # Longitudinal forward cut
            if speed < nozzlevelmax[0]:
                # Set nozzle-x-velocity-component to max-nozzle-x-velocity-component
                nozzle_path_x[i] = 1 / (1 - speed / nozzlevelmax[0]) * (x2 - x1) + nozzle_path_x[i-1]
            else:
                print(f'Error in cut path nr. {i}, for longitudinal forward cut,'
                      f' x1, y1 = {x1}, {y1}; x2, y2 = {x2}, {y2}.')
                print(f'Nozzle-max-x-velocity {nozzlevelmax[0]}, is smaller than the belt-speed {speed}.')
                return
        else:
            print(f'Error in cut path. x1, y1 = {x1}, {y1}; x2, y2 = {x2}, {y2}.')
            return

    nozzle_window_length = np.max(nozzle_path_x) - nozzle_window_pos[0]

    # plt.plot(cutpath[0], cutpath[1])
    # plt.plot(nozzle_path_x, nozzle_path_y)
    # plt.show()
    # return

    # Nozzle steps array, reprecenting a dot during animation
    # Initialize the nozzle_steps array. All time-steps are in the idle position at nozzle_path[0, 0].
    nozzle_steps = np.array([nozzle_path_x[0], nozzle_path_y[0]]) * np.ones((time_steps.size, 2))  # Shape (N, 2)

    # Time points between cut lines
    ind = np.zeros(nozzle_path_x.size, dtype=np.int)
    time_points = np.zeros(nozzle_path_x.size)
    cutpath_x_diff = np.block([0, np.diff(cutpath[0])])
    cutpath_x_steps = cutpath_pos_steps
    for i in range(ind.size):
        cutpath_x_steps += cutpath_x_diff[i]
        ind[i] = np.where(cutpath_x_steps > nozzle_path_x[i])[0][0]
        time_points[i] = time_steps[ind[i]]

    # i0 = np.where(cutpath_pos_steps >= nozzle_path_x[0])[0][0]
    # i1 = np.where(cutpath_pos_steps + cutpath[0, 1] - cutpath[0, 0] >= nozzle_path_x[1])[0][0]
    # i2 = np.where(cutpath_pos_steps + cutpath[0, 2] - cutpath[0, 1] >= nozzle_path_x[2])[0][0]
    # tmp = np.array([time_steps[i0], time_steps[i1], time_steps[i2]])

    # Time interval for each line
    nozzle_run_time = np.diff(time_points)

    for i in range(nozzle_run_time.size):
        if nozzle_run_time[i] <= 0:
            print(f'Error in cut path nr. {i}, the run-time is negative or zero: {nozzle_run_time[i]}.')
            #       f'The allowable nozzle cut-length in the x-direction ({np.diff(nozzle_path_x)[i]:.2f} mm) is')
            # print(f'shorter than or equal to the x-component-length to cut on the piece: {cutpath_x_diff[i+1]:.2f} mm')
            return

    # x and y cut velocity components for each line
    nozzle_vel = np.array([np.diff(nozzle_path_x) / nozzle_run_time, np.diff(cutpath[1]) / nozzle_run_time])
    # print('runtime', nozzle_run_time)
    # print('vel', nozzle_vel)

    for i in range(nozzle_run_time.size):
        # nozzle steps in x and y directions
        # nozzle_cut_steps = [[x,y], [x,y], ... ]
        nozzle_cut_steps = (np.array([nozzle_vel[:, i]]).T * time_steps[1] * np.arange(ind[i+1] - ind[i])).T

        # Translate the nozzle to the beginning of the cut window and the initial vertical position of the cut path
        nozzle_cut_steps[:, 0] += nozzle_path_x[i]
        nozzle_cut_steps[:, 1] += nozzle_path_y[i]

        # Add the "nozzle cut steps array" into the initial "nozzle steps" array at the "nozzle run time-period"
        nozzle_steps[ind[i]:ind[i+1]] = nozzle_cut_steps

        # nozzle_steps[ind[i+1]:] = nozzle_steps[ind[i+1]] * np.ones((time_steps.size - ind[i+1], 2))

    # Stop the nozzle at a x,y point, at the end of the last nozzle-path-segment
    nozzle_steps[ind[-1]:] = nozzle_steps[ind[-1]-1] * np.ones((time_steps.size - ind[-1], 2))

    nozzle_pos_steps = nozzle_steps.T  # Transform to create a (2, N) shape array, for plotting
    nozzle_vel_steps = np.gradient(nozzle_pos_steps, time_steps[1], axis=1)  # Shape (2, N)

    # Graphics

    # fig, ax0 = plt.subplots(figsize=fig_size, dpi=fig_dpi, constrained_layout=True)
    fig = plt.figure(figsize=fig_size, dpi=fig_dpi, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[.5, .5], height_ratios=[.6, .4])
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    tick_kw = dict(labelsize=7)
    ax0.tick_params(**tick_kw)
    ax1.tick_params(**tick_kw)
    ax2.tick_params(**tick_kw)

    ax0.set_title(f'Belt speed: {speed} mm/s, window length: {nozzle_window_length:.1f} mm', fontsize=7)
    ax0.set_xlim(1.1 * piece_pos_start, 1.1 * piece_pos_steps[-1] + piece[0])
    ax0.set_ylim(-1.3 * piece[1] / 2, 1.5 * piece[1] / 2)
    ax0.set_aspect(1)
    xlim, ylim = ax0.get_xlim(), ax0.get_ylim()

    label_text = text.Text(xlim[0] + 10, ylim[1] - 30, f'Time: {time_steps[0]:.2f}s')
    label_text.set_fontsize(7)
    ax0.add_artist(label_text)

    # The cut window
    # win_rect = patches.Rectangle(nozzle_window_pos, window[0], window[1],
    #                              ec='grey', fill=False, ls=':')
    # ax0.add_patch(win_rect)
    win_start = lines.Line2D([nozzle_window_pos[0], nozzle_window_pos[0]], ylim,
                             linestyle='dotted', linewidth=.5)
    ax0.add_line(win_start)
    win_end = lines.Line2D([nozzle_window_length, nozzle_window_length], ylim,
                           linestyle='dotted', linewidth=.5)
    ax0.add_line(win_end)

    # The piece
    piece_rect = patches.Rectangle((piece_pos_start, -.5 * piece[1]), piece[0], piece[1],
                                   fill=True, ec=None, fc='lightgrey', alpha=.4)
    ax0.add_patch(piece_rect)

    # The intented cut path on the piece during animation
    if iscut is not None:
        cut_lines = []
        linestyles = []
        linewidths = []
        for c, x1, x2, y1, y2 in zip(iscut, cutpath[0, :-1], cutpath[0, 1:], cutpath[1, :-1], cutpath[1, 1:]):
            if c:
                linestyles.append('solid')
                linewidths.append(1.)
            else:
                linestyles.append('dotted')
                linewidths.append(.5)
            cut_lines.append([(x1, y1), (x2, y2)])
        print('collection')
        piece_cut = collections.LineCollection(cut_lines, color='lightgrey', linewidths=linewidths,
                                               linestyles=linestyles)
    else:
        piece_cut = lines.Line2D(cutpath[0], cutpath[1], color='lightgrey', linewidth=1.)
    ax0.add_artist(piece_cut)

    # The nozzle

    # Nozzle movement path, within the cut window
    if iscut is not None:
        nozzle_path_lines = []
        linestyles = []
        linewidths = []
        for c, x1, x2, y1, y2 in zip(iscut, nozzle_path_x[:-1], nozzle_path_x[1:], nozzle_path_y[:-1], nozzle_path_y[1:]):
            if c:
                linestyles.append('solid')
                linewidths.append(1.)
            else:
                linestyles.append('dotted')
                linewidths.append(.5)
            nozzle_path_lines.append([(x1, y1), (x2, y2)])
        nozzle_path = collections.LineCollection(nozzle_path_lines, color='lightblue', linestyles=linestyles,
                                                 linewidths=linewidths, alpha=.5)
    else:
        nozzle_path = lines.Line2D(nozzle_path_x, nozzle_path_y, color='lightblue', linewidth=1, alpha=.5)
    ax0.add_artist(nozzle_path)

    # Nozzle dot, during animation
    nozzle_dot = lines.Line2D([nozzle_path_x[0]], [nozzle_path_y[0]], marker='x', color='lightblue')
    ax0.add_line(nozzle_dot)

    # A path showing the part of the cut path, which has been cut during each animation step
    nozzle_cut = lines.Line2D([], [], color='black', zorder=10)
    ax0.add_line(nozzle_cut)

    def cam_curve(dim, ax, pos, vel):
        ax.annotate(dim, xy=(.01, .975), xycoords='axes  fraction', horizontalalignment='left',
                    verticalalignment='top', color='black', fontsize=10)
        ax.grid(True)
        ax_1 = ax.twinx()
        ax_1.spines['left'].set_position(('outward', 30))

        ax_1.set_frame_on(True)
        ax_1.patch.set_visible(False)
        for spine in ax_1.spines.values():
            spine.set_visible(False)
        ax_1.yaxis.set_ticks_position('left')
        ax_1.yaxis.set_label_position('left')
        ax_1.spines['left'].set_visible(True)

        p2, = ax_1.plot(time_steps, vel, color='red', label='v', alpha=0.5, linewidth=1.)
        p1, = ax.plot(time_steps, pos, color='blue', label='x', linewidth=1.)

        ax.yaxis.label.set_color(p1.get_color())
        ax_1.yaxis.label.set_color(p2.get_color())
        ax.tick_params(axis='y', colors=p1.get_color())
        ax_1.tick_params(axis='y', colors=p2.get_color(), **tick_kw)

        drag_line = lines.Line2D([time_steps[0], time_steps[0]], ax.get_ylim(), color='black')
        ax.add_line(drag_line)

        return drag_line

    drag_line_x = cam_curve('x', ax1, nozzle_pos_steps[0], nozzle_vel_steps[0])
    drag_line_y = cam_curve('y', ax2, nozzle_pos_steps[1], nozzle_vel_steps[1])

    # plt.show()
    # return

    class Draggable:
        def __init__(self):
            self.picked = False
            self.ax = None
            fig.canvas.mpl_connect('button_press_event', self.on_button_press)
            fig.canvas.mpl_connect('button_release_event', self.on_button_release)
            fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        def on_button_press(self, event):
            self.picked, attrd = drag_line_x.contains(event)
            self.ax = event.inaxes
            if not self.picked:
                self.picked, attrd = drag_line_y.contains(event)

        def on_button_release(self, event):
            self.picked = False
            self.ax = None

        def on_mouse_move(self, event):
            if self.picked and event.inaxes is self.ax:
                # print('move', self.picked, event)
                try:
                    i = np.where(time_steps >= event.xdata)[0][0]
                except IndexError:
                    return
                # print(i)

                drag_line_x.set_xdata([time_steps[i], time_steps[i]])
                drag_line_y.set_xdata([time_steps[i], time_steps[i]])

                label_text.set_text(f'Time: {time_steps[i]:.2f}s')
                piece_rect.set_x(piece_pos_steps[i])
                piece_cut.set_transform(transforms.Affine2D().translate(
                    cutpath_pos_steps[i] - cutpath_pos_steps[0], 0) + ax0.transData)
                nozzle_dot.set_xdata(nozzle_steps[i, 0])
                nozzle_dot.set_ydata(nozzle_steps[i, 1])

                fig.canvas.blit(ax0.bbox)
                fig.canvas.blit(ax1.bbox)
                fig.canvas.blit(ax2.bbox)
                fig.canvas.draw_idle()

    # For animation

    def anim_data_gen():
        for t, p, c, n in zip(time_steps, piece_pos_steps, cutpath_pos_steps - cutpath_pos_steps[0], nozzle_steps):
            yield t, p, c, n[0], n[1]

    def anim_update(data):
        t, p, c, nx, ny = data
        label_text.set_text(f'Time: {t:.2f}s')

        # The piece
        piece_rect.set_x(p)
        piece_cut.set_transform(transforms.Affine2D().translate(c, 0) + ax0.transData)

        # The position of the nozzle-dot
        nozzle_dot.set_xdata(nx)
        nozzle_dot.set_ydata(ny)

        drag_line_x.set_xdata([t, t])
        drag_line_y.set_xdata([t, t])

        # The cut
        # if t1 <= t <= t2:
        #     nozzle_cut.set_data([[piece_cut.get_xdata()[0], nx], [piece_cut.get_ydata()[0], ny]])
        #
        # if t >= t2:
        #     nozzle_cut.set_xdata([p + cutstart[0], p + cutstart[0] + cut[0]])

        return label_text, piece_rect, piece_cut, nozzle_dot, nozzle_cut, drag_line_x, drag_line_y

    if doanim:
        piece_rect.set_animated(doanim)
        piece_cut.set_animated(doanim)
        nozzle_dot.set_animated(doanim)
        nozzle_cut.set_animated(doanim)
        drag_line_x.set_animated(doanim)
        drag_line_y.set_animated(doanim)

        anim = animation.FuncAnimation(fig=fig, func=anim_update, frames=anim_data_gen, interval=anim_delay, blit=True,
                                       repeat=False, save_count=int(runtime * fps))
        # anim.save('angular_cut_1.mp4', fps=200)
        plt.show()

    else:
        draggable = Draggable()
        plt.show()


def line_cut_nozzle_vel(speed, runtime, fps, piece, window, cutlen, cutangle, cutstart, nozzlecutx):
    window_pos = (0, -.5 * window[1])  # Position of the cut window lower-left corner
    piece_pos_start = -1.6 * piece[0] - window_pos[0]  # The piece travels in the x-direction

    # Rotate the cut line
    cut_angle_rad = np.deg2rad(cutangle)
    cut_angle_cos = np.cos(cut_angle_rad)
    cut_angle_sin = np.sin(cut_angle_rad)
    cut = np.array([[cut_angle_cos, -cut_angle_sin],
                    [cut_angle_sin, cut_angle_cos]]) @ np.array([cutlen, 0])

    time_steps = np.linspace(0, runtime, int(fps * runtime))
    piece_pos_steps = speed * time_steps + piece_pos_start

    nozzle_path_x = window_pos[0] + np.array([0, nozzlecutx])

    # Time points when the piece-cut-path-x-component enters and leaves the cut window.
    # Thats when the nozzle is moving and cutting
    i1 = np.where(piece_pos_steps + cutstart[0] >= nozzle_path_x[0])[0][0]
    i2 = np.where(piece_pos_steps + cutstart[0] + cut[0] >= nozzle_path_x[-1])[0][0]
    t1 = time_steps[i1]
    t2 = time_steps[i2]
    nozzle_run_time = t2 - t1

    nozzle_vel = np.array([nozzlecutx / nozzle_run_time, cut[1] / nozzle_run_time])

    return nozzle_vel


def line_cut_nozzle_vel2(speed, runtime, fps, piece, window, cutlen, cutangle, cutstart, nozzle_vel_max=(1000, 1000)):
    window_pos = (0, -.5 * window[1])  # Position of the cut window lower-left corner
    piece_pos_start = -1.6 * piece[0] - window_pos[0]  # The piece travels in the x-direction

    # Rotate the cut line
    cut_angle_rad = np.deg2rad(cutangle)
    cut_angle_cos = np.cos(cut_angle_rad)
    cut_angle_sin = np.sin(cut_angle_rad)
    cut = np.array([[cut_angle_cos, -cut_angle_sin],
                    [cut_angle_sin, cut_angle_cos]]) @ np.array([cutlen, 0])

    time_steps = np.linspace(0, runtime, int(fps * runtime))
    piece_pos_steps = speed * time_steps + piece_pos_start

    if cutangle == 180:
        # A back longitudinal cut
        nozzle_vel_x = 0
        nozzle_vel_y = 0

    elif 90 < cutangle < 180 or 180 < cutangle < 270:
        # A backward angular cut
        # Using only the y-axis if possible.
        nozzle_path_x = window_pos[0] + np.array([0, 0])

        # Time points when the piece-cut-path-x-component enters and leaves the cut window.
        # Thats when the nozzle is moving and cutting
        i1 = np.where(piece_pos_steps + cutstart[0] >= nozzle_path_x[0])[0][0]
        i2 = np.where(piece_pos_steps + cutstart[0] + cut[0] >= nozzle_path_x[-1])[0][0]
        t1 = time_steps[i1]
        t2 = time_steps[i2]
        nozzle_run_time = t2 - t1

        nozzle_vel_x = window[0] / nozzle_run_time
        nozzle_vel_y = cut[1] / nozzle_run_time

        if nozzle_vel_y >= nozzle_vel_max[1]:
            print('nozzle_run_time', nozzle_run_time)
            print('nozzle_vel_x', nozzle_vel_x)
            print('nozzle_vel_y', nozzle_vel_y)

            nozzle_vel_y = nozzle_vel_max[1]
            nozzle_run_time = cut[1] / nozzle_vel_y
            t2 = nozzle_run_time + t1
            i2 = np.where(time_steps >= t2)[0][0]
            cut_end_pos = piece_pos_steps + cutstart[0] + cut[0]
            nozzle_cut_x = cut_end_pos[i2] - window_pos[0]
            nozzle_vel_x = nozzle_cut_x / nozzle_run_time

            print('\nnozzle_run_time', nozzle_run_time)
            print('nozzle_cut_x', nozzle_cut_x)
            print('nozzle_vel_x', nozzle_vel_x)
            print('nozzle_vel_y', nozzle_vel_y)

    else:
        # A up or down side cut, or a forward longitudinal cut, or forward angular cut
        # Utilize all the window length
        nozzle_path_x = window_pos[0] + np.array([0, window[0]])

        # Time points when the piece-cut-path-x-component enters and leaves the cut window.
        # Thats when the nozzle is moving and cutting
        i1 = np.where(piece_pos_steps + cutstart[0] >= nozzle_path_x[0])[0][0]
        i2 = np.where(piece_pos_steps + cutstart[0] + cut[0] >= nozzle_path_x[-1])[0][0]
        t1 = time_steps[i1]
        t2 = time_steps[i2]
        nozzle_run_time = t2 - t1

        nozzle_vel_x = window[0] / nozzle_run_time
        nozzle_vel_y = cut[1] / nozzle_run_time

    # nozzle_vel = np.array([nozzlecutx / nozzle_run_time, cut[1] / nozzle_run_time])

    print('v_x', nozzle_vel_x, 'v_y', nozzle_vel_y)

    return nozzle_vel_x, nozzle_vel_y


def line_cut_nozzle_vel3(speed, cutlen, cutangle, nozzlecutx):
    # Rotate the cut line
    cut_angle_rad = np.deg2rad(cutangle)

    cut_angle_cos = np.cos(cut_angle_rad)
    cut_angle_sin = np.sin(cut_angle_rad)

    cut = np.array([[cut_angle_cos, -cut_angle_sin],
                    [cut_angle_sin, cut_angle_cos]]) @ np.array([cutlen, 0])  # [x2, y2]

    nozzle_run_time = (nozzlecutx - cut[0]) / speed  # t_n = (x_n - x2) / v

    return np.array([nozzlecutx / nozzle_run_time, cut[1] / nozzle_run_time])  # [v_n_x, v_n_y]


def line_cut(speed, runtime, fps, piece, window, cutlen, cutangle, cutstart, nozzlecutx):
    # runtime = 2.8  # s
    # fps = 200  # frames per second
    # speed = 200  # mm/s
    # piece = (200, 200)  # mm x mm, length x width
    # window = (180, 250)  # mm x mm, length x width. The lower-left corner is the global origin
    # cutlen = 130  # mm
    # cutangle = 290  # deg
    # cutstart = np.array([20, 140])  # With respect to piece lower-left corner
    # nozzlecutx = 160  # # The nozzle cut length in x-direction

    window_pos = (0, -.5 * window[1])  # Position of the cut window lower-left corner
    fig_size = (12, 5)  # inches, default 6.4, 4.8
    fig_dpi = 200  # default 100
    anim_delay = 1  # ms

    piece_pos_start = -1.6 * piece[0] - window_pos[0]  # The piece travels in the x-direction

    # Rotate the cut line
    cut_angle_rad = np.deg2rad(cutangle)
    cut_angle_cos = np.cos(cut_angle_rad)
    cut_angle_sin = np.sin(cut_angle_rad)
    cut = np.array([[cut_angle_cos, -cut_angle_sin],
                    [cut_angle_sin, cut_angle_cos]]) @ np.array([cutlen, 0])

    cut_path = np.array([cutstart[0] + np.array([0, cut[0]]) + piece_pos_start,
                         cutstart[1] + np.array([0, cut[1]]) - .5 * piece[1]])

    time_steps = np.linspace(0, runtime, int(fps * runtime))
    piece_pos_steps = speed * time_steps + piece_pos_start

    # print(f'delta t {time_steps[1]:.4f}s, delta x {piece_pos_steps[1] - piece_pos_steps[0]:.4f}mm')

    fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)

    ax.set_xlim(1.1 * piece_pos_start, 1.5 * piece[0] + window[0] + window_pos[0])
    ax.set_ylim(-1.3 * window[1] / 2, 1.5 * window[1] / 2)
    ax.set_aspect(1)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    label_text = text.Text(xlim[0] + 10, ylim[1] - 30, ' ')
    ax.add_artist(label_text)

    # The cut window

    win_rect = patches.Rectangle(window_pos, window[0], window[1],
                                 ec='grey', fill=False, ls=':')
    ax.add_patch(win_rect)

    # The piece

    piece_rect = patches.Rectangle((piece_pos_start, -.5 * piece[1]), piece[0], piece[1],
                                   animated=True, fill=True, ec=None, fc='lightgrey', alpha=.4)
    ax.add_patch(piece_rect)

    # The intented cut path on the piece during animation
    piece_cut = lines.Line2D(cut_path[0], cut_path[1], color='lightgrey')
    ax.add_artist(piece_cut)

    # The nozzle

    # Nozzle movement path, within the cut window
    # Allways a vertical line
    nozzle_path = lines.Line2D(window_pos[0] + np.array([0, nozzlecutx]), cut_path[1], color='black')
    ax.add_line(nozzle_path)
    nozzle_path_x = nozzle_path.get_xdata()
    nozzle_path_y = nozzle_path.get_ydata()

    # Nozzle dot, during animation
    nozzle_dot = lines.Line2D([nozzle_path_x[0]], [nozzle_path_y[0]], marker='x', color='red', animated=True)
    ax.add_line(nozzle_dot)

    # A path showing the part of the cut path, which has been cut during each animation step
    nozzle_cut = lines.Line2D([], [], color='red', animated=True, zorder=10)
    ax.add_line(nozzle_cut)

    # Nozzle steps array, reprecenting a dot during animation
    # Initialize the nozzle_steps array. All time-steps are in the idle position at nozzle_path[0, 0].
    nozzle_steps = np.array([nozzle_path_x[0], nozzle_path_y[0]]) \
                   * np.ones((time_steps.size, 2))  # size nx2

    # Time points when the pieces enter and leave the cut window.
    # Thats when the nozzle is moving and cutting
    i1 = np.where(piece_pos_steps + cutstart[0] >= nozzle_path_x[0])[0][0]
    i2 = np.where(piece_pos_steps + cutstart[0] + cut[0] >= nozzle_path_x[-1])[0][0]
    t1 = time_steps[i1]
    t2 = time_steps[i2]
    nozzle_run_time = t2 - t1

    if nozzle_run_time <= 0:
        # It is impossible to do a forward cut (nozzle running faster than the belt in the belt moving direction).
        # The belt speed is too high with respect to cut window length.
        print('The forward cut is impossible.')
        print(f'The allowable nozzle cut-length in the x-direction ({nozzlecutx:.2f} mm) is')
        print(f'shorter than or equal to the length to cut: {nozzlecutx:.2f} mm')
        return

    # Only vertical speed
    nozzle_vel = np.array([[nozzlecutx / nozzle_run_time, cut[1] / nozzle_run_time]])

    nozzle_cut_steps = (nozzle_vel.T * time_steps[1] * np.arange(i2 - i1)).T  # size nx2

    # Translate the nozzle to the beginning of the cut window
    nozzle_cut_steps[:, 0] += window_pos[0]

    # Translate the nozzle at the initial vertical position of cut path
    nozzle_cut_steps[:, 1] += nozzle_path_y[0]

    # Add the "nozzle cut steps array" into the initial "nozzle steps" array at the "nozzle run time-period"
    nozzle_steps[i1:i2] = nozzle_cut_steps
    nozzle_steps[i2:] = nozzle_cut_steps[-1] * np.ones((time_steps.size - i2, 2))
    # print(nozzle_steps[i2:].shape, np.ones((time_steps.size - i2, 2)).shape, nozzle_cut_steps[-1])
    # plt.plot(nozzle_steps[:, 0], nozzle_steps[:, 1], 'o')
    # plt.show()

    ax.set_title(f'Vp {speed} Vnx {nozzle_vel[0, 0]:.2f} Vny {nozzle_vel[0, 1]:.2f} mm/s\n')

    def data_gen():
        for t, p, n in zip(time_steps, piece_pos_steps, nozzle_steps):
            yield t, p, n[0], n[1]

    def update(data):
        t, p, nx, ny = data
        label_text.set_text(f'Travel: {p:.2f}mm, Time: {t:.2f}s')

        # The piece
        piece_rect.set_x(p)
        piece_cut.set_xdata([p + cutstart[0], p + cutstart[0] + cut[0]])

        # The position of the nozzle-dot
        nozzle_dot.set_xdata(nx)
        nozzle_dot.set_ydata(ny)

        # The cut
        if t1 <= t <= t2:
            nozzle_cut.set_data([[piece_cut.get_xdata()[0], nx], [piece_cut.get_ydata()[0], ny]])

        if t >= t2:
            nozzle_cut.set_xdata([p + cutstart[0], p + cutstart[0] + cut[0]])

        return label_text, piece_rect, piece_cut, nozzle_dot, nozzle_cut

    anim = animation.FuncAnimation(fig=fig, func=update, frames=data_gen, interval=anim_delay, blit=True,
                                   repeat=False, save_count=int(runtime * fps))
    # anim.save('angular_cut_1.mp4', fps=200)
    plt.show()

    return nozzle_vel


def _side_cut():
    run_time = 2.8  # s
    frames_per_sec = 160
    piece_speed = 200  # mm/s
    piece_size = (200, 200)  # mm, length x width
    window = (180, 250)  # mm, length x width.
    cut_len = 120  # mm
    cut_angle = 90  # deg, A side-cut is a 90° cut
    cut_start = np.array([130, 30])  # With respect to piece lower-left corner

    nozzle_cut_x = 100  # window[0]  # The nozzle cut length in x-direction
    window_pos = (0, -.5 * window[1])  # Position of the cut window lower-left corner
    fig_size = (12, 5)  # inches, default 6.4, 4.8
    fig_dpi = 200  # default 100
    anim_delay = 1  # ms

    piece_pos_start = -1.6 * piece_size[0] - window_pos[0]  # The piece travels in the x-direction

    # Rotate the cut line
    cut_angle_rad = np.deg2rad(cut_angle)
    cut_angle_cos = np.cos(cut_angle_rad)
    cut_angle_sin = np.sin(cut_angle_rad)
    cut_end = np.array([[cut_angle_cos, -cut_angle_sin],
                        [cut_angle_sin, cut_angle_cos]]) @ np.array([cut_len, 0])  # x and y lengths of cut

    cut_path = np.array([cut_start[0] + np.array([0, cut_end[0]]) + piece_pos_start,
                         cut_start[1] + np.array([0, cut_end[1]]) - .5 * piece_size[1]])

    time_steps = np.linspace(0, run_time, int(frames_per_sec * run_time))
    piece_pos_steps = piece_speed * time_steps + piece_pos_start

    # print(f'delta t {time_steps[1]:.4f}s, delta x {piece_pos_steps[1] - piece_pos_steps[0]:.4f}mm')

    fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)

    ax.set_xlim(1.1 * piece_pos_start, 1.5 * piece_size[0] + window[0] + window_pos[0])
    ax.set_ylim(-1.3 * window[1] / 2, 1.5 * window[1] / 2)
    ax.set_aspect(1)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    label_text = text.Text(xlim[0] + 10, ylim[1] - 30, ' ')
    ax.add_artist(label_text)

    # The cut window

    win_rect = patches.Rectangle(window_pos, window[0], window[1],
                                 ec='grey', fill=False, ls=':')
    ax.add_patch(win_rect)

    # The piece

    piece_rect = patches.Rectangle((-100 - piece_size[0], -piece_size[1] / 2), piece_size[0], piece_size[1],
                                   animated=True, fill=True, ec=None, fc='lightgrey', alpha=.4)
    ax.add_patch(piece_rect)

    # The intented cut path on the piece during animation
    piece_cut = lines.Line2D(cut_path[0], cut_path[1], color='lightgrey')
    ax.add_line(piece_cut)

    # The nozzle

    # Nozzle movement path, within the cut window
    # Use all the cut window length (x-direction) for the cut
    nozzle_path = lines.Line2D(window_pos[0] + np.array([0, nozzle_cut_x]), cut_path[1], color='black')
    ax.add_artist(nozzle_path)
    nozzle_path_x = nozzle_path.get_xdata()
    nozzle_path_y = nozzle_path.get_ydata()

    # Nozzle dot, during animation
    nozzle_dot = lines.Line2D([nozzle_path_x[0]], [nozzle_path_y[0]], marker='x', color='red', animated=True)
    ax.add_artist(nozzle_dot)

    # A path showing the part of the cut path, which has been cut during each animation step
    nozzle_cut = lines.Line2D([], [], color='red', animated=True, zorder=10)
    ax.add_line(nozzle_cut)

    # Nozzle steps array, reprecenting a dot during animation
    # Initialize the nozzle_steps array. All time-steps are in the idle position at nozzle_path[0, 0].
    nozzle_steps = np.array([nozzle_path_x[0], nozzle_path_y[0]]) \
        * np.ones((time_steps.size, 2))  # size nx2

    # Time points when the pieces enter and leave the cut window.
    # Thats when the nozzle is moving and cutting
    i1 = np.where(piece_pos_steps + cut_start[0] >= nozzle_path_x[0])[0][0]
    i2 = np.where(piece_pos_steps + cut_start[0] + cut_end[0] >= nozzle_path_x[-1])[0][0]
    t1 = time_steps[i1]
    t2 = time_steps[i2]
    nozzle_run_time = t2 - t1

    nozzle_vel = np.array([[nozzle_cut_x / nozzle_run_time, cut_end[1] / nozzle_run_time]])

    nozzle_cut_steps = (nozzle_vel.T * time_steps[1] * np.arange(i2 - i1)).T  # size nx2

    # Translate the nozzle to the beginning of the cut window
    nozzle_cut_steps[:, 0] += window_pos[0]

    # Translate the nozzle at the initial vertical position of cut path
    nozzle_cut_steps[:, 1] += nozzle_path_y[0]

    # Add the "nozzle cut steps array" into the initial "nozzle steps" array at the "nozzle run time period"
    nozzle_steps[i1:i2] = nozzle_cut_steps
    nozzle_steps[i2:] = nozzle_cut_steps[-1] * np.ones((time_steps.size - i2, 2))
    # print(nozzle_steps[i2:].shape, np.ones((time_steps.size - i2, 2)).shape, nozzle_cut_steps[-1])
    # plt.plot(nozzle_steps[:, 0], nozzle_steps[:, 1], 'o')
    # plt.show()

    ax.set_title(f'Vp {piece_speed} Vnx {nozzle_vel[0, 0]:.2f} Vny {nozzle_vel[0, 1]:.2f} mm/s\n'
                 + f'Window {window[0]:.2f}x{window[1]:.2f}')

    def data_gen():
        for t, p, n in zip(time_steps, piece_pos_steps, nozzle_steps):
            yield t, p, n[0], n[1]

    def update(data):
        t, p, nx, ny = data
        label_text.set_text(f'Travel: {p:.2f}mm, Time: {t:.2f}s')

        # The piece
        piece_rect.set_x(p)
        piece_cut.set_xdata([p + cut_start[0], p + cut_start[0] + cut_end[0]])

        # The position of the nozzle-dot
        nozzle_dot.set_xdata(nx)
        nozzle_dot.set_ydata(ny)

        # The cut
        if t1 <= t <= t2:
            nozzle_cut.set_data([[piece_cut.get_xdata()[0], nx], [piece_cut.get_ydata()[0], ny]])

        if t >= t2:
            nozzle_cut.set_xdata([p + cut_start[0], p + cut_start[0] + cut_end[0]])

        return label_text, piece_rect, piece_cut, nozzle_dot, nozzle_cut

    anim = animation.FuncAnimation(fig=fig, func=update, frames=data_gen, interval=anim_delay, blit=True,
                                   repeat=False, save_count=int(run_time * frames_per_sec))
    # anim.save('side_cut.mp4', fps=200)
    plt.show()


def _angular_cut():
    run_time = 2.8  # s
    frames_per_sec = 200
    piece_speed = 200  # mm/s
    piece_size = (200, 200)  # mm x mm, length x width
    window = (180, 250)  # mm x mm, length x width. The lower-left corner is the global origin

    cut_len = 130  # mm
    cut_angle = 290  # deg
    cut_start = np.array([20, 140])  # With respect to piece lower-left corner

    nozzle_cut_x = 160  # # The nozzle cut length in x-direction
    window_pos = (0, -.5 * window[1])  # Position of the cut window lower-left corner
    fig_size = (12, 5)  # inches, default 6.4, 4.8
    fig_dpi = 200  # default 100
    anim_delay = 1  # ms

    piece_pos_start = -1.6 * piece_size[0] - window_pos[0]  # The piece travels in the x-direction

    # Rotate the cut line
    cut_angle_rad = np.deg2rad(cut_angle)
    cut_angle_cos = np.cos(cut_angle_rad)
    cut_angle_sin = np.sin(cut_angle_rad)
    cut = np.array([[cut_angle_cos, -cut_angle_sin],
                    [cut_angle_sin, cut_angle_cos]]) @ np.array([cut_len, 0])

    cut_path = np.array([cut_start[0] + np.array([0, cut[0]]) + piece_pos_start,
                         cut_start[1] + np.array([0, cut[1]]) - .5 * piece_size[1]])

    time_steps = np.linspace(0, run_time, int(frames_per_sec * run_time))
    piece_pos_steps = piece_speed * time_steps + piece_pos_start

    # print(f'delta t {time_steps[1]:.4f}s, delta x {piece_pos_steps[1] - piece_pos_steps[0]:.4f}mm')

    fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)

    ax.set_xlim(1.1 * piece_pos_start, 1.5 * piece_size[0] + window[0] + window_pos[0])
    ax.set_ylim(-1.3 * window[1] / 2, 1.5 * window[1] / 2)
    ax.set_aspect(1)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    label_text = text.Text(xlim[0] + 10, ylim[1] - 30, ' ')
    ax.add_artist(label_text)

    # The cut window

    win_rect = patches.Rectangle(window_pos, window[0], window[1],
                                 ec='grey', fill=False, ls=':')
    ax.add_patch(win_rect)

    # The piece

    piece_rect = patches.Rectangle((piece_pos_start, -.5 * piece_size[1]), piece_size[0], piece_size[1],
                                   animated=True, fill=True, ec=None, fc='lightgrey', alpha=.4)
    ax.add_patch(piece_rect)

    # The intented cut path on the piece during animation
    piece_cut = lines.Line2D(cut_path[0], cut_path[1], color='lightgrey')
    ax.add_artist(piece_cut)

    # The nozzle

    # Nozzle movement path, within the cut window
    # Allways a vertical line
    nozzle_path = lines.Line2D(window_pos[0] + np.array([0, nozzle_cut_x]), cut_path[1], color='black')
    ax.add_line(nozzle_path)
    nozzle_path_x = nozzle_path.get_xdata()
    nozzle_path_y = nozzle_path.get_ydata()

    # Nozzle dot, during animation
    nozzle_dot = lines.Line2D([nozzle_path_x[0]], [nozzle_path_y[0]], marker='x', color='red', animated=True)
    ax.add_line(nozzle_dot)

    # A path showing the part of the cut path, which has been cut during each animation step
    nozzle_cut = lines.Line2D([], [], color='red', animated=True, zorder=10)
    ax.add_line(nozzle_cut)

    # Nozzle steps array, reprecenting a dot during animation
    # Initialize the nozzle_steps array. All time-steps are in the idle position at nozzle_path[0, 0].
    nozzle_steps = np.array([nozzle_path_x[0], nozzle_path_y[0]]) \
        * np.ones((time_steps.size, 2))  # size nx2

    # Time points when the pieces enter and leave the cut window.
    # Thats when the nozzle is moving and cutting
    i1 = np.where(piece_pos_steps + cut_start[0] >= nozzle_path_x[0])[0][0]
    i2 = np.where(piece_pos_steps + cut_start[0] + cut[0] >= nozzle_path_x[-1])[0][0]
    t1 = time_steps[i1]
    t2 = time_steps[i2]
    nozzle_run_time = t2 - t1

    # Only vertical speed
    nozzle_vel = np.array([[nozzle_cut_x / nozzle_run_time, cut[1] / nozzle_run_time]])

    nozzle_cut_steps = (nozzle_vel.T * time_steps[1] * np.arange(i2 - i1)).T  # size nx2

    # Translate the nozzle to the beginning of the cut window
    nozzle_cut_steps[:, 0] += window_pos[0]

    # Translate the nozzle at the initial vertical position of cut path
    nozzle_cut_steps[:, 1] += nozzle_path_y[0]

    # Add the "nozzle cut steps array" into the initial "nozzle steps" array at the "nozzle run time-period"
    nozzle_steps[i1:i2] = nozzle_cut_steps
    nozzle_steps[i2:] = nozzle_cut_steps[-1] * np.ones((time_steps.size - i2, 2))
    # print(nozzle_steps[i2:].shape, np.ones((time_steps.size - i2, 2)).shape, nozzle_cut_steps[-1])
    # plt.plot(nozzle_steps[:, 0], nozzle_steps[:, 1], 'o')
    # plt.show()

    ax.set_title(f'Vp {piece_speed} Vnx {nozzle_vel[0, 0]:.2f} Vny {nozzle_vel[0, 1]:.2f} mm/s\n'
                 + f'Window {window[0]:.2f}x{window[1]:.2f}')

    def data_gen():
        for t, p, n in zip(time_steps, piece_pos_steps, nozzle_steps):
            yield t, p, n[0], n[1]

    def update(data):
        t, p, nx, ny = data
        label_text.set_text(f'Travel: {p:.2f}mm, Time: {t:.2f}s')

        # The piece
        piece_rect.set_x(p)
        piece_cut.set_xdata([p + cut_start[0], p + cut_start[0] + cut[0]])

        # The position of the nozzle-dot
        nozzle_dot.set_xdata(nx)
        nozzle_dot.set_ydata(ny)

        # The cut
        if t1 <= t <= t2:
            nozzle_cut.set_data([[piece_cut.get_xdata()[0], nx], [piece_cut.get_ydata()[0], ny]])

        if t >= t2:
            nozzle_cut.set_xdata([p + cut_start[0], p + cut_start[0] + cut[0]])

        return label_text, piece_rect, piece_cut, nozzle_dot, nozzle_cut

    anim = animation.FuncAnimation(fig=fig, func=update, frames=data_gen, interval=anim_delay, blit=True,
                                   repeat=False, save_count=int(run_time * frames_per_sec))
    # anim.save('angular_cut_1.mp4', fps=200)
    plt.show()


def _test_angular_cut_vs_vel_y():
    cutangle = np.linspace(135, 95, 20)
    v_y = np.zeros(cutangle.size)

    for a, i in zip(cutangle, np.arange(cutangle.size)):
        nozzle_vel = line_cut_nozzle_vel(speed=200, runtime=2.8, fps=200, piece=(200, 200), window=(180, 250),
                                         cutlen=120, cutangle=a, cutstart=(150, 20), nozzlecutx=0)
        v_y[i] = nozzle_vel[1]

    plt.plot(cutangle, v_y)
    plt.show()


def _test_line_cut_nozzle_vel():
    # Side up cut
    # nozzle_vel = line_cut_nozzle_vel(speed=200, runtime=2.8, fps=200, piece=(200, 200), window=(180, 250),
    #                                  cutlen=120, cutangle=90, cutstart=(150, 20), nozzlecutx=160)

    # Angular up-back cut
    # nozzle_vel = line_cut_nozzle_vel(speed=200, runtime=2.8, fps=200, piece=(200, 200), window=(180, 250),
    #                                  cutlen=120, cutangle=150, cutstart=(150, 20), nozzlecutx=160)

    # Longitudinal back cut
    # nozzle_vel = line_cut_nozzle_vel(speed=200, runtime=2.8, fps=200, piece=(200, 200), window=(180, 250),
    #                                  cutlen=120, cutangle=180, cutstart=(150, 40), nozzlecutx=0)

    # Angular down-back cut
    nozzle_vel = line_cut_nozzle_vel(speed=200, runtime=2.8, fps=1000000, piece=(200, 200), window=(180, 250),
                                     cutlen=120, cutangle=210, cutstart=(150, 100), nozzlecutx=160)
    print(nozzle_vel)
    nozzle_vel = line_cut_nozzle_vel3(speed=200, cutlen=120, cutangle=210, nozzlecutx=160)
    print(nozzle_vel)

    # nozzlecutx = np.arange(10, 180, 10)
    # # nozzle_vel = np.zeros(nozzlecutx.size)
    # nozzle_vel_y = []
    # for i in range(nozzlecutx.size):
    #     nozzle_vel = line_cut_nozzle_vel(speed=200, runtime=2.8, fps=200, piece=(200, 200), window=(180, 250),
    #                                      cutlen=120, cutangle=90, cutstart=(150, 20), nozzlecutx=nozzlecutx[i])
    #     nozzle_vel_y.append(nozzle_vel[1])
    #     print(i, nozzlecutx[i], nozzle_vel)
    #
    # plt.plot(nozzlecutx, nozzle_vel_y)
    # plt.xlabel('nozzlecutx')
    # plt.ylabel('Vy')
    # plt.show()

    # print(nozzle_vel)


def _test_line_cut():
    # Side up cut
    # nozzle_vel = line_cut(speed=200, runtime=2.8, fps=200, piece=(200, 200), window=(180, 250),
    #                       cutlen=120, cutangle=90, cutstart=(150, 20), nozzlecutx=160)

    # Angular up-back cut
    # nozzle_vel = line_cut(speed=200, runtime=2.8, fps=200, piece=(200, 200), window=(180, 250),
    #                       cutlen=120, cutangle=150, cutstart=(150, 20), nozzlecutx=160)

    # line_arr = np.array([[180, 160], [40, 170]])
    # cutlen = np.linalg.norm(np.diff(line_arr))
    # cutangle = np.degrees(np.arccos(np.diff(line_arr).T @ np.array([1, 0]) / cutlen))

    # line_arr = np.array([[160, 30], [170, 100]])
    # cutlen = np.linalg.norm(np.diff(line_arr))
    # cutangle = 360-np.degrees(np.arccos(np.diff(line_arr).T @ np.array([1, 0]) / cutlen))

    # nozzle_vel = line_cut(speed=200, runtime=2.8, fps=400, piece=(200, 200), window=(180, 250),
    #                       cutlen=cutlen, cutangle=cutangle[0], cutstart=(line_arr[0, 0], line_arr[1, 0]),
    #                       nozzlecutx=90)

    # Longitudinal back cut
    # nozzle_vel = line_cut(speed=200, runtime=2.8, fps=200, piece=(200, 200), window=(180, 250),
    #                       cutlen=120, cutangle=180, cutstart=(150, 40), nozzlecutx=0)

    # Angular down-back cut
    # nozzle_vel = line_cut(speed=200, runtime=2.8, fps=200, piece=(200, 200), window=(180, 250),
    #                       cutlen=120, cutangle=210, cutstart=(150, 100), nozzlecutx=160)

    # Side down cut
    # nozzle_vel = line_cut(speed=200, runtime=2.8, fps=200, piece=(200, 200), window=(180, 250),
    #                       cutlen=120, cutangle=270, cutstart=(50, 140), nozzlecutx=160)

    # Angular down forward cut
    # nozzle_vel = line_cut(speed=200, runtime=2.8, fps=200, piece=(200, 200), window=(180, 250),
    #                       cutlen=120, cutangle=310, cutstart=(50, 140), nozzlecutx=160)

    # Longitudinal forward cut
    # nozzle_vel = line_cut(speed=200, runtime=2.8, fps=400, piece=(200, 200), window=(180, 250),
    #                       cutlen=120, cutangle=0, cutstart=(30, 140), nozzlecutx=160)

    line_arr = np.array([[30, 75], [40, 40]])
    cutlen = np.linalg.norm(np.diff(line_arr))
    cutangle = np.degrees(np.arccos(np.diff(line_arr).T @ np.array([1, 0]) / cutlen))

    nozzle_vel = line_cut(speed=200, runtime=2.8, fps=1000, piece=(200, 200), window=(180, 250),
                          cutlen=cutlen, cutangle=cutangle[0], cutstart=(line_arr[0, 0], line_arr[1, 0]),
                          nozzlecutx=45)

    # Angular up forward cut
    # nozzle_vel = line_cut(speed=200, runtime=2.8, fps=200, piece=(200, 200), window=(180, 250),
    #                       cutlen=120, cutangle=30, cutstart=(30, 80), nozzlecutx=160)


def _test_polyline_cut():
    # Lines
    # cutpath = [[180, 160], [40, 170]]  # Angular back, steep line
    # cutpath = [[180, 80], [120, 20]]  # Angular back
    # cutpath = [[120, 180, 80], [120, 120, 20]]  # Longitudinal forward and angular back
    # cutpath = [[180, 160, 70], [40, 170, 40]]  # Double angular back, first line is steep
    # cutpath = [[180, 90, 70], [40, 170, 40]]  # Double angular back, second line is steep
    # cutpath = [[80, 180], [120, 20]]  # Angular forward
    # cutpath = [[10, 180, 80], [120, 120, 20]]
    # cutpath = [[190, 150, 180, 140, 120, 100, 110, 80, 70], [150, 150, 50, 50, 150, 150, 50, 50, 150]]
    # cutpath = [
    #     [200, 165, 177, 200, 165, 177, 153, 141],
    #     [104, 104, 200, 95, 95, 0, 0, 95]]
    # polyline_cut(speed=600, runtime=.8, fps=6000, piece=(200, 200),
    #              cutpath=cutpath, nozzlevelmax=(1000, 1000))
    # polyline_cut2(speed=600, runtime=.8, fps=6000, piece=(200, 200),
    #               cutpath=cutpath, nozzlevelmax=(1000, 1000))

    # Rectangle
    # polyline_cut2(speed=200, runtime=2.8, fps=500, piece=(200, 200),
    #              cutpath=[[120, 120, 30, 30, 120], [40, 170, 170, 40, 40]], nozzlevelmax=(1000, 1000))

    # Diamond
    polyline_cut2(speed=200, runtime=1.5, fps=1000, piece=(200, 200),
                 cutpath=[[140, 80, 75, 80, 140], [80, 185, 80, 30, 80]], nozzlevelmax=(1000, 1000))

    # Two rectangles
    # cutpath = np.array([[0.168439056534295, 0.168439056534295, 0.126043180805086, 0.126043180805086, 0.168439056534295, 9.64525899900049E-02, 9.64525899900049E-02, 4.76540718037317E-02, 4.76540718037317E-02, 9.64525899900049E-02],
    #                     [0.077331184814178, 0.139108032305311, 0.139108032305311, 0.077331184814178, 0.077331184814178, 0.077331184814178, 0.158662048457967, 0.158662048457967, 0.077331184814178, 0.077331184814178]]) * 1000
    # iscut = [1, 1, 1, 1, 0, 1, 1, 1, 1]
    # polyline_cut(speed=600, runtime=.9, fps=6000, piece=(200, 200), cutpath=cutpath, nozzlevelmax=(1000, 1000),
    #              iscut=iscut, doanim=False)

    # Two diamonds
    # cutpath = np.array([[0.160998147732842, 0.139886732063603, 0.104239587572921, 0.134695400341659, 0.160998147732842, 8.22629499500248E-02, 0.061151534280786, 2.55043897901041E-02, 0.055960202558842, 8.22629499500248E-02],
    #                     [0.133743656192636, 0.179773464127206, 0.133743656192636, 9.29051799800099E-02, 0.133743656192636, 0.127514058126303, 0.173543866060873, 0.127514058126303, 8.66755819136771E-02, 0.127514058126303]]) * 1000
    # iscut = [1, 1, 1, 1, 0, 1, 1, 1, 1]
    # polyline_cut(speed=600, runtime=.9, fps=6000, piece=(200, 200), cutpath=cutpath, nozzlevelmax=(1000, 1000),
    #              iscut=iscut, doanim=False)

    # Twelve-dided polygon
    # cutpath = np.array([[0.157344696306295, 0.157344696306295, 0.145998469127282, 0.126346227179013, 0.103653772820987, 0.084001530872718, 7.26553036937051E-02, 7.26553036937051E-02, 0.084001530872718, 0.103653772820987, 0.126346227179013, 0.145998469127282, 0.157344696306295],
    #                     [0.111653772820987, 0.134346227179013, 0.153998469127282, 0.165344696306295, 0.165344696306295, 0.153998469127282, 0.134346227179013, 0.111653772820987, 0.092001530872718, 8.06553036937051E-02, 8.06553036937051E-02, 0.092001530872718, 0.111653772820987]]) * 1000
    # polyline_cut(speed=600, runtime=.8, fps=6000, piece=(200, 200), cutpath=cutpath, nozzlevelmax=(1000, 1000),
    #              doanim=False)

    # Strip cuts
    # polyline_cut(speed=200, runtime=2.8, fps=1000, piece=(200, 200),
    #              cutpath=[[160, 160, 120, 120, 80, 80, 40, 40],
    #                       [200, 0, 0, 200, 200, 0, 0, 200]], nozzlevelmax=(1000, 1000))

    # Simple butterfly cutpath
    # cutpath = [
    #     [200, 165, 177, 200, 165, 177, 153, 141, 165, 165, 141, 153, 125, 108, 141, 141, 108, 125, 123, 54, 10, 2, 108,
    #      108, 2, 10, 54, 123, 2, 0, 2, 0],
    #     [104, 104, 200, 95, 95, 0, 0, 95, 95, 104, 104, 200, 200, 104, 104, 95, 95, 0, 14, 6, 46, 95, 95, 104, 104, 154,
    #      194, 186, 104, 104, 95, 95]]
    # iscut = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1]
    # polyline_cut(speed=600, runtime=2.5, fps=6000, piece=(200, 200),
    #              cutpath=cutpath, iscut=iscut, nozzlevelmax=(1000, 1000), doanim=False)


if __name__ == '__main__':
    # _test_line_cut_nozzle_vel()
    # _test_line_cut()
    # _test_angular_cut_vs_vel_y()
    _test_polyline_cut()

    # line_cut_nozzle_vel2(speed=200, runtime=2.8, fps=1000, piece=(200, 200), window=(180, 250),
    #                      cutlen=120, cutangle=95, cutstart=(150, 40), nozzle_vel_max=1000)
    # line_cut(speed=200, runtime=2.8, fps=1000, piece=(200, 200), window=(180, 250),
    #          cutlen=120, cutangle=95, cutstart=(150, 40), nozzlecutx=14)

