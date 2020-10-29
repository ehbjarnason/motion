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


def side_cut():
    run_time = 2.8  # s
    frames_per_sec = 400
    piece_speed = 200  # mm/s
    piece_size = (200, 200)  # mm x mm, length x width
    window = (180, 250)  # mm x mm, length x width.
    cut_len = 120  # mm
    cut_angle = 90  # deg, A side-cut is a 90° cut
    cut_start = np.array([130, 30])  # With respect to piece lower-left corner

    cut_len_x = window[0]  # The nozzle cut length in x-direction
    window_pos = (0, -.5 * window[1])  # Position of the cut window lower-left corner

    piece_pos_start = -1.6 * piece_size[0] - window_pos[0]  # The piece travels in the x-direction

    # Rotate the cut line
    cut_angle_rad = np.deg2rad(cut_angle)
    cut_angle_cos = np.cos(cut_angle_rad)
    cut_angle_sin = np.sin(cut_angle_rad)
    cut_end = np.array([[cut_angle_cos, -cut_angle_sin],
                        [cut_angle_sin, cut_angle_cos]]) @ np.array([cut_len, 0])

    cut_path = np.array([cut_start[0] + np.array([0, cut_end[0]]) + piece_pos_start,
                         cut_start[1] + np.array([0, cut_end[1]]) - .5 * piece_size[1]])

    time_steps = np.linspace(0, run_time, int(frames_per_sec * run_time))
    piece_pos_steps = piece_speed * time_steps + piece_pos_start

    # print(f'delta t {time_steps[1]:.4f}s, delta x {piece_pos_steps[1] - piece_pos_steps[0]:.4f}mm')

    fig, ax = plt.subplots()

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
    cut_path_line = lines.Line2D(cut_path[0], cut_path[1], color='lightgrey')
    ax.add_artist(cut_path_line)

    # The nozzle

    # Nozzle movement path, within the cut window
    # Use all the cut window length (x-direction) for the cut
    nozzle_path_line = lines.Line2D(window_pos[0] + np.array([0, cut_len_x]), cut_path[1], color='black')
    ax.add_artist(nozzle_path_line)
    nozzle_path_line_x = nozzle_path_line.get_xdata()
    nozzle_path_line_y = nozzle_path_line.get_ydata()

    # Nozzle dot, during animation
    nozzle_line = lines.Line2D([nozzle_path_line_x[0]], [nozzle_path_line_y[0]],
                               marker='x', color='red', animated=True)
    ax.add_artist(nozzle_line)

    # A path showing the part of the cut path, which has been cut during each animation step
    nozzle_cut_line = lines.Line2D([nozzle_path_line_x[0]], [nozzle_path_line_y[0]],
                                   color='red', animated=True, zorder=10)
    nozzle_cut_line.set_visible(False)
    ax.add_line(nozzle_cut_line)

    # Nozzle steps array, reprecenting a dot during animation
    # Initialize the nozzle_steps array. All time-steps are in the idle position at nozzle_path_line[0, 0].
    nozzle_steps = np.array([nozzle_path_line_x[0], nozzle_path_line_y[0]]) \
        * np.ones((time_steps.size, 2))  # size nx2

    # Time points when the pieces enter and leave the cut window.
    # Thats when the nozzle is moving and cutting
    i1 = np.where(piece_pos_steps + cut_start[0] >= nozzle_path_line_x[0])[0][0]
    i2 = np.where(piece_pos_steps + cut_start[0] + cut_end[0] >= nozzle_path_line_x[-1])[0][0]
    t1 = time_steps[i1]
    t2 = time_steps[i2]
    nozzle_run_time = t2 - t1

    # nozzle_vel = np.array([[window[0] / nozzle_run_time, cut_len / nozzle_run_time]])
    nozzle_vel = np.array([[cut_len_x / nozzle_run_time, cut_end[1] / nozzle_run_time]])

    nozzle_cut_steps = (nozzle_vel.T * time_steps[1] * np.arange(i2 - i1)).T  # size nx2

    # Translate the nozzle to the beginning of the cut window
    nozzle_cut_steps[:, 0] += window_pos[0]

    # Translate the nozzle at the initial vertical position of cut path
    nozzle_cut_steps[:, 1] += nozzle_path_line_y[0]

    # Add the "nozzle cut steps array" into the initial "nozzle steps" array at the "nozzle run time period"
    nozzle_steps[i1:i2] = nozzle_cut_steps
    nozzle_steps[i2:] = nozzle_cut_steps[-1] * np.ones((time_steps.size - i2, 2))
    # print(nozzle_steps[i2:].shape, np.ones((time_steps.size - i2, 2)).shape, nozzle_cut_steps[-1])
    # plt.plot(nozzle_steps[:, 0], nozzle_steps[:, 1], 'o')
    # plt.show()

    ax.set_title(f'Speeds (mm/s): piece {piece_speed}, '
                 + f'nozzleX {nozzle_vel[0, 0]:.2f}, nozzleY {nozzle_vel[0, 1]:.2f}\n'
                 + f'Cut window {window[0]:.2f} x {window[1]:.2f}')

    def data_gen():
        for t, p, n in zip(time_steps, piece_pos_steps, nozzle_steps):
            yield t, p, n[0], n[1]

    def update(data):
        t, p, nx, ny = data
        label_text.set_text(f'Travel: {p:.2f}mm, Time: {t:.2f}s')

        # The piece
        piece_rect.set_x(p)
        cut_path_line.set_xdata([p + cut_start[0], p + cut_start[0] + cut_end[0]])

        # The position of the nozzle-dot
        nozzle_line.set_xdata(nx)
        nozzle_line.set_ydata(ny)

        # The cut
        if t1 <= t <= t2:
            # Modify the line
            nozzle_cut_line_ydata = list(nozzle_cut_line.get_ydata())
            nozzle_cut_line_ydata.append(ny)
            nozzle_cut_line.set_ydata(nozzle_cut_line_ydata)

            if cut_len_x == 0:
                nozzle_cut_line_xdata = list(nozzle_cut_line.get_xdata())
                nozzle_cut_line_xdata.append(-(p + cut_start[0] - 2 * window_pos[0]))
                nozzle_cut_line.set_xdata(nozzle_cut_line_xdata)

        if t >= t1:
            nozzle_cut_line.set_visible(True)
            nozzle_cut_line.set_transform(
                transforms.Affine2D().translate(p + cut_start[0] - window_pos[0], 0) + ax.transData)

        return piece_rect, label_text, cut_path_line, nozzle_line, nozzle_cut_line

    anim = animation.FuncAnimation(fig=fig, func=update, frames=data_gen, interval=1, blit=True,
                                   repeat=False, save_count=int(run_time * frames_per_sec))
    # anim.save('side_cut.mp4', fps=200)
    plt.show()


def angular_cut():
    run_time = 2.8  # s
    frames_per_sec = 400
    piece_speed = 200  # mm/s
    piece_size = (200, 200)  # mm x mm, length x width
    window = (180, 250)  # mm x mm, length x width. The lower-left corner is the global origin

    cut_len = 130  # mm
    cut_angle = 130  # deg
    cut_start = np.array([180, 50])  # With respect to piece lower-left corner

    cut_len_x = 0  # # The nozzle cut length in x-direction
    window_pos = (0, -.5 * window[1])  # Position of the cut window lower-left corner

    piece_pos_start = -1.6 * piece_size[0] - window_pos[0]  # The piece travels in the x-direction

    # Rotate the cut line
    cut_angle_rad = np.deg2rad(cut_angle)
    cut_angle_cos = np.cos(cut_angle_rad)
    cut_angle_sin = np.sin(cut_angle_rad)
    cut_end = np.array([[cut_angle_cos, -cut_angle_sin],
                        [cut_angle_sin, cut_angle_cos]]) @ np.array([cut_len, 0])

    cut_path = np.array([cut_start[0] + np.array([0, cut_end[0]]) + piece_pos_start,
                         cut_start[1] + np.array([0, cut_end[1]]) - .5 * piece_size[1]])

    time_steps = np.linspace(0, run_time, int(frames_per_sec * run_time))
    piece_pos_steps = piece_speed * time_steps + piece_pos_start

    # print(f'delta t {time_steps[1]:.4f}s, delta x {piece_pos_steps[1] - piece_pos_steps[0]:.4f}mm')

    fig, ax = plt.subplots(figsize=(12, 5), dpi=200)

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
    cut_path_line = lines.Line2D(cut_path[0], cut_path[1], color='lightgrey')
    ax.add_artist(cut_path_line)

    # The nozzle

    # Nozzle movement path, within the cut window
    # Allways a vertical line
    nozzle_path_line = lines.Line2D(window_pos[0] + np.array([0, cut_len_x]), cut_path[1], color='black')
    ax.add_artist(nozzle_path_line)
    nozzle_path_line_x = nozzle_path_line.get_xdata()
    nozzle_path_line_y = nozzle_path_line.get_ydata()

    # Nozzle dot, during animation
    nozzle_line = lines.Line2D([nozzle_path_line_x[0]], [nozzle_path_line_y[0]],
                               marker='x', color='red', animated=True)
    ax.add_artist(nozzle_line)

    # A path showing the part of the cut path, which has been cut during each animation step
    nozzle_cut_line = lines.Line2D([nozzle_path_line_x[0]], [nozzle_path_line_y[0]],
                                   color='red', animated=True, zorder=10)
    nozzle_cut_line.set_visible(False)
    ax.add_line(nozzle_cut_line)

    # Nozzle steps array, reprecenting a dot during animation
    # Initialize the nozzle_steps array. All time-steps are in the idle position at nozzle_path_line[0, 0].
    nozzle_steps = np.array([nozzle_path_line_x[0], nozzle_path_line_y[0]]) \
        * np.ones((time_steps.size, 2))  # size nx2

    # Time points when the pieces enter and leave the cut window.
    # Thats when the nozzle is moving and cutting
    i1 = np.where(piece_pos_steps + cut_start[0] >= nozzle_path_line_x[0])[0][0]
    i2 = np.where(piece_pos_steps + cut_start[0] + cut_end[0] >= nozzle_path_line_x[-1])[0][0]
    t1 = time_steps[i1]
    t2 = time_steps[i2]
    nozzle_run_time = t2 - t1

    # Only vertical speed
    # nozzle_vel = np.array([[0, piece_speed * (cut_end[1] / -cut_end[0])]])
    nozzle_vel = np.array([[cut_len_x / nozzle_run_time, cut_end[1] / nozzle_run_time]])

    nozzle_cut_steps = (nozzle_vel.T * time_steps[1] * np.arange(i2 - i1)).T  # size nx2

    # Translate the nozzle to the beginning of the cut window
    nozzle_cut_steps[:, 0] += window_pos[0]

    # Translate the nozzle at the initial vertical position of cut path
    nozzle_cut_steps[:, 1] += nozzle_path_line_y[0]

    # Add the "nozzle cut steps array" into the initial "nozzle steps" array at the "nozzle run time-period"
    nozzle_steps[i1:i2] = nozzle_cut_steps
    nozzle_steps[i2:] = nozzle_cut_steps[-1] * np.ones((time_steps.size - i2, 2))
    # print(nozzle_steps[i2:].shape, np.ones((time_steps.size - i2, 2)).shape, nozzle_cut_steps[-1])
    # plt.plot(nozzle_steps[:, 0], nozzle_steps[:, 1], 'o')
    # plt.show()

    ax.set_title(f'Speeds (mm/s): piece {piece_speed}, '
                 + f'nozzleX {nozzle_vel[0, 0]:.2f}, nozzleY {nozzle_vel[0, 1]:.2f}\n'
                 + f'Cut window {window[0]:.2f} x {window[1]:.2f}')

    def data_gen():
        for t, p, n in zip(time_steps, piece_pos_steps, nozzle_steps):
            yield t, p, n[0], n[1]

    def update(data):
        t, p, nx, ny = data
        label_text.set_text(f'Travel: {p:.2f}mm, Time: {t:.2f}s')

        # The piece
        piece_rect.set_x(p)
        cut_path_line.set_xdata([p + cut_start[0], p + cut_start[0] + cut_end[0]])

        # The position of the nozzle-dot
        nozzle_line.set_xdata(nx)
        nozzle_line.set_ydata(ny)
        print(nx)
        # The cut
        if t1 <= t <= t2:
            nozzle_cut_line.set_visible(True)

            # Modify the line
            nozzle_cut_line_ydata = list(nozzle_cut_line.get_ydata())
            nozzle_cut_line_ydata.append(ny)
            nozzle_cut_line.set_ydata(nozzle_cut_line_ydata)

            if cut_len_x == 0:
                # An angular cut, only with movement along the y-axis
                nozzle_cut_line_xdata = list(nozzle_cut_line.get_xdata())
                nozzle_cut_line_xdata.append(-(p + cut_start[0] - 2 * window_pos[0]))
                nozzle_cut_line.set_xdata(nozzle_cut_line_xdata)

            # elif cut_angle != 90 or cut_angle != 180:
            #     nozzle_cut_line.set_transform(
            #         transforms.Affine2D().translate(p + cut_start[0] - window_pos[0], 0) + ax.transData)
            #
            #     # An angualar cut, with movement along both the x- and y-axis
            #     nozzle_cut_line_xdata = list(nozzle_cut_line.get_xdata())
            #     # nozzle_cut_line_xdata.append(-(p + cut_start[0] - 2 * window_pos[0]))
            #     nozzle_cut_line_xdata.append(nx)
            #     nozzle_cut_line.set_xdata(nozzle_cut_line_xdata)

        if t >= t1:
            nozzle_cut_line.set_transform(
                transforms.Affine2D().translate(p + cut_start[0] - window_pos[0], 0) + ax.transData)

        return piece_rect, label_text, cut_path_line, nozzle_line, nozzle_cut_line

    anim = animation.FuncAnimation(fig=fig, func=update, frames=data_gen, interval=1, blit=True,
                                   repeat=False, save_count=int(run_time * frames_per_sec))
    # anim.save('angular_cut_1.mp4', fps=200)
    plt.show()


def _test_skew_transformation():
    # Cut window
    x_w = 2  # mm

    # Transformation matrix for skewing
    k = np.array([[1, x_w, -x_w], [0, 1, 1], [0, 0, 1]])

    # Cut path: straight vertical line
    # c = np.array([np.ones(10), np.arange(10), np.zeros(10)])

    # Cut path: L-shaped path
    c = np.array([[3, 2, 1, 0, 0, 0, 0],
                  [3, 3, 3, 3, 2, 1, 0],
                  np.zeros(7)])

    # Cut path: circle
    # r = 2
    # theta = np.linspace(0, np.pi)
    # c = np.array([r * np.cos(theta), r * np.sin(theta), np.zeros(theta.size)])

    # Cut path: rectangle
    # c = np.array([[0, 1, 2, 3, 3, 3, 3, 2, 1, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 1, 2, 3, 3, 3, 3, 2, 1, 0],
    #               np.zeros(13)])

    c_k = k @ c

    print(c)
    print(c_k)

    plt.plot(c[0], c[1], 'o')
    plt.plot(c_k[0], c_k[1], 'o')
    plt.axis('equal')
    plt.show()


def _test_skew_transformation2():
    pos = (-3, -2)
    width = 3
    height = 4
    x_skew = 1

    fig, ax = plt.subplots()

    rect = patches.Rectangle(pos, width, height, fill=False)
    ax.add_patch(rect)

    rect_skewed = patches.Rectangle(pos, width, height, fill=False, color='blue')
    ax.add_patch(rect_skewed)

    x_shear_angle = np.arctan(x_skew / height)
    print(x_shear_angle)

    trans = transforms.Affine2D().translate(-pos[0], -pos[1])
    trans += transforms.Affine2D().skew(x_shear_angle, 0)
    trans += transforms.Affine2D().translate(pos[0], pos[1])

    rect_skewed.set_transform(trans + ax.transData)

    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_aspect(1)

    plt.show()


def _test_resampling():
    import scipy.ndimage as ndimage

    x = np.arange(9).reshape(3, 3)
    print(x)

    print('Resampled by a factor of 2 with nearest interpolation:')
    print(ndimage.zoom(x, 2, order=0))

    print('Resampled by a factor of 2 with bilinear interpolation:')
    print(ndimage.zoom(x, 2, order=1))

    print('Resampled by a factor of 2 with cubic interpolation:')
    print(ndimage.zoom(x, 2, order=3))


if __name__ == '__main__':
    # side_cut()
    angular_cut()
