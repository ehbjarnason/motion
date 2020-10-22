import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.text as text
import matplotlib.lines as lines
import matplotlib.transforms as transforms


def side_cut():
    run_time = 3.4  # s
    frames_per_sec = 400
    piece_speed = 200  # mm/s
    piece_size = (200, 200)  # mm x mm, length x width
    cut_window = (300, 300)  # mm x mm, length x width. The lower-left corner is the global origin
    cut_path = np.array([100 + np.ones(2), 50 + np.array([0, 100])])  # From the piece lower-left corner

    piece_pos_start = -1.6 * piece_size[0]

    time_steps = np.linspace(0, run_time, int(frames_per_sec * run_time))
    piece_pos_steps = piece_speed * time_steps + piece_pos_start

    # print(f'delta t {time_steps[1]:.4f}s, delta x {piece_pos_steps[1] - piece_pos_steps[0]:.4f}mm')

    def data_gen():
        for t, p, z in zip(time_steps, piece_pos_steps, nozzle_steps):
            yield t, p, z[0], z[1]

    def update(data):
        t, p, zx, zy = data
        piece_rect.set_x(p)
        cut_path_line.set_xdata(cut_path[0] + p)
        label_text.set_text(f'Speed: {piece_speed}mm/s, pos: {p:.2f}mm, time: {t:.2f}s')
        nozzle_line.set_xdata(zx)
        nozzle_line.set_ydata(zy)
        return piece_rect, label_text, cut_path_line, nozzle_line

    fig, ax = plt.subplots()

    ax.set_xlim(-100 - piece_size[0], 100 + piece_size[0] + cut_window[0])
    ax.set_ylim(-100 - cut_window[1] / 2, 100 + cut_window[1] / 2)
    ax.set_aspect(1)

    label_text = text.Text(-100, 200, ' ')
    ax.add_artist(label_text)

    piece_rect = patches.Rectangle((-100 - piece_size[0], -piece_size[1] / 2), piece_size[0], piece_size[1],
                                   animated=True, fill=True, ec=None, fc='lightgrey', alpha=.4)
    ax.add_patch(piece_rect)

    win_rect = patches.Rectangle((0, -cut_window[1] / 2), cut_window[0], cut_window[1],
                                 ec='grey', fill=False, ls=':')
    ax.add_patch(win_rect)

    cut_path_line = lines.Line2D(cut_path[0] + piece_pos_start, cut_path[1] - .5 * piece_size[1], color='lightgrey')
    ax.add_artist(cut_path_line)

    # Create the nozzle path, transformed from the piece cut path
    nozzle_path_line = lines.Line2D(cut_path_line.get_xdata() - cut_path_line.get_xdata()[0],
                                    cut_path_line.get_ydata(), color='blue')
    ax.add_artist(nozzle_path_line)

    xdata = nozzle_path_line.get_xdata()
    ydata = nozzle_path_line.get_ydata()

    x_shear_angle = np.arctan(cut_window[0] / np.abs(np.max(ydata) - np.min(ydata)))

    trans = transforms.Affine2D().translate(-xdata[0], -ydata[0])  # To zero point
    trans += transforms.Affine2D().skew(x_shear_angle, 0)  # Skew
    trans += transforms.Affine2D().translate(xdata[0], ydata[0])  # Back to original pos

    nozzle_path_line.set_transform(trans + ax.transData)

    # Create nozzle steps array
    nozzle_steps = np.array([nozzle_path_line.get_xdata()[0], nozzle_path_line.get_ydata()[0]]) \
                            * np.ones((time_steps.size, 2))  # size nx2
    # print(nozzle_steps.shape, nozzle_steps[0])

    nozzle_line = lines.Line2D([nozzle_steps[0, 0]], [nozzle_steps[0, 1]], marker='x', color='red',
                                animated=True)
    ax.add_artist(nozzle_line)

    # time points when the pieces enter and leave the cut window
    t1_ind = np.where(piece_pos_steps + cut_path[0, 0] >= 0)[0][0]
    t2_ind = np.where(piece_pos_steps + cut_path[0, 0] <= cut_window[0])[0][-1]
    t1 = time_steps[t1_ind]
    t2 = time_steps[t2_ind]
    nozzle_run_time = t2 - t1
    nozzle_speed = np.array([[cut_window[0] / nozzle_run_time,
                            (cut_path[1, -1] - cut_path[1, 0]) / nozzle_run_time]])

    nozzle_cut_steps = (nozzle_speed.T * time_steps[1] * np.arange(t2_ind - t1_ind)).T  # size nx2
    # print(nozzle_cut_steps.shape)

    # nozzle_cut_steps = np.pad(nozzle_cut_steps, ((t1_ind - 1, time_steps.size - t1_ind + 1),), mode='edge')
    nozzle_cut_steps[:, 1] -= cut_path[1, 0]
    nozzle_steps[t1_ind:t2_ind] = nozzle_cut_steps
    # print(nozzle_steps[t2_ind:].shape, np.ones((time_steps.size - t2_ind, 2)).shape, nozzle_cut_steps[-1])
    nozzle_steps[t2_ind:] = nozzle_cut_steps[-1] * np.ones((time_steps.size - t2_ind, 2))
    # plt.plot(nozzle_steps[:, 0], nozzle_steps[:, 1], 'o')
    # plt.show()

    # print(nozzle_steps.shape, nozzle_cut_steps.shape)
    # nozzle_steps_2 = np.pad(nozzle_cut_steps.T, 10, mode='edge')
    # print(nozzle_steps_2)
    # print(nozzle_steps_2.shape)

    anim = animation.FuncAnimation(fig=fig, func=update, frames=data_gen, interval=1, blit=True, repeat=False)
    plt.show()


def test_skew_transformation():
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


def test_skew_transformation2():
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


def test_resampling():
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
    side_cut()
