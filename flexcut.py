import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.text as text


def moving_cut_path():
    # Cut window
    x_w = 2  # mm

    # Transformation matrix
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


def moving_piece():
    piece_size = (200, 200)  # mm
    cut_window = (300, 300)  # mm

    pos_steps = np.arange(-100 - piece_size[0], 100 + piece_size[0], 1)
    print(f'dist: {pos_steps[-1] - pos_steps[0]} mm')

    def pos_gen():
        for p in pos_steps:
            yield p

    def update_piece_pos(pos):
        rect_piece.set_x(pos)
        return rect_piece,

    fig, ax = plt.subplots()

    ax.set_xlim(-100 - piece_size[0], 100 + piece_size[0] + cut_window[0])
    ax.set_ylim(-100 - cut_window[1] / 2, 100 + cut_window[1] / 2)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect(1)

    rect_piece = patches.Rectangle((-100 - piece_size[0], -piece_size[1] / 2), piece_size[0], piece_size[1],
                                     animated=True, fill=True, ec=None, fc='lightgrey')
    ax.add_patch(rect_piece)

    rect_win = patches.Rectangle((0, -cut_window[1] / 2), cut_window[0], cut_window[1],
                                 ec='grey', fill=False, ls=':')
    ax.add_patch(rect_win)

    anim = animation.FuncAnimation(fig=fig, func=update_piece_pos, frames=pos_gen, interval=1, blit=True, repeat=True)
    plt.show()


def moving_piece_speed():
    belt_speed = 200  # mm/s
    piece_size = (200, 200)  # mm x mm, length x width
    cut_window = (300, 300)  # mm x mm, length x width
    frames_per_sec = 2
    cut_path = np.array([np.zeros(100), np.arange(100)])

    pos_start, pos_end = -1.6 * piece_size[0], 1.6 * piece_size[0]
    travel_dist = pos_end - pos_start

    # Move the cut path onto the piece-area
    # cut_path[0] += pos_start + 50
    # cut_path[1] -= 10

    pos_steps = np.linspace(pos_start, pos_end, int(travel_dist * frames_per_sec))

    # travel_time = travel_dist / belt_speed
    # print(f'travel dist: {travel_dist} mm, travel time: {travel_time} sec')

    time_steps = (pos_steps - pos_steps[0]) / belt_speed  # sec
    print(pos_steps.size)
    # print(pos_steps, pos_steps.size)
    # print(time_steps)

    def data_gen():
        for p, t in zip(pos_steps, time_steps):
            yield p, t

    def update(data):
        rect_piece.set_x(data[0])
        text_label.set_text(f'pos: {data[0]:.2f}mm, time: {data[1]:.2f}s')
        return [rect_piece, text_label]

    fig, ax = plt.subplots()

    ax.set_xlim(-100 - piece_size[0], 100 + piece_size[0] + cut_window[0])
    ax.set_ylim(-100 - cut_window[1] / 2, 100 + cut_window[1] / 2)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect(1)

    rect_piece = patches.Rectangle((-100 - piece_size[0], -piece_size[1] / 2), piece_size[0], piece_size[1],
                                   animated=True, fill=True, ec=None, fc='lightgrey')
    ax.add_patch(rect_piece)

    rect_win = patches.Rectangle((0, -cut_window[1] / 2), cut_window[0], cut_window[1],
                                 ec='grey', fill=False, ls=':')
    ax.add_patch(rect_win)

    text_label = text.Text(0, 200, ' ')
    ax.add_artist(text_label)

    poly_cut_path = patches.Polygon(cut_path.T, closed=False)
    ax.add_patch(poly_cut_path)

    anim = animation.FuncAnimation(fig=fig, func=update, frames=data_gen, interval=1, blit=True, repeat=True)
    plt.show()


def test():
    cut_path = np.array([np.zeros(100), np.arange(100)])
    fig, ax = plt.subplots()
    poly_cut_path = patches.Polygon([[0, 0], [0.5, 0.8], [0.6, 0.4], [0.3, 0.2]], closed=False)
    ax.add_patch(poly_cut_path)
    plt.show()


if __name__ == '__main__':
    # moving_cut_path()
    # moving_piece()
    # moving_piece_speed()
    test()
