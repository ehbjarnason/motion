
import matplotlib as mpl

from motion import *
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import matplotlib.patches as pch
import mpl_toolkits.axes_grid1
import matplotlib.widgets
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


mpl.use('TkAgg')


class PiecesMotionProfiles:
    """Abstract class"""


def sep(num_pieces, piece_width,  base_speed, sep_profile, travel_dist, num_points):
    """Simulate separation after cutting.

    Sep_profile: A motion profile object.

    Returns a dictionary:
    {   'num_pieces': num_pieces,               -- M
        'num_points': num_points + 1,           -- N
        'piece_width': piece_width,
        'base_speed': base_speed,
        'travel_dist': travel_dist,
        'end_pos': [p0 p1 ... pM-1],            -- The last position for each piece
        'piece_spacing': [p1-p0 p2-p1 ... ],    -- the final spacing for each piece
        't': [t0 t1 ... tN],                    -- Time Steps
        'p': [[p10 p11 ... p1N],                -- Position steps for each piece
              [p20 p21 ... p2N],
              ...
              [pM0 pM1 ... pMN]],
        'v': [[v10 v11 ... v1N],                -- Velocity steps for each piece
              [v20 v21 ... v2N],
              ...
              [vM0 vM1 ... vMN]],
        'sep': sep_d                            -- A separation motion profile dictionary
    }
    """

    # The total time period for the calculation.
    # Something long enough is chosen.
    travel_time = travel_dist / base_speed  # s

    # Time steps array over the travel time.
    time_steps = np.arange(num_points + 1) * travel_time / num_points  # s

    # Movement array.
    pos_steps = np.arange(num_points + 1) * travel_dist / num_points  # mm

    # Velocity array.
    vel_steps = np.ones(num_points + 1) * base_speed  # mm/s

    # Separation profile number of points, scaled to fit with the separation time.
    # t_s/n_s = T/N => n_s = t_s/T * N
    sep_num_points = int(round(sep_profile.time / travel_time * num_points))

    # Separation motion profile dictionary
    sep_d = sep_profile.calc(sep_num_points)

    # Collect the end position of all pieces in an array to see how much they
    # have separated in the end.
    end_pos = []

    dout = {
        'base_speed': base_speed,
        'num_pieces': num_pieces,
        'num_points': num_points + 1,
        'travel_dist': travel_dist,
        'piece_width': piece_width,
        'end_pos': [],
        'piece_spacing': np.zeros(num_pieces),
        't': time_steps,
        'p': np.zeros((num_pieces, num_points + 1)),
        'v': np.zeros((num_pieces, num_points + 1)),
        'sep_profile': sep_d}

    for i in range(num_pieces, 0, -1):
        # The last piece first, counting down.

        # Start time of separation (cut time)
        t_s = (piece_width * i) / base_speed  # s

        # Index of start time in 'time_steps'.
        i_s = ind(time_steps >= t_s)

        # Scale the separation curve array to the size of pos_steps array.
        # sep_d = [s0 s1 ... sN]
        # sep_pos = [s0 s0 ... s0] [s0 s1 ... sN] [sN sN ... sN]
        sep_pos = np.concatenate((
            sep_d['p'][0] * np.ones(i_s),
            sep_d['p'],
            sep_d['p'][-1] * np.ones(len(time_steps) - sep_num_points - i_s - 1)))

        # Add the separation profile to the movement array.
        pos_steps += sep_pos

        sep_vel = np.concatenate((
            sep_d['v'][0] + np.zeros(i_s),
            sep_d['v'],
            sep_d['v'][-1] + np.zeros(len(time_steps) - sep_num_points - i_s - 1)))
        vel_steps += sep_vel

        # Add the last value (position) in pos_steps to the end_pos array.
        end_pos.append(pos_steps[-1])
        dout['end_pos'].append(pos_steps[-1])

        # Get the resulting spacing
        piece_spacing = end_pos[-1] - end_pos[-2] if len(end_pos) >= 2 else 0

        # if save:
        #     save_datapoints(time_steps, pos_steps, filename + str(i) + '.txt')

        # collect output data
        dout['p'][i - 1] = pos_steps
        dout['v'][i - 1] = vel_steps
        dout['piece_spacing'][i - 1] = piece_spacing

    return dout


def space(num_pieces, piece_width, base_vel, spacing_vel_ratio,
          sep_distance, travel_dist, num_points):
    """Simulate spacing with two const speed conveyors.

    Returns a dictionary:
    {   'base_speed': base_speed,
        'spacing_vel_ratio': spacing_vel_ratio,
        'num_pieces': num_pieces,               -- M
        'num_points': num_points + 1,           -- N
        'sep_distance': sep_distance,           -- Initial seperation between spieces. Cannot be zero.
        'travel_dist': travel_dist,
        'piece_width': piece_width,
        'end_pos': [p0 p1 ... pM-1],            -- The last position for each piece
        'piece_spacing': [p1-p0 p2-p1 ... ],    -- the final spacing for each piece
        'spacing_ratio': spacing_ratio          -- Relative spacing
        't': [t0 t1 ... tN],                    -- Time Steps
        'p': [[p10 p11 ... p1N],                -- Position steps for each piece
              [p20 p21 ... p2N],
              ...
              [pM0 pM1 ... pMN]],
        'v': [[v10 v11 ... v1N],                -- Velocity steps for each piece
              [v20 v21 ... v2N],
              ...
              [vM0 vM1 ... vMN]]
    }
    """
    # The subsequent "speedup/spacing" conveyor speed.
    spacing_vel = spacing_vel_ratio * base_vel  # mm/s

    # The total time period for the calculation.
    # Something long enough is chosen.
    travel_time = travel_dist / base_vel  # s

    # Time steps array over the travel time.
    time_steps = np.arange(num_points + 1) * travel_time / num_points  # s

    # Movement array.
    pos_steps = np.arange(num_points + 1) * travel_dist / num_points  # mm

    # Velocity array.
    vel_steps = np.ones(num_points + 1) * base_vel  # mm/s

    # Collect the end position of all pieces in an array to see how much they
    # have separated in the end.
    end_pos = []

    dout = {
        'base_speed': base_vel,
        'spacing_vel_ratio': spacing_vel_ratio,
        'num_pieces': num_pieces,
        'num_points': num_points + 1,
        'sep_distance': sep_distance,
        'travel_dist': travel_dist,
        'piece_width': piece_width,
        'end_pos': [],
        'piece_spacing': np.zeros(num_pieces),
        't': time_steps,
        'p': np.zeros((num_pieces, num_points + 1)),
        'v': np.zeros((num_pieces, num_points + 1))}

    for i in range(num_pieces, 0, -1):
        # The last piece first, counting down.

        # The position in mm where the spacing for piece nr. i, starts.
        # spacing_start_pos = i * (piece_width + sep_distance) - piece_width / 2  # mm
        spacing_start_pos = piece_width / 2 + (i - 1) * (piece_width + sep_distance)
        spacing_start_step = np.where(pos_steps >= spacing_start_pos)[0][0]

        spacing_pos_steps = spacing_vel * time_steps
        spacing_pos_steps = shift(spacing_pos_steps, spacing_start_step)
        pos_steps = np.concatenate((
            pos_steps[:spacing_start_step],
            pos_steps[spacing_start_step] + spacing_pos_steps[spacing_start_step:]))

        spacing_vel_steps = spacing_vel * np.ones(num_points + 1)
        spacing_vel_steps = shift(spacing_vel_steps, spacing_start_step)
        vel_steps_i = np.concatenate((
            vel_steps[:spacing_start_step], spacing_vel_steps[spacing_start_step:]))

        # Add the last value (position) in pos_steps the end_pos array.
        end_pos.append(pos_steps[-1])
        dout['end_pos'].append(pos_steps[-1])

        # Get the resulting spacing
        piece_spacing = end_pos[-1] - end_pos[-2] + sep_distance if len(end_pos) >= 2 else sep_distance

        # if save:
        #     save_datapoints(time_steps, pos_steps, filename + str(i) + '.txt')

        # collect output data
        dout['p'][i - 1] = pos_steps
        dout['v'][i - 1] = vel_steps_i
        dout['piece_spacing'][i - 1] = piece_spacing

    return dout


def sepspace(num_pieces, piece_width,  base_speed, spacing_vel_ratio,
             sep_conv_length, sep_profile,
             travel_dist, num_points,
             save=False, filename='accelprofile.pkl'):
    """Calculate a seperation/spacing motion sequence.

    Scene:
    Products travel through an infeed conveyor (C1), onto a
    separation conveyor (C2) and thereafter onto a spacing conveyor (C3).
    The products are cut into peices between C1 and C2.
    At the same instant a piece is cut, C2 is accelerated/decelerated with
    respect to it's base speed, to separate it from the remaining product.
    Finally when the pieces enter C3, which is at a higher speed than C2,
    their distance will increase due to the C2-C3 speed difference.

    Returns a dictionary:
    {   'base_speed': base_speed,
        'spacing_vel_ratio': spacing_vel_ratio,
        'num_pieces': num_pieces,               -- M
        'num_points': num_points + 1,           -- N
        'travel_dist': travel_dist,
        'piece_width': piece_width,
        'sep_conv_length': sep_conv_length,
        'end_pos': [p0 p1 ... pM-1],            -- The last position for each piece
        'piece_spacing': [p1-p0 p2-p1 ... ],    -- the final spacing for each piece
        't': [t0 t1 ... tN],                    -- Time Steps
        'p': [[p10 p11 ... p1N],                -- Position steps for each piece
              [p20 p21 ... p2N],
              ...
              [pM0 pM1 ... pMN]],
        'v': [[v10 v11 ... v1N],                -- Velocity steps for each piece
              [v20 v21 ... v2N],
              ...
              [vM0 vM1 ... vMN]],
        's': [[p10-p20 p11-p21 ... p1N-p2N],    -- distance steps between each piece i and piece i+1
              [p20-p30 p21-p31 ... p2N-p3N],
              ...
              [p(M-1)0-pM0 p(M-1)1-pM1 ... p(M-1)N-pMN]],
        'sep': d                                -- The separation motion profile dictionary
    }
    """
    # The first conveyor constant speed and the separation conveyor base speed.
    # base_speed = 180  # mm/s

    # spacing_vel_ratio = 1.7  # times higher than the base_speed

    # The subsequent "speedup/spacing" conveyor speed.
    spacing_speed = spacing_vel_ratio * base_speed  # mm/s

    # print(base_speed, spacing_speed)

    # The length of the separation conveyor.
    # sep_conv_length = 100  # mm

    # Number of pieces to calculate.
    # num_pieces = 20

    # Piece widths to be cut and position to start separating.
    # All pieces have the same width.
    # piece_width = 10  # mm

    # The total distance each piece travels during the calculation.
    # travel_dist = 400  # mm

    # The total time period for the calculation.
    # Something long enough is chosen.
    travel_time = travel_dist / base_speed  # s

    # The total number of time-steps to calculate.
    # num_points = 2500

    # Separation to be created between pieces.
    # sep_length = 10  # mm

    # The separation time.
    # The separation time should be the time it takes a piece to travel
    #
    # sep_length = sep_profile.dist
    sep_time = sep_profile.dist / base_speed  # s

    # Time steps array over the travel time.
    time_steps = np.arange(num_points + 1) * travel_time / num_points  # s

    # Movement array.
    pos_steps = np.arange(num_points + 1) * travel_dist / num_points  # mm

    # Velocity array.
    vel_steps = np.ones(num_points + 1) * base_speed  # mm/s

    # Separation profile number of points.
    sep_num_points = int(sep_time * len(pos_steps) / (np.max(time_steps) + time_steps[1]))

    # Separation motion profile
    # d = scurve(sep_length, sep_time, sep_num_points)
    # d = triangular(sep_length, sep_time, sep_num_points)
    # Given sep length, x and acceleration, a: (t = 2 sqrt(x/a)
    # d = triangular(sep_length, 2*np.sqrt(sep_length/sep_accel), sep_num_points)
    # Given sep length, x and velocity, v: t = 2 x / v
    # d = triangular(sep_length, 2 * sep_length / sep_velocity, sep_num_points)
    # print('Triangular sep: acc: %.3f, sep time: %.3f, v max: %.3f, v avg: %.3f, dist %.3f'\
    #       % (d['A'], d['time'], d['Vmax'], d['Vavg'], d['dist']))
    d = sep_profile.calc(sep_num_points)

    # Collect the end position of all pieces in an array to see how much they
    # have separated in the end.
    end_pos = []

    # if plot:
    #     # f, ax = plt.subplots(1, 2)
    #     # ax[0].set_title('Pos')
    #     # ax[1].set_title('Vel')
    #     f = plt.figure()
    #     gs0 = gridspec.GridSpec(1, 2, figure=f)
    #
    #     gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0])
    #     ax1 = plt.Subplot(f, gs00[0, 0])
    #     f.add_subplot(ax1)
    #
    #     gs01 = gridspec.GridSpecFromSubplotSpec(num_pieces, 1, subplot_spec=gs0[1], hspace=0.0)
    #     ax2 = []
    #     for i in range(num_pieces - 1):
    #         ax2.append(plt.Subplot(f, gs01[i, 0]))
    #         f.add_subplot(ax2[i])
    #         ax2[i].tick_params(labelbottom=False, labelleft=False)
    #     ax2.append(plt.Subplot(f, gs01[i+1, 0]))
    #     f.add_subplot(ax2[i+1])

    #     plt.show()

    dout = {
        'base_speed': base_speed,
        'spacing_vel_ratio': spacing_vel_ratio,
        'num_pieces': num_pieces,
        'num_points': num_points + 1,
        'travel_dist': travel_dist,
        'piece_width': piece_width,
        'sep_conv_length': sep_conv_length,
        'end_pos': [],
        'piece_spacing': np.zeros(num_pieces),
        't': time_steps,
        'p': np.zeros((num_pieces, num_points + 1)),
        'v': np.zeros((num_pieces, num_points + 1)),
        'sep_profile': d}

    for i in range(num_pieces, 0, -1):
        # The last piece first, counting down.

        # Start time.
        t0 = (piece_width * i) / base_speed  # s

        # Index of start time in 'time_steps'.
        i0 = ind(time_steps >= t0)

        # Scale the separation curve array to the size of pos_steps array.
        sep_pos = np.concatenate((
            d['pos'][0] * np.ones(i0),
            d['pos'],
            d['pos'][-1] * np.ones(len(time_steps) - sep_num_points - i0 - 1)))

        # Add the separation curve to the movement profile
        pos_steps += sep_pos

        sep_vel = np.concatenate((
            d['vel'][0] + np.zeros(i0),
            d['vel'],
            d['vel'][-1] + np.zeros(len(time_steps) - sep_num_points - i0 - 1)))
        vel_steps += sep_vel

        # The position in mm where the spacing for piece nr. i, starts.
        spacing_start_pos = i * piece_width + sep_conv_length - piece_width / 2  # mm
        spacing_start_step = np.where(pos_steps >= spacing_start_pos)[0][0]

        spacing_pos_steps = spacing_speed * time_steps
        spacing_pos_steps = shift(spacing_pos_steps, spacing_start_step)
        pos_steps = np.concatenate((
            pos_steps[:spacing_start_step],
            pos_steps[spacing_start_step] + spacing_pos_steps[spacing_start_step:]))

        spacing_vel_steps = spacing_speed * np.ones(num_points + 1)
        spacing_vel_steps = shift(spacing_vel_steps, spacing_start_step)
        vel_steps_i = np.concatenate((
            vel_steps[:spacing_start_step], spacing_vel_steps[spacing_start_step:]))

        # Add the last value (position) in pos_steps the end_pos array.
        end_pos.append(pos_steps[-1])
        dout['end_pos'].append(pos_steps[-1])

        # ax[0].plot(time_steps, pos_steps)
        # ax[1].plot(time_steps, vel_steps)
        # plt.show()

        # Get the resulting spacing
        piece_spacing = end_pos[-1] - end_pos[-2] if len(end_pos) >= 2 else 0

        # if plot:
        #     ax1.plot(time_steps, pos_steps, label=str(i) + ' ' + '%.1f' % piece_spacing)
        #     ax2[i-1].plot(time_steps, vel_steps_i, label=str(i))

        if save:
            save_datapoints(time_steps, pos_steps, filename + str(i) + '.txt')

        # collect output data
        dout['p'][i-1] = pos_steps
        dout['v'][i-1] = vel_steps_i
        dout['piece_spacing'][i-1] = piece_spacing

        # if save:
        #     with open(filename, 'wb') as f:
        #         pickle.dump(dout, f)

    # if plot:
    #     f.suptitle('PieceWidth: ' + str(piece_width) + ' SepConvLength: ' + str(sep_conv_length))
    #     ax1.set_xlabel('Time (s)')
    #     ax1.set_ylabel('Movement (mm)')
    #     ax1.legend()
    #     ax2[-1].set_xlabel('Time (s)')
    #     ax2[-1].set_ylabel('Velocity (mm/s)')
    #     [ax.legend() for ax in ax2]
    #     plt.show()

    return dout


def plot_sepspace(motion_dict):
    """d is a dictionary"""
    f = plt.figure()
    gs0 = gridspec.GridSpec(1, 2, figure=f)

    if not isinstance(motion_dict, tuple):
        motion_dict = (motion_dict,)

    num = []
    [num.append(d['num_pieces']) for d in motion_dict]

    # Left side, position plot
    gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0])
    ax1 = plt.Subplot(f, gs00[0, 0])
    f.add_subplot(ax1)

    # Right side, velocity plots
    gs01 = gridspec.GridSpecFromSubplotSpec(max(num), 1, subplot_spec=gs0[1], hspace=0.0)
    ax2 = []
    for i in range(max(num)):
        ax2.append(plt.Subplot(f, gs01[i, 0]))
        f.add_subplot(ax2[i])
        ax2[i].tick_params(labelbottom=False, labelleft=False)

    for d in motion_dict:
        spacing = shift(d['piece_spacing'], 1, 0)
        for i in range(d['num_pieces']):
            p = ax1.plot(d['t'], d['p'][i], label='%d %.1f' % (i+1, spacing[i]))
            ax2[i].plot(d['t'], d['v'][i], color=p[0].get_color())

    # f.suptitle('PieceWidth: ' + str(d['piece_width']) + ' SepConvLength: ' + str(d['sep_conv_length']))
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Movement (mm)')
    ax1.legend()
    ax2[-1].set_xlabel('Time (s)')
    ax2[-1].set_ylabel('Velocity (mm/s)')
    ax2[-1].tick_params(labelbottom=True, labelleft=True)
    [ax.set_axis_off() for ax in ax2[:-1]]
    plt.show()


class Anim:
    def __init__(self, d, fargs=None, save_count=None, interval=200,
                 repeat=None, repeat_delay=None, blit=None, figsize=(8, 6), figdpi=100):
        self.fargs = fargs
        self.save_count = save_count
        self.interval = interval
        self.repeat_delay = repeat_delay
        self.repeat = repeat
        self.blit = blit
        self.play = True

        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(figsize)
        self.fig.set_dpi(figdpi)
        self.d = d
        self.patches = []
        self.anim = None

        self.step_num = 0
        self.step_min = 0
        self.step_max = d['num_points']
        self.is_running = True
        self.is_forward = True

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)

    def init_func(self):
        pass

    def func(self, p):
        pass

    def update(self):
        for i in range(self.d['num_points']):
            # The input to 'func'
            yield self.d['p'][:, i]

    def run(self):
        self.anim = FuncAnimation(
            self.fig, self.func, frames=self.update, init_func=self.init_func,
            fargs=self.fargs, save_count=self.save_count,
            interval=self.interval, repeat=self.repeat,
            repeat_delay=self.repeat_delay, blit=self.blit)
        plt.show()

    def on_key_press(self, event):
        if event.key == '5':
            # print('pressed', event.key)
            if self.play:
                self.play ^= True
                # print('pause', self.play)
                self.anim.event_source.stop()
            else:
                self.play ^= True
                # print('play', self.play)
                self.anim.event_source.start()

        elif event.key == '6':
            # print('pressed', event.key)
            if not self.play:
                # is paused
                # print('step')
                self.anim._step()

    def on_mouse_press(self, event):
        # print('mouse pressed', event.button)
        if event.button == 1:
            # print('pressed', event.button)
            if self.play:
                self.play ^= True
                # print('pause', self.play)
                self.anim.event_source.stop()
            else:
                self.play ^= True
                # print('play', self.play)
                self.anim.event_source.start()

        elif event.button == 3:
            # print('pressed', event.button)
            if not self.play:
                # is paused
                # print('step')
                self.anim._step()


class AnimSep(Anim):
    def __init__(self, d, fargs=None, save_count=None, interval=200,
                 repeat=None, repeat_delay=None, blit=None, figsize=(8, 6), figdpi=100):
        Anim.__init__(self, d, fargs, save_count, interval, repeat, repeat_delay, blit, figsize, figdpi)

    def init_func(self):
        r = 6  # mm, conveyor nose diameter
        self.ax.set_xlim(-self.d['num_pieces'] * self.d['piece_width'], self.d['travel_dist'])
        self.ax.set_ylim(-r, 2 * self.d['piece_width'])
        self.ax.set_aspect('equal')
        start, end = self.ax.get_xlim()
        self.ax.xaxis.set_ticks(np.arange(start, end, 5))
        self.ax.grid(True, axis='x')
        self.ax.set_yticks([])
        return []

    def func(self, p):
        # draw the peices
        # p is a list of positions for each piece at time t
        w = self.d['piece_width']
        self.patches = []
        for i in range(self.d['num_pieces']):
            self.patches.append(self.ax.add_patch(
                plt.Rectangle((-i * (w) - w + p[i], 0), w, w,
                              animated=True, fill=True, linewidth=1.0)))
        return self.patches


class AnimSpace(Anim):
    def __init__(self, d, fargs=None, save_count=None, interval=200,
                 repeat=None, repeat_delay=None, blit=None, figsize=(8, 6), figdpi=100):
        Anim.__init__(self, d, fargs, save_count, interval, repeat, repeat_delay, blit, figsize, figdpi)

    def init_func(self):
        r = 6  # mm, conveyor nose diameter
        self.ax.set_xlim(-self.d['num_pieces'] * self.d['piece_width'], self.d['travel_dist'])
        self.ax.set_ylim(-r, 2 * self.d['piece_width'])
        self.ax.set_aspect('equal')
        start, end = self.ax.get_xlim()
        self.ax.xaxis.set_ticks(np.arange(start, end, 5))
        self.ax.grid(True, axis='x')
        self.ax.set_yticks([])
        return []

    def func(self, p):
        # draw the peices
        # p is a list of positions for each piece at time t
        s = self.d['sep_distance']
        w = self.d['piece_width']
        self.patches = []
        for i in range(self.d['num_pieces']):
            self.patches.append(self.ax.add_patch(
                plt.Rectangle((-i * (w + s) - w + p[i], 0), w, w,
                              animated=True, fill=True, linewidth=1.0)))
        return self.patches


class AnimSepSpace(Anim):
    def __init__(self, d, fargs=None, save_count=None, interval=200,
                 repeat=None, repeat_delay=None, blit=None, figsize=(8, 6), figdpi=100):
        Anim.__init__(self, d, fargs, save_count, interval, repeat, repeat_delay, blit, figsize, figdpi)

    def init_func(self):
        r = 6  # mm, conveyor nose diameter
        self.ax.set_xlim(-self.d['num_pieces'] * self.d['piece_width'], self.d['travel_dist'])
        self.ax.set_ylim(-r, 2 * self.d['piece_width'])
        self.ax.set_aspect('equal')
        start, end = self.ax.get_xlim()
        self.ax.xaxis.set_ticks(np.arange(start, end, 5))

        # Infeed conveyor
        self.ax.add_patch(pch.Rectangle(
            (-(self.d['num_pieces'] + 1) * self.d['piece_width'], -r), (self.d['num_pieces'] + 1) * self.d['piece_width'], r,
            fill=False, linewidth=0.5))

        # Sep conveyor
        self.ax.add_patch(pch.Rectangle(
            (0, -r), self.d['sep_conv_length'], r,
            fill=False, linewidth=0.5))

        # Turn infeed conveyor
        self.ax.add_patch(pch.Rectangle(
            (self.d['sep_conv_length'], -r), 1000, r,
            fill=False, linewidth=0.5))

        # self.ax.set_axis_off()
        return []

    def func(self, p):
        # draw the peices
        w = self.d['piece_width']
        self.patches = []
        for i in range(self.d['num_pieces']):
            self.patches.append(self.ax.add_patch(
                plt.Rectangle((-i * w - w + p[i], 0), w, w,
                              animated=True, fill=False, linewidth=0.5)))
        return self.patches


class AnimSepSpaceOld:
    def __init__(self, d, fargs=None, save_count=None, interval=200,
                 repeat=None, repeat_delay=None, blit=None, figsize=(8, 6), figdpi=100):
        self.fargs = fargs
        self.save_count = save_count
        self.interval = interval
        self.repeat_delay = repeat_delay
        self.repeat = repeat
        self.blit = blit
        self.play = True

        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(figsize)
        self.fig.set_dpi(figdpi)
        self.d = d
        self.patches = []
        self.anim = None

        self.step_num = 0
        self.step_min = 0
        self.step_max = d['num_points']
        self.is_running = True
        self.is_forward = True

        # button_pos = (0.125, 0.92)
        # playerax = self.fig.add_axes([pos[0], pos[1], 0.22, 0.04])
        # divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        # bax = divider.append_axes("right", size="80%", pad=0.05)
        # sax = divider.append_axes("right", size="80%", pad=0.05)
        # fax = divider.append_axes("right", size="80%", pad=0.05)
        # ofax = divider.append_axes("right", size="100%", pad=0.05)
        #
        # self.button_oneback = matplotlib.widgets.Button(playerax, label=u'$\u29CF$')
        # self.button_back = matplotlib.widgets.Button(bax, label=u'$\u25C0$')
        # self.button_stop = matplotlib.widgets.Button(sax, label=u'$\u25A0$')
        # self.button_forward = matplotlib.widgets.Button(fax, label=u'$\u25B6$')
        # self.button_oneforward = matplotlib.widgets.Button(ofax, label=u'$\u29D0$')
        #
        # self.button_oneback.on_clicked(self.step_back)
        # self.button_back.on_clicked(self.back)
        # self.button_stop.on_clicked(self.stop)
        # self.button_forward.on_clicked(self.forward)
        # self.button_oneforward.on_clicked(self.step_forward)

        # play_ax = self.fig.add_axes([button_pos[0], button_pos[1], 0.16, 0.04])
        # divider = mpl_toolkits.axes_grid1.make_axes_locatable(play_ax)
        # pause_ax = divider.append_axes("right", size="100%", pad=0.05)
        # self.button_play = matplotlib.widgets.Button(play_ax, label=u'$\u25B6$')
        # self.button_pause = matplotlib.widgets.Button(pause_ax, label=u'$\u258B\u258B$')
        # self.button_play.on_clicked(self.on_play)
        # self.button_pause.on_clicked(self.on_pause)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)

    def init_func(self):
        r = 6  # mm, conveyor nose diameter
        self.ax.set_xlim(-self.d['num_pieces']*self.d['piece_width'], self.d['travel_dist'])
        self.ax.set_ylim(-r, 2 * self.d['piece_width'])
        self.ax.set_aspect('equal')
        start, end = self.ax.get_xlim()
        self.ax.xaxis.set_ticks(np.arange(start, end, 5))

        # Infeed conveyor
        self.ax.add_patch(pch.Rectangle(
            (-(self.d['num_pieces']+1)*self.d['piece_width'], -r), (self.d['num_pieces']+1)*d['piece_width'], r,
            fill=False, linewidth=0.5))

        # Sep conveyor
        self.ax.add_patch(pch.Rectangle(
            (0, -r), self.d['sep_conv_length'], r,
            fill=False, linewidth=0.5))

        # Turn infeed conveyor
        self.ax.add_patch(pch.Rectangle(
            (self.d['sep_conv_length'], -r), 1000, r,
            fill=False, linewidth=0.5))

        # self.ax.set_axis_off()
        return []

    def func(self, p):
        # draw the peices
        w = self.d['piece_width']
        self.patches = []
        for i in range(self.d['num_pieces']):
            self.patches.append(self.ax.add_patch(
                plt.Rectangle((-i*w - w + p[i], 0), w, w,
                              animated=True, fill=False, linewidth=0.5)))
        return self.patches

    def update(self):
        for i in range(self.d['num_points']):
            # The input to 'func'
            yield self.d['p'][:, i]
        # while self.is_running:
        #     self.step_num += self.is_forward - (not self.is_forward)
        #     if self.step_min < self.step_num < self.step_max:
        #         yield d['p'][:, self.step_num]
        #     else:
        #         self.stop()
        #         yield d['p'][:, self.step_num]

    def run(self):
        self.anim = FuncAnimation(
            self.fig, self.func, frames=self.update, init_func=self.init_func,
            fargs=self.fargs, save_count=self.save_count,
            interval=self.interval, repeat=self.repeat,
            repeat_delay=self.repeat_delay, blit=self.blit)
        plt.show()

    def on_key_press(self, event):
        if event.key == '5':
            # print('pressed', event.key)
            if self.play:
                self.play ^= True
                # print('pause', self.play)
                self.anim.event_source.stop()
            else:
                self.play ^= True
                # print('play', self.play)
                self.anim.event_source.start()

        elif event.key == '6':
            # print('pressed', event.key)
            if not self.play:
                # is paused
                # print('step')
                self.anim._step()

    def on_mouse_press(self, event):
        # print('mouse pressed', event.button)
        if event.button == 1:
            # print('pressed', event.button)
            if self.play:
                self.play ^= True
                # print('pause', self.play)
                self.anim.event_source.stop()
            else:
                self.play ^= True
                # print('play', self.play)
                self.anim.event_source.start()

        elif event.button == 3:
            # print('pressed', event.button)
            if not self.play:
                # is paused
                # print('step')
                self.anim._step()

    # def on_play(self, event):
    #     self.anim.event_source.start()
    #
    # def on_pause(self, event):
    #     self.anim.event_source.stop()

    # def start(self, event):
    #     self.is_running = True
    #     self.anim.event_source.start()
    #
    # def stop(self, event):
    #     self.is_running = False
    #     self.anim.event_source.stop()
    #
    # def forward(self, event):
    #     self.is_forward = True
    #     self.start()
    #
    # def back(self, event):
    #     self.is_forward = False
    #     self.start()
    #
    # def step_forward(self, event):
    #     self.is_forward = True
    #     self.step()
    #
    # def step_back(self, event):
    #     self.is_forward = False
    #     self.step()
    #
    # def step(self):
    #     if self.step_min < self.step_num < self.step_max:
    #         self.step_num += self.is_forward - (not self.is_forward)
    #
    #     elif self.step_num == self.step_min and self.is_forward:
    #         self.step_num += 1
    #
    #     elif self.step_num == self.step_max and (not self.is_forward):
    #         self.step_num -= 1
    #
    #     self.func(self.d['p'][:, self.step_num])
    #     self.fig.canvas.draw_idle()


def accel_profile(save=False, plot=True, filename='accelprofile'):
    """Calculate a seperation/spacing motion sequence.

    Scene:
    Products travel through an infeed conveyor (C1), onto a
    separation conveyor (C2) and thereafter onto a spacing conveyor (C3).
    The products are cut into peices between C1 and C2.

    Product is cut into pieces. Each piece is separated after the cut.
    Thereafter, further spacing is acquired after the separation conveyor with
    a speed difference between the separation conveeyor and
    the spacing (speedup) conveyor.
    """
    # The first conveyor constant speed and the separation conveyor base speed.
    base_speed = 180  # mm/s

    # The subsequent "speedup/spacing" conveyor speed.
    spacing_speed = 1.70 * base_speed  # mm/s

    print(base_speed, spacing_speed)

    # The length of the separation conveyor.
    sep_conv_length = 100  # mm

    # Number of pieces to calculate.
    num_pieces = 6

    # Piece widths to be cut and position to start separating.
    # All pieces have the same width.
    piece_width = 30  # mm

    # The total distance each piece travels during the calculation.
    travel_dist = 250  # mm

    # The total time period for the calculation.
    # Something long enough is chosen.
    travel_time = travel_dist / base_speed  # s

    # The total number of time-steps to calculate.
    num_points = 2500

    # ???
    # Distance, the piece behind, can travel during spacing.
    # spacing_length = 10  # mm

    # Separation to be created between pieces.
    sep_length = 10  # mm

    # The separation time.
    sep_time = sep_length / base_speed  # s

    # Time steps array over the travel time.
    time_steps = np.arange(num_points) * travel_time / num_points  # s

    # Movement array.
    pos_steps = base_speed * time_steps  # mm

    # Speed array.
    vel_steps = np.ones(num_points) * base_speed  # mm/s

    # Separation profile number of points.
    sep_num_points = int(sep_time * len(pos_steps) / (
            np.max(time_steps) + time_steps[1]))

    # Separation motion profile
    # d = scurve(sep_length, sep_time, sep_num_points)
    d = triangular(sep_length, sep_time, sep_num_points)

    # Collect the end position of all pieces in an array to see how much they
    # have separated in the end.
    end_pos = []

    if plot:
        f, ax = plt.subplots(1, 2)
        ax[0].set_title('Pos')
        ax[1].set_title('Vel')

    for i in range(num_pieces, 0, -1):
        # The last piece first, counting down.
        # j = num_pieces - i

        # Start time.
        t0 = (piece_width * i) / base_speed  # s

        # Index of start time in 'time_steps'.
        i0 = ind(time_steps >= t0)

        #
        # Create the position profile for piece no. j
        #

        # Scale the separation curve array to the Pos array.
        sep_pos = np.concatenate((
            d['pos'][0] * np.ones(i0),
            d['pos'],
            d['pos'][-1] * np.ones(len(time_steps) - sep_num_points - i0)))

        # Add the separation curve to the movement profile
        pos_steps = sep_pos + pos_steps

        # The spacing takes place at the piece width center of cravity.
        spacing_start_pos = i * piece_width + sep_conv_length - piece_width / 2

        # Position steps during the spacing.
        spacing_pos_steps = spacing_speed * time_steps[:len(np.where(
            pos_steps >= spacing_start_pos)[0])]

        # Position steps before the spacing should take place.
        pos_steps_before_spacing = pos_steps[np.where(
            pos_steps <= spacing_start_pos)]

        # Add the spacing to the movement profile
        pos_steps = np.concatenate((pos_steps_before_spacing,
                                    pos_steps_before_spacing[-1] + spacing_pos_steps))

        # Add the last value (position) in pos_steps the end_pos array.
        end_pos.append(pos_steps[-1])

        # Get the resulting spacing
        piece_spacing = end_pos[-1] - end_pos[-2] if len(end_pos) >= 2 else 0

        if plot:
            ax[0].plot(time_steps, pos_steps, label=str(i) + ' ' + '%.1f' % piece_spacing)

        if save:
            save_datapoints(time_steps, pos_steps, filename + str(i) + '.txt')

        #
        # Create the speed profile for piece no. i
        # TODO
        if plot:
            ax[1].plot(time_steps, vel_steps)

    if plot:
        ax[0].set_xlabel('Time (s)')
        ax[0].legend()
        plt.show()


def map_spacing():
    """Map spacing with respect to piece width and speedup velocity ratio"""
    w = np.arange(5, 51)
    r_v = np.arange(1.1, 2.1, 0.1)
    print(len(w), len(r_v))
    r_s = np.empty([len(w), len(r_v)])
    i = 0
    for wi in w:
        print(i, wi)
        j = 0
        for r_vj in r_v:
            # print(i, j)
            d = space(num_pieces=2, piece_width=wi, base_vel=1, spacing_vel_ratio=r_vj,
                      sep_distance=10, travel_dist=200, num_points=1000)
            r_s[i, j] = max(d['piece_spacing']) / d['sep_distance']
            j += 1
        i += 1

    m = {'sep_distance': d['sep_distance'], 'w': w, 'r_v': r_v, 'r_s': r_s}

    with open('spacemap5.pkl', 'wb') as f:
        pickle.dump(m, f)

    with open('spacemap5.pkl', 'rb') as f:
        m = pickle.load(f)

    w = m['w']
    r_v = m['r_v']
    print(len(w), len(r_v))
    wg, r_vg = np.meshgrid(r_v, w)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(w, r_v, d['r_s'], cmap=cm.coolwarm)
    # plt.show()

    # plt.plot(w, d['r_s'][:, 3], label='%.2f' % r_v[3])
    for i in range(len(r_v)):
        # Equation of a line:
        # y = a(x - x1) + y1
        # Incline a = (y2-y1)/(x2-x1)
        a = (m['r_s'][-1, i] - m['r_s'][0, i]) / (w[-1] - w[0])
        plt.plot(w, m['r_s'][:, i], label='%.2f, %.2f' % (r_v[i], a))

    plt.title(str(m['sep_distance']))
    plt.xlabel('Piece width (mm)')
    plt.ylabel('Relative spacing')
    plt.legend()
    plt.show()


def simulate_spacing():
    d = space(num_pieces=5, piece_width=20, base_vel=143, spacing_vel_ratio=1.4,
              sep_distance=10, travel_dist=200, num_points=1000)
    # print(d['end_pos'])
    # print(d['piece_spacing'])
    # print(d['sep_distance'], max(d4['piece_spacing']))
    print('w=%.2f, r_v=%.2f, r_s=%.2f' %
          (d['piece_width'], d['spacing_vel_ratio'], max(d['piece_spacing']) / d['sep_distance']))

    anim = AnimSpace(d, interval=0, blit=True, figsize=(20, 10), figdpi=72)
    anim.run()


if __name__ == '__main__':
    # d = sepspace(num_pieces=2, piece_width=15,
    #              sep_conv_length=100, sep_profile=Triangular(dist=10, accel=10000),
    #              base_speed=143, spacing_vel_ratio=1.7,
    #              travel_dist=200, num_points=2000)
    #
    # d2 = sepspace(num_pieces=2, piece_width=30,
    #               sep_conv_length=100, sep_profile=Triangular(dist=10, accel=10000),
    #               base_speed=143, spacing_vel_ratio=1.7,
    #               travel_dist=200, num_points=2000)

    # d3 = sepspace(num_pieces=3, piece_width=30,
    #               sep_conv_length=100, sep_profile=Triangular(dist=10, accel=1000),
    #               base_speed=143, spacing_vel_ratio=1.7,
    #               travel_dist=200, num_points=2000)

    # d4 = space(num_pieces=5, piece_width=20, base_vel=143, spacing_vel_ratio=1.4,
    #            sep_distance=10, travel_dist=200, num_points=1000)
    # # print(d4['end_pos'])
    # # print(d4['piece_spacing'])
    # # print(d4['sep_distance'], max(d4['piece_spacing']))
    # print('w=%.2f, r_v=%.2f, r_s=%.2f' %
    #       (d4['piece_width'], d4['spacing_vel_ratio'], max(d4['piece_spacing'])/d4['sep_distance']))

    # d5 = space(num_pieces=2, piece_width=20, base_vel=148, spacing_vel_ratio=1.5,
    #            sep_distance=10, travel_dist=100, num_points=500)

    # with open('sepspace.pkl', 'wb') as f:
    #     pickle.dump(d, f)

    # with open('sepspace.pkl', 'rb') as f:
    #     d = pickle.load(f)

    # print(d['p'])
    # print(d['p'][:, 100])

    # print('Vbase=%.2f, Vsepmax=%.2f, Vsepavg=%.2f, Vspacing=%.2f' %
    #       (d['base_speed'],
    #        d['sep_profile']['Vmax'] + d['base_speed'],
    #        d['sep_profile']['Vavg'] + d['base_speed'],
    #        d['base_speed'] * d['spacing_vel_ratio']))
    # print(d['piece_spacing'])

    # plot_motion_profile(d['sep_profile'])

    # plot_sepspace((d))

    # anim = AnimSepSpace(d, interval=0, blit=True, figsize=(20, 10), figdpi=72)
    # anim.run()

    # anim = AnimSpace(d4, interval=0, blit=True, figsize=(20, 10), figdpi=72)
    # anim.run()

    # d = triangular(10, 4)
    # print(d['t'])
    # plot_motion_profile(d)
    # plot_motion_profile(scurve(10, 4))

    # accel_profile(save=True, plot=False, filename='accelprofile2')
    # accel_profile(save=False, plot=True)
    # plt.show()

    # map_spacing()

    # simulate_spacing()

    #
    # Separation with different acceleration
    #
    # d = []
    # for i in [2000, 10000.0, 20000]:
    #     d.append(sep(num_pieces=2, base_speed=143, piece_width=20, sep_profile=Triangular(10, accel=i),
    #              travel_dist=100, num_points=1000))
    # plot_sepspace(tuple(d))


    # d = []
    # d.append(sep(num_pieces=2, base_speed=143, piece_width=20,
    #              sep_profile=Triangular(10, accel=10000), travel_dist=100, num_points=1000))
    # d.append(sep(num_pieces=2, base_speed=143, piece_width=20,
    #              sep_profile=Trapezoidal(10, accel=10000, v_max=300-143), travel_dist=100, num_points=1000))
    # plot_sepspace(tuple(d))

    d = sep(num_pieces=2, base_speed=143, piece_width=20,
            sep_profile=Trapezoidal(10, accel=10000, v_max=300-143), travel_dist=100, num_points=1000)
    anim = AnimSep(d, interval=0, blit=True, figsize=(20, 10), figdpi=72)
    anim.run()
