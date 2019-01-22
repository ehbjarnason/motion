from calcspacing import *


def sep(num_pieces, piece_width, base_speed, sep_profiles, travel_dist, num_points):

    # The total time period for the calculation.
    # Something long enough is chosen. The actual time is unknown.
    travel_time = travel_dist / base_speed  # s

    time_steps = np.arange(num_points + 1) * travel_time / num_points  # s
    # pos_steps = np.arange(num_points + 1) * travel_dist / num_points  # mm
    vel_steps = np.ones(num_points + 1) * base_speed  # mm/s

    # print(time_steps, pos_steps)

    # Initialize separation motion profile dictionary array
    sep_d = [None, None]
    sep_num_points = [0, 0]
    if sep_profiles[0] is not None:
        sep_num_points[0] = int(round(sep_profiles[0].time / travel_time * num_points))
        sep_d[0] = sep_profiles[0].calc(sep_num_points[0])

    if sep_profiles[1] is not None:
        # Separation profile number of points, scaled to fit with the separation time.
        # t_s/n_s = T/N => n_s = t_s/T * N
        sep_num_points[1] = int(round(sep_profiles[1].time / travel_time * num_points))
        sep_d[1] = sep_profiles[1].calc(sep_num_points[1])

    # plot_motion_profile(tuple(sep_d))
    # print(travel_time, sep_profiles[0].time, sep_profiles[1].time)

    # Initialize the return dictionary
    d = {
        'base_speed': base_speed,
        'num_pieces': num_pieces,
        'num_points': num_points + 1,
        'travel_dist': travel_dist,
        'piece_width': piece_width,
        't': time_steps,
        'p': np.zeros((num_pieces, num_points + 1)),
        'v': np.zeros((num_pieces, num_points + 1)),
        'sep_profile': sep_d,
        'end_pos': [],
        'piece_spacing': np.zeros(num_pieces)}

    # Initialize arrays
    d['t'] = time_steps
    for i in range(num_pieces):
        # Initially piece nr. i is travelling at speed 'base_speed'.
        # And travels for a time 'travel_time' if there is no disturbance.
        # It starts at position -i*w from the cut point (the zero point).
        d['p'][i] = base_speed * time_steps - piece_width * (i + 1)
    d['v'] += vel_steps

    # print(d['p'])

    sep_pos = np.zeros((2, num_points + 1))
    sep_vel = np.zeros((2, num_points + 1))
    for i in range(num_pieces - 1):
        # Do nothing with the last piece.
        i_zero = ind(d['p'][i] >= 0)
        if sep_profiles[0] is not None:
            sep_pos[0] = replace_range(np.zeros(num_points + 1), sep_d[0]['p'], i_zero)
            sep_vel[0] = replace_range(np.zeros(num_points + 1), sep_d[0]['v'], i_zero)
        if sep_profiles[1] is not None:
            sep_pos[1] = replace_range(np.zeros(num_points + 1), sep_d[1]['p'], i_zero)
            sep_vel[1] = replace_range(np.zeros(num_points + 1), sep_d[1]['v'], i_zero)

        for j in range(num_pieces):
            if j <= i:
                # Speedup
                d['p'][j] += sep_pos[1]
                d['v'][j] += sep_vel[1]
            else:
                # Slowdown
                d['p'][j] -= sep_pos[0]
                d['v'][j] -= sep_vel[0]

    # # The first piece, when it is in pos zero:
    # # Add speedup to piece1
    # # Add slowdown to pieces 2 and 3. ... the ramaining pieces
    # i_cut = ind(d['p'][0] >= 0)
    # speedup_pos = replace_range(np.zeros(num_points + 1), sep_d[1]['p'], i_cut)
    # slowdown_pos = replace_range(np.zeros(num_points + 1), sep_d[0]['p'], i_cut)
    # speedup_vel = replace_range(np.zeros(num_points + 1), sep_d[1]['v'], i_cut)
    # slowdown_vel = replace_range(np.zeros(num_points + 1), sep_d[0]['v'], i_cut)
    # d['p'][0] += speedup_pos
    # d['p'][1] -= slowdown_pos
    # d['p'][2] -= slowdown_pos
    # d['v'][0] += speedup_vel
    # d['v'][1] -= slowdown_vel
    # d['v'][2] -= slowdown_vel
    #
    # # The second piece
    # i_cut = ind(d['p'][1] >= 0)
    # speedup_pos = replace_range(np.zeros(num_points + 1), sep_d[1]['p'], i_cut)
    # slowdown_pos = replace_range(np.zeros(num_points + 1), sep_d[0]['p'], i_cut)
    # speedup_vel = replace_range(np.zeros(num_points + 1), sep_d[1]['v'], i_cut)
    # slowdown_vel = replace_range(np.zeros(num_points + 1), sep_d[0]['v'], i_cut)
    # d['p'][0] += speedup_pos
    # d['p'][1] += speedup_pos
    # d['p'][2] -= slowdown_pos
    # d['v'][0] += speedup_vel
    # d['v'][1] += speedup_vel
    # d['v'][2] -= slowdown_vel
    #
    # # The third piece
    # # Do nothing

    return d


def sep_old(num_pieces, piece_width,  base_speed, sep_profiles, travel_time, travel_dist, num_points):
    """Simulate separation after cutting.

    Sep_profile: A tuple of two motion profile objects: (Infeed slowdown, outfeed speedup).

    Returns a dictionary
    """

    # The total time period for the calculation.
    # Something long enough is chosen. The actual time is unknown.
    # travel_time = travel_dist / base_speed  # s

    # Time steps array over the travel time.
    time_steps = np.arange(num_points + 1) * travel_time / num_points  # s
    print('detla t: ', time_steps[1])

    # Movement array.
    pos_steps = np.arange(num_points + 1) * travel_dist / num_points  # mm, [0 ... dist]
    # The separation takes place at the zero position.
    # pos_steps -= (num_pieces + 1) * piece_width  # [-(n+1)w ... 0 ... (dist - (n+1)w)]

    # Velocity array.
    vel_steps = np.ones(num_points + 1) * base_speed  # mm/s

    # Separation motion profile dictionary array
    sep_d = [None, None]
    sep_num_points = 0
    sep_num_points_in = 0

    if sep_profiles[0] is not None:
        sep_num_points_in = int(round(sep_profiles[0].time / travel_time * num_points))
        sep_d[0] = sep_profiles[0].calc(sep_num_points_in)
        # test

    if sep_profiles[1] is not None:
        # Separation profile number of points, scaled to fit with the separation time.
        # t_s/n_s = T/N => n_s = t_s/T * N
        sep_num_points = int(round(sep_profiles[1].time / travel_time * num_points))
        sep_d[1] = sep_profiles[1].calc(sep_num_points)

    # Collect the end position of all pieces in an array to see how much they
    # have separated in the end.
    end_pos = []

    dout = {
        'base_speed': base_speed,
        'num_pieces': num_pieces,
        'num_points': num_points + 1,
        'travel_dist': travel_dist,
        'piece_width': piece_width,
        'delta_t': travel_time / num_points,
        'end_pos': [],
        'piece_spacing': np.zeros(num_pieces),
        't': time_steps,
        'p': np.zeros((num_pieces, num_points + 1)),
        'v': np.zeros((num_pieces, num_points + 1)),
        'sep_profile': sep_d}

    # initialize position and velocity arrays.
    for i in range(num_pieces):
        dout['p'][i] = pos_steps
        dout['v'][i] = vel_steps

    for i in range(num_pieces, 0, -1):
        # The last piece first, counting down.
        print('piece ', i)

        if sep_d[0] is not None and i != 1:
            # Infeed slowdown
            # The first piece is never slowed down.
            # Slow down the piece behind, nr. i+1

            t_s_in = (piece_width * (i - 1)) / base_speed  # s

            i_s_in = ind(time_steps >= t_s_in)

            sep_pos_in = np.concatenate((
                sep_d[0]['p'][0] * np.ones(i_s_in),
                sep_d[0]['p'] * -1,
                sep_d[0]['p'][-1] * -1 * np.ones(len(time_steps) - sep_num_points_in - i_s_in - 1)))

            print(i)

            dout['p'][i - 1] += sep_pos_in

        if sep_d[1] is not None and i != num_pieces:
            # Outfeed speedup
            # The last piece is never speeded up.

            # Start time of separation (cut time)
            # When the piece has traveled distance piece_width * i.
            t_s = (piece_width * i) / base_speed  # s

            # Index of start time in 'time_steps'.
            i_s = ind(time_steps >= t_s)

            # Scale the separation curve array to the size of pos_steps array.
            # sep_d = [s0 s1 ... sN]
            # sep_pos = [s0 s0 ... s0] [s0 s1 ... sN] [sN sN ... sN]
            sep_pos = np.concatenate((
                sep_d[1]['p'][0] * np.ones(i_s),
                sep_d[1]['p'],
                sep_d[1]['p'][-1] * np.ones(len(time_steps) - sep_num_points - i_s - 1)))

            # Add the separation profile to the movement array.
            pos_steps += sep_pos

            sep_vel = np.concatenate((
                sep_d[1]['v'][0] + np.zeros(i_s),
                sep_d[1]['v'],
                sep_d[1]['v'][-1] + np.zeros(len(time_steps) - sep_num_points - i_s - 1)))
            vel_steps += sep_vel

        # Add the last value (position) in pos_steps to the end_pos array.
        end_pos.append(pos_steps[-1])
        dout['end_pos'].append(pos_steps[-1])

        # Get the resulting spacing
        piece_spacing = end_pos[-1] - end_pos[-2] if len(end_pos) >= 2 else 0

        # if save:
        #     save_datapoints(time_steps, pos_steps, filename + str(i) + '.txt')

        # Relocate the piece
        # i_p = ind(pos_steps >= -i * piece_width)
        # pos_steps_i = shift(pos_steps, i_p)

        # collect output data

        dout['p'][i - 1] = pos_steps
        dout['v'][i - 1] = vel_steps
        dout['piece_spacing'][i - 1] = piece_spacing

    return dout


def simulate_sep(base_speed, sep_dist, sep_accel):
    sep_dist = 5  # mm
    sep_accel = 10000  # mm/s2
    base_speed = np.arange(143, 200)

    for v in base_speed:
        d = sep(num_pieces=2, piece_width=20, base_speed=v,
                sep_profiles=(Trapezoidal(sep_dist, v_max=v, None)),
                travel_dist=100, num_points=2000)

    return sep_time


if __name__ == '__main__':
    # d = sep(num_pieces=6, base_speed=143, piece_width=20,
    #         sep_profiles=(Triangular(10, accel=10000), Triangular(10, accel=10000)),
    #         travel_dist=200, num_points=1000)

    # d = sep(num_pieces=6, base_speed=143, piece_width=20,
    #         sep_profiles=(None, Triangular(10, accel=10000)),
    #         travel_dist=200, num_points=1000)

    # d = sep(num_pieces=6, base_speed=143, piece_width=20,
    #         sep_profiles=(Triangular(5, accel=10000), None),
    #         travel_dist=200, num_points=1000)

    # 50% infeed contribution
    #
    # d = sep(num_pieces=3, base_speed=143, piece_width=20,
    #         sep_profiles=(Trapezoidal(5, accel=10000, v_max=143), Trapezoidal(5, accel=10000, v_max=500-143)),
    #         travel_dist=100, num_points=3000)

    # 0% infeed contribution
    #
    # d = sep(num_pieces=3, base_speed=143, piece_width=20,
    #         sep_profiles=(None, Trapezoidal(10, accel=10000, v_max=500-143)),
    #         travel_dist=100, num_points=1000)

    # 100% infeed contribution
    #
    d = sep(num_pieces=2, base_speed=143, piece_width=20,
            sep_profiles=(Trapezoidal(10, accel=10000, v_max=143), None),
            travel_dist=100, num_points=2000)

    # No separation
    #
    # d = sep(num_pieces=3, base_speed=143, piece_width=20,
    #         sep_profiles=(None, None),
    #         travel_dist=100, num_points=2000)

    # plot_sepspace(d)

    anim = AnimSep(d, interval=0, blit=True, figsize=(20, 10), figdpi=72)
    anim.run()
