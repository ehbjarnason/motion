from motion import *


def turner_profiles_pos(save=False, plot=False):
    """Creates all motion profiles based on constant conveyor speed."""

    # The first specified time values assumed max capacity of 8 turns per
    # second at a constant conveyor speed of 481mm/s.
    # Based on calculation in "capacityestimation.xlsm".
    # conv_speed = 481 # mm/s
    conv_speed = 300  # mm/s
    # speed = position / time
    # position = speed * time

    # A one cycle by definition, includes two turns (to each side).
    # So eight turns per second are four cycles per second and one cycle
    # will take a quarter of a second.
    # turnes_per_sec = 8
    turnes_per_sec = 5
    cycle_time = 2 / turnes_per_sec  # sec

    print('Conv. speed: {} mm/s'.format(conv_speed))
    print('Turnes per second: {} '.format(turnes_per_sec))
    print('Cycle time: {} s'.format(cycle_time))
    print('')

    # time = 0.25 # sek, Total cycle time. Four cycles per second
    num_points = 2500  # num points
    range_vert = 65  # mm, vertical movement range
    range_ang = 90  # deg, rotational range

    # Intervals as a ratio with respect to cycle time/displacement
    r1 = 0.16  # Move down
    r2 = 0.44  # Rotate forward
    r3 = 0.24  # Move up
    r4 = 0.40  # Rotate back
    r5 = -0.0732  # Overlap between start r2 and stop r1
    r6 = 0.0  # Overlap/Delay between start r3 and stop r2
    r7 = -0.172  # Overlap between start r4 ans stop r3

    print('Interval ratios:')
    print('Move down: r1={:.2f}%'.format(r1 * 100))
    print('Rotate forward, r2={:.2f}%'.format(r2 * 100))
    print('Move up, r3={:.2f}%'.format(r3 * 100))
    print('Rotate back, r4={:.2f}%'.format(r4 * 100))
    print('Overlap between start r2 and stop r1, r5={:.2f}%'.format(r5 * 100))
    print('Overlap between start r3 and stop r2, r6={:.2f}%'.format(r6 * 100))
    print('Overlap between start r4 and stop r3, r7={:.2f}%'.format(r7 * 100))
    print('Sum of interval ratios: r1+r2+r3+r4+r5+r6+r7={:.2f}%'.format(
        (r1 + r2 + r3 + r4 + r5 + r6 + r7) * 100))
    print('')

    # Duration time for each movement
    # t1 = 0.04 # move down
    # t2 = 0.11 # rotate forward
    # t3 = 0.06 # move up
    # t4 = 0.1 # rotate back
    t1 = r1 * cycle_time
    t2 = r2 * cycle_time
    t3 = r3 * cycle_time
    t4 = r4 * cycle_time

    # Resulting conveyor displacement for each movement
    p1 = t1 * conv_speed
    p2 = t2 * conv_speed
    p3 = t3 * conv_speed
    p4 = t4 * conv_speed

    # Time interval between each movements
    # d1 = -0.0183 # start rotating forward stop moving down
    # d2 = 0.0 # start moving up and stop rotatng forward
    # d3 = -0.043 # start rotating back and stop moving up
    d1 = r5 * cycle_time
    d2 = r6 * cycle_time
    d3 = r7 * cycle_time

    # Resulting conveyor displacement at each interval
    q1 = d1 * conv_speed
    q2 = d2 * conv_speed
    q3 = d3 * conv_speed

    print('Intervals')
    print('Move down: t1={:.3f}s, p1={:.2f}mm, {:.2f}% of cycle'.format(
        t1, p1, t1 / cycle_time * 100))
    print('Rotate forward: t2={:.3f}s, p2={:.2f}mm, {:.2f}% of cycle'.format(
        t2, p2, t2 / cycle_time * 100))
    print('Move up: t3={:.3f}s, p3={:.2f}mm, {:.2f}% of cycle'.format(
        t3, p3, t3 / cycle_time * 100))
    print('Rotate back: t4={:.3f}s, p4={:.2f}mm, {:.2f}% of cycle'.format(
        t4, p4, t4 / cycle_time * 100))
    print('')

    print('Overlaps (negative value)/delay (positive value):')
    print('Between start rotating forward and stop moving down:')
    print('  d1={:.3f}s, q1={:.2f}mm, {:.2f}% of cycle'.format(
        d1, q1, d1 / cycle_time * 100))
    print('Between start moving up and stop rotating forward:')
    print('  d2={:.3f}s, q2={:.2f}mm, {:.2f}% of cycle'.format(
        d2, q2, d2 / cycle_time * 100))
    print('Between start rotating back and stop moving up:')
    print('  d3={:.3f}s, q3={:.2f}mm, {:.2f}% of cycle'.format(
        d3, q3, d3 / cycle_time * 100))
    print('')

    print('Sum of all intervals (should be less than the cycle time):')
    print('t1+t2+t3+t4+d1+d2+d3={:.3f}s'.format(t1 + t2 + t3 + t4 + d1 + d2 + d3))
    print('p1+p2+p3+p4+q1+q2+q3={:.3f}mm'.format(p1 + p2 + p3 + p4 + q1 + q2 + q3))
    print('Cycle: time {:.3f}s, displacement {:.2f}mm'.format(cycle_time,
                                                              conv_speed * cycle_time))
    print('')

    # Start times
    # move down starts at t=0
    t01 = t1 + d1  # rotate forward
    t02 = t01 + t2 + d2  # move up
    t03 = t02 + t3 + d3  # rotate back

    t = np.arange(num_points) * cycle_time / num_points
    v1 = np.zeros(num_points)
    r1 = np.zeros(num_points)

    # Move down
    v1 = add_scurve_segment(v1, 0, t1, range_vert, t, neg=True)

    # Rotate forward
    r1 = add_scurve_segment(r1, t01, t2, range_ang, t)

    # Move up
    v1 = add_scurve_segment(v1, t02, t3, range_vert, t)

    # Rotate back
    r1 = add_scurve_segment(r1, t03, t4, range_ang, t, neg=True)

    # The other side
    v2 = np.concatenate((v1[int(num_points / 2):], v1[:int(num_points / 2)]))
    r2 = np.concatenate((r1[int(num_points / 2):], r1[:int(num_points / 2)]))

    if save:
        save_datapoints(t, v1, 'vertical1_1.txt')
        save_datapoints(t, r1, 'rotation1_1.txt')
        save_datapoints(t, v2, 'vertical2_1.txt')
        save_datapoints(t, r2, 'rotation2_1.txt')

    print("Angle when arm is down: ", r1[ind(t >= t1)], "deg")
    print("Time when arm height is down to 40mm: ", t[ind(v1 <= 40)], "mm")
    print("Time when arm height is down to 50mm: ", t[ind(v1 <= 50)], "mm")
    print('')

    if plot:
        # plt.plot([t1, t02, t02+t3], [0, 0, x], 'bo')
        # plt.plot([t01, t01+t2, t03, t03+t4], [0, a, a, 0], 'ro')

        P = t * conv_speed  # Position = speed * time

        plt.plot(P, v1, color='blue', label='Vertical Left')
        plt.plot(P, r1, color='deepskyblue', label='Rotation Left')
        plt.plot(P, v2, color='red', label='Vertical Right')
        plt.plot(P, r2, color='orange', label='Rotation Righr')

        # Interval1
        # print(np.array([t01, t01]) * conv_speed)
        # print([0, V1[ind(T>=t01)]])
        plt.plot(np.array([t01, t01]) * conv_speed,
                 [0, v1[ind(t >= t01)]], 'g',
                 label='Overlap: down-turn')
        plt.plot(np.array([t1, t1]) * conv_speed,
                 [0, r1[ind(t >= t1)]], 'g')

        # Interval2
        # plt.plot([t02, t02]*v, [0, R1[ind(T>=t02)]], 'k')
        # plt.plot([t01+t2, t01+t2]*v, [V1[ind(T>=t01+t2)], a], 'k')

        # Interval3
        # plt.plot([t02+t3, t02+t3]*v, [R1[ind(T>=t02+t3)], x], 'm')
        # plt.plot([t03, t03]*v, [V1[ind(T>=t03)], a], 'm')
        plt.plot(np.array([t02 + t3, t02 + t3]) * conv_speed,
                 [r1[ind(t >= t02 + t3)], range_vert], color='limegreen',
                 label='Overlap: up-turnback')
        plt.plot(np.array([t03, t03]) * conv_speed,
                 [v1[ind(t >= t03)], range_ang], color='limegreen')

        # End
        # plt.plot([t03+t4]*v, [0], 'ro')

        plt.yticks(np.arange(0, 94, step=10))
        plt.xticks(np.arange(0, cycle_time * conv_speed,
                             step=cycle_time * conv_speed / 20))
        plt.xlabel('Turning st. Infeed conveyor position (mm)')
        plt.ylabel('Linear position (mm) or rotation (deg)')
        plt.legend()
        plt.show()


def turner_profiles2(save=False, plot=False):
    """Creates all motion profiles based on constant conveyor speed."""

    # The first specified time values assumed max capacity of 8 turns per
    # second at a constant conveyor speed of 481mm/s.
    # Based on calculation in "capacityestimation.xlsm".
    conv_speed = 481  # mm/s
    # speed = position / time
    # position = speed * time

    # A one cycle by definition, includes two turns (to each side).
    # So eight turns per second are four cycles per second and one cycle
    # will take a quarter of a second.
    turnes_per_sec = 8
    cycle_time = 2 / turnes_per_sec  # sec

    print('Conv. speed: {} mm/s'.format(conv_speed))
    print('Turnes per second: {} '.format(turnes_per_sec))
    print('Cycle time: {} s'.format(cycle_time))
    print('')

    # time = 0.25 # sek, Total cycle time. Four cycles per second
    num_points = 2500  # num points
    range_vert = 65  # mm, vertical movement range
    range_ang = 90  # deg, rotational range

    # Intervals as a ratio with respect to cycle time/displacement
    r1 = 0.16  # Move down
    r2 = 0.44  # Rotate forward
    r3 = 0.24  # Move up
    r4 = 0.40  # Rotate back
    r5 = -0.0732  # Overlap between start r2 and stop r1
    r6 = 0.0  # Overlap/Delay between start r3 and stop r2
    r7 = -0.172  # Overlap between start r4 ans stop r3

    print('Interval ratios:')
    print('Move down: r1={:.2f}%'.format(r1 * 100))
    print('Rotate forward, r2={:.2f}%'.format(r2 * 100))
    print('Move up, r3={:.2f}%'.format(r3 * 100))
    print('Rotate back, r4={:.2f}%'.format(r4 * 100))
    print('Overlap between start r2 and stop r1, r5={:.2f}%'.format(r5 * 100))
    print('Overlap between start r3 and stop r2, r6={:.2f}%'.format(r6 * 100))
    print('Overlap between start r4 and stop r3, r7={:.2f}%'.format(r7 * 100))
    print('Sum of interval ratios: r1+r2+r3+r4+r5+r6+r7={:.2f}%'.format(
        (r1 + r2 + r3 + r4 + r5 + r6 + r7) * 100))
    print('')

    # Duration time for each movement
    t1 = 0.04  # move down
    t2 = 0.11  # rotate forward
    t3 = 0.06  # move up
    t4 = 0.1  # rotate back

    # Resulting conveyor displacement for each movement
    p1 = t1 * conv_speed
    p2 = t2 * conv_speed
    p3 = t3 * conv_speed
    p4 = t4 * conv_speed

    # Time interval between each movements
    d1 = -0.0183  # start rotating forward stop moving down
    d2 = 0.0  # start moving up and stop rotatng forward
    d3 = -0.043  # start rotating back and stop moving up

    # Resulting conveyor displacement at each interval
    q1 = d1 * conv_speed
    q2 = d2 * conv_speed
    q3 = d3 * conv_speed

    print('Intervals')
    print('Move down: t1={:.3f}s, p1={:.2f}mm, {:.2f}% of cycle'.format(
        t1, p1, t1 / cycle_time * 100))
    print('Rotate forward: t2={:.3f}s, p2={:.2f}mm, {:.2f}% of cycle'.format(
        t2, p2, t2 / cycle_time * 100))
    print('Move up: t3={:.3f}s, p3={:.2f}mm, {:.2f}% of cycle'.format(
        t3, p3, t3 / cycle_time * 100))
    print('Rotate back: t4={:.3f}s, p4={:.2f}mm, {:.2f}% of cycle'.format(
        t4, p4, t4 / cycle_time * 100))
    print('')

    print('Overlaps (negative value)/delay (positive value):')
    print('Between start rotating forward and stop moving down:')
    print('  d1={:.3f}s, q1={:.2f}mm, {:.2f}% of cycle'.format(
        d1, q1, d1 / cycle_time * 100))
    print('Between start moving up and stop rotating forward:')
    print('  d2={:.3f}s, q2={:.2f}mm, {:.2f}% of cycle'.format(
        d2, q2, d2 / cycle_time * 100))
    print('Between start rotating back and stop moving up:')
    print('  d3={:.3f}s, q3={:.2f}mm, {:.2f}% of cycle'.format(
        d3, q3, d3 / cycle_time * 100))
    print('')

    print('Sum of all intervals (should be less than the cycle time):')
    print('t1+t2+t3+t4+d1+d2+d3={:.3f}s'.format(t1 + t2 + t3 + t4 + d1 + d2 + d3))
    print('p1+p2+p3+p4+q1+q2+q3={:.3f}mm'.format(p1 + p2 + p3 + p4 + q1 + q2 + q3))
    print('Cycle: time {:.3f}s, displacement {:.2f}mm'.format(cycle_time,
                                                              conv_speed * cycle_time))
    print('')

    # Start times
    # move down starts at t=0
    t01 = t1 + d1  # rotate forward
    t02 = t01 + t2 + d2  # move up
    t03 = t02 + t3 + d3  # rotate back

    t = np.arange(num_points) * cycle_time / num_points
    v1 = np.zeros(num_points)
    r1 = np.zeros(num_points)

    # Move down
    v1 = add_scurve_segment(v1, 0, t1, range_vert, t, neg=True)

    # Rotate forward
    r1 = add_scurve_segment(r1, t01, t2, range_ang, t)

    # Move up
    v1 = add_scurve_segment(v1, t02, t3, range_vert, t)

    # Rotate back
    r1 = add_scurve_segment(r1, t03, t4, range_ang, t, neg=True)

    # The other side
    v2 = np.concatenate((v1[int(num_points / 2):], v1[:int(num_points / 2)]))
    r2 = np.concatenate((r1[int(num_points / 2):], r1[:int(num_points / 2)]))

    if save:
        save_datapoints(t, v1, 'vertical1_1.txt')
        save_datapoints(t, r1, 'rotation1_1.txt')
        save_datapoints(t, v2, 'vertical2_1.txt')
        save_datapoints(t, r2, 'rotation2_1.txt')

    print("Angle when arm is down: ", r1[ind(t >= t1)], "deg")
    print("Time when arm height is down to 40mm: ", t[ind(v1 <= 40)], "mm")
    print("Time when arm height is down to 50mm: ", t[ind(v1 <= 50)], "mm")
    print('')

    if plot:
        # plt.plot([t1, t02, t02+t3], [0, 0, x], 'bo')
        # plt.plot([t01, t01+t2, t03, t03+t4], [0, a, a, 0], 'ro')

        P = t * conv_speed  # Position = speed * time

        plt.plot(P, v1, 'b', label='Vertical')  # Vertical movement
        plt.plot(P, r1, 'r', label='Rotation')  # Rotation
        plt.plot(P, v2, 'g', label='Other side Vertical')
        plt.plot(P, r2, 'm', label='Other side Rotation')

        # Interval1
        print([t01, t01] * conv_speed)
        print([0, v1[ind(t >= t01)]])
        # plt.plot([t01, t01]*v, [0, V1[ind(T>=t01)]], 'g')
        # plt.plot([t1, t1]*v, [0, R1[ind(T>=t1)]], 'g')

        # Interval2
        # plt.plot([t02, t02]*v, [0, R1[ind(T>=t02)]], 'k')
        # plt.plot([t01+t2, t01+t2]*v, [V1[ind(T>=t01+t2)], a], 'k')

        # Interval3
        # plt.plot([t02+t3, t02+t3]*v, [R1[ind(T>=t02+t3)], x], 'm')
        # plt.plot([t03, t03]*v, [V1[ind(T>=t03)], a], 'm')

        # End
        # plt.plot([t03+t4]*v, [0], 'ro')

        plt.yticks(np.arange(0, 94, step=10))
        plt.xticks(np.arange(0, cycle_time * conv_speed,
                             step=cycle_time * conv_speed / 20))
        plt.xlabel('Conveyor position (mm)')
        plt.ylabel('Linear position (mm) or rotation (deg)')
        plt.legend()
        plt.show()


def turner_profiles(save=False, plot=False):
    """Creates all motion profiles based on fixed time specs."""

    # The first specified time values assumed that products were travelling
    # constantly at speed 250mm/s.
    # And that spacing between products was 20mm.
    convspeed = 481  # mm/s
    # speed = position / time
    # position = speed * time

    time = 0.25  # sek, Total cycle time for turning two products.
    # Resulting in 8 pieces turned per second.
    num = 2500  # num points
    x = 65  # mm, vertical movement range
    a = 90  # deg, rotational range

    # Duration time for each movement
    t1 = 0.04  # move down
    t2 = 0.11  # rotate forward
    t3 = 0.06  # move up
    t4 = 0.1  # rotate back

    # Time interval between each movements
    d1 = -0.0183  # start rotating forward stop moving down
    d2 = 0.0  # start moving up and stop rotatng forward
    d3 = -0.043  # start rotating back and stop moving up

    print('Time intervals:')
    print("t1=", t1, ", Move down")
    print("t2=", t2, ", Rotate forward")
    print("t3=", t3, ", Move up")
    print("t4=", t4, ", Rotate back")
    print("d1=", d1, ", between start rotating forward stop moving down")
    print("d2=", d2, ", between start moving up and stop rotatng forward")
    print("d3=", d3, ", between start rotating back stop end moving up")
    print('')
    print('Sum of all time intervals (should be less than the cycle time):')
    print("t1+t2+t3+t4+d1+d2+d3=", t1 + t2 + t3 + t4 + d1 + d2 + d3)
    print("Cycle time=", time, ' sek')

    # Start times
    # move down starts at t=0
    t01 = t1 + d1  # rotate forward
    t02 = t01 + t2 + d2  # move up
    t03 = t02 + t3 + d3  # rotate back

    t = np.arange(num) * time / num
    v1 = np.zeros(num)
    r1 = np.zeros(num)

    # Move down
    v1 = add_scurve_segment(v1, 0, t1, x, t, neg=True)

    # Rotate forward
    r1 = add_scurve_segment(r1, t01, t2, a, t)

    # Move up
    v1 = add_scurve_segment(v1, t02, t3, x, t)

    # Rotate back
    r1 = add_scurve_segment(r1, t03, t4, a, t, neg=True)

    # The other side
    v2 = np.concatenate((v1[int(num / 2):], v1[:int(num / 2)]))
    r2 = np.concatenate((r1[int(num / 2):], r1[:int(num / 2)]))

    if save:
        save_datapoints(t, v1, 'vertical1_1.txt')
        save_datapoints(t, r1, 'rotation1_1.txt')
        save_datapoints(t, v2, 'vertical2_1.txt')
        save_datapoints(t, r2, 'rotation2_1.txt')

    print("Angle when arm is down: ", r1[ind(t >= t1)], "deg")
    print("Time when arm height is down to 40mm: ", t[ind(v1 <= 40)], "mm")
    print("Time when arm height is down to 50mm: ", t[ind(v1 <= 50)], "mm")

    if plot:
        # plt.plot([t1, t02, t02+t3], [0, 0, x], 'bo')
        # plt.plot([t01, t01+t2, t03, t03+t4], [0, a, a, 0], 'ro')

        plt.plot(t, v1, 'b', label='Vertical')  # Vertical movement
        plt.plot(t, r1, 'r', label='Rotation')  # Rotation
        plt.plot(t, v2, 'g', label='Other side Vertical')
        plt.plot(t, r2, 'm', label='Other side Rotation')

        # Interval1
        plt.plot([t01, t01], [0, v1[ind(t >= t01)]], 'g')
        plt.plot([t1, t1], [0, r1[ind(t >= t1)]], 'g')

        # Interval2
        plt.plot([t02, t02], [0, r1[ind(t >= t02)]], 'k')
        plt.plot([t01 + t2, t01 + t2], [v1[ind(t >= t01 + t2)], a], 'k')

        # Interval3
        plt.plot([t02 + t3, t02 + t3], [r1[ind(t >= t02 + t3)], x], 'm')
        plt.plot([t03, t03], [v1[ind(t >= t03)], a], 'm')

        # End
        plt.plot([t03 + t4], [0], 'ro')

        plt.xlabel('Time (s)')
        plt.ylabel('Linear position (mm) or rotation (deg)')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    turner_profiles_pos(save=False, plot=True)
    # turner_profiles(save=False, plot=True)
    # turner_profiles2(save=False, plot=True)
