
import numpy as np
import matplotlib.pyplot as plt


class MotionProfile:
    """An abstract class, which other motion profiles should inherit and implement."""
    def __init__(self):
        self.n = None  # Number of data points
        self.dist = None  # Total distance

        self.time = None  # Time duration
        self.v_max = None  # Max velocity
        self.accel = None  # (Max) acceleration

        self.v_avg = None  # dist/time

        # Arrays
        self.t = None  # Time array of size n + 1
        self.pos = None  # Position array of size n + 1
        self.vel = None  # Velocity array of size n + 1
        self.acc = None  # Acceleration array of size n + 1
        self.jerk = None  # Jerk array of size n + 1

        self.d = {'Vmax': 0,
                  'Vavg': 0,
                  'A': 0,
                  'time': 0,
                  'dist': 0,
                  't': None,  # time array
                  'j': None,  # jerk array
                  'a': None,  # acceleration array
                  'v': None,  # velocity array
                  'p': None}  # position array

    def calc(self, n):
        """Create the t, p, v, a and j arrays with sizes n + 1."""
        return self.d


def trapezoidal(dist, time, accel, n=100):
    """Returns a trapezoidal motion profile; 2nd order polynomial

    dist -- the displacement distance (movement length)
    time -- the time duration of the displacement, cannot be zero.
    accel -- the acceleration and deceleration parts of the trapezoid
    n -- the number of ponts to return

    If the 'accel' is too small relative to 'time' and 'dist', it will
    be scaled down to form a triangular motion profile.

    The max speed of a trapezoid is always lower than on a triangular motion.
    """

    accel_triangular = dist / (time / 2)**2
    print('accel_triangular: ', str(accel_triangular))
    if accel > accel_triangular:
        # known: x, t, a
        # unknown: t_a, t_c
        # t = t_a + t_c + t_a = 2 t_a + t_c
        # t_c = t - 2 t_a
        # The distance travelled is the area under the curve:
        # x = v_max * (t_a + t_c)
        # a = v_max / t_a
        # v_max = a t_a
        # x = a t_a (t_a + (t - 2 t_a))
        # x = a t_a (t - t_a)
        # x = (a t) t_a - (a) t_a^2
        # (-a) t_a^2 + (a t) t_a - x = 0
        # (a) t_a^2 + (-a t) t_a + x = 0
        # [a x^2 + b x + c = 0 => x = (-b +/- sqrt(b^2 - 4 a c)) / 2 a]
        #
        # t_a = (a t +/- sqrt((a t)^2 - 4 a x)) / 2a
        s1 = accel * time
        s2 = 4 * accel * dist
        if s1**2 > s2:
            # The square root is not negative.
            # t_a1 = (accel * time + np.sqrt((accel * time)**2 - 4 * accel * dist)) / (2 * accel)
            # t_a2 = (accel * time - np.sqrt((accel * time)**2 - 4 * accel * dist)) / (2 * accel)
            # print(time, dist, accel, t_a1, t_a2)

            t_a = (s1 - np.sqrt(s1**2 - s2)) / (2 * accel)
            t_c = time - 2 * t_a
            # print('time: ', time, ', t_a: ', t_a, ', t_c: ', t_c)

            d = {'Vmax': accel * t_a,
                 'Vavg': dist / time,
                 'A': accel,
                 'time': time,
                 'dist': dist,
                 't': np.arange(n + 1) * time / n,
                 'jerk': np.zeros(n + 1),
                 'acc': np.zeros(n + 1),
                 'vel': np.zeros(n + 1),
                 'pos': np.zeros(n + 1)}

            for i, t in zip(range(n + 1), d['t']):
                if 0 <= t < t_a:
                    d['pos'][i] = 1 / 2 * d['A'] * t ** 2
                    d['vel'][i] = d['A'] * t
                    d['acc'][i] = d['A']

                elif t_a <= t < t_a + t_c:
                    d['pos'][i] = d['Vmax'] * t - (1 / 2 * d['A'] * t_a ** 2)
                    d['vel'][i] = d['Vmax']
                    d['acc'][i] = 0

                elif t_a + t_c <= t <= time:
                    d['pos'][i] = d['Vmax'] * t - (1 / 2 * d['A'] * t_a ** 2)\
                                  - 1 / 2 * d['A'] * (t - (t_a + t_c)) ** 2
                    d['vel'][i] = d['Vmax'] - (t - (t_a + t_c)) * d['A']
                    d['acc'][i] = -d['A']

        else:
            d = triangular(dist, time, n)
            print('(a t)^2 - 4 a x = ', str(s1**2), ' - ', str(s2))
            print('distance, time, acceleration combination is not possible')

    else:
        d = triangular(dist, time, n)
        print('The acceleration is to small: triangular accel = ', str(d['A']), ', accel = ', str(accel))
        print('returning a triangular motion profile with the specified distance and time.')

    return d


def triangular(dist, time=None, vel=None, accel=None, n=100):
    """Returns a triangular motion profile, 2nd order polynomial

    dist -- the displacement distance (movement length)
    time -- the time duration of the displacement, cannot be zero.
    n -- the number of ponts to return

    one of three: time, vel or accel must be non-None and
    two of three must be None.
    """

    if not time:
        if vel and not accel:
            time = 2 * dist / vel
        elif accel and not vel:
            time = 2 * np.sqrt(dist / accel)
        else:
            return None

    d = {'Vmax': dist / (time / 2),
         'Vavg': dist / time,
         'A': 0,
         'time': time,
         'dist': dist,
         't': np.arange(n+1) * time / n,
         'jerk': np.zeros(n+1),
         'acc': np.zeros(n+1),
         'vel': np.zeros(n+1),
         'pos': np.zeros(n+1)}
    d['A'] = d['Vmax'] / (time / 2)

    for i, t in zip(range(n+1), d['t']):
        if 0 <= t < time / 2:
            d['pos'][i] = 1 / 2 * d['A'] * t ** 2
            d['vel'][i] = d['A'] * t
            d['acc'][i] = d['A']

        elif time / 2 <= t <= time:
            d['pos'][i] = -dist / 2 + d['Vmax'] * t - 1 / 2 * d['A'] * (t - time / 2) ** 2
            d['vel'][i] = d['Vmax'] - (t - time / 2) * d['A']
            d['acc'][i] = -d['A']

    return d


class Trapezoidal(MotionProfile):
    def __init__(self, dist, time=None, v_max=None, accel=None, n=None):
        """Two of time, v_max or accel must be given.

        Necessary argument combinations:
            accel, time
            accel, v_max
            time, v_max -> will allways be a triangular profile

        a_max is the maximum allowable acceleration.
        """
        MotionProfile.__init__(self)
        self.dist = dist
        self.n = n
        self.is_triangular = False
        self.is_trapezoidal = False
        self.t_a = None  # The acceleration and deceleration time.
        # self.tc = None  # The time at constant speed.
        self.tri = None

        if accel:
            if time and not v_max:
                self.time = time
                self.accel = accel
                s1 = accel * time
                s2 = 4 * accel * dist
                accel_triangular = dist / (time / 2) ** 2
                if (s1**2 < s2) or (accel < accel_triangular):
                    self.t_c = 0
                    self.tri = Triangular(dist, time, accel, n)
                    if n:
                        self.d = self.tri.calc(n)
                else:
                    self.is_trapezoidal = True
                    self.t_a = s1 - np.sqrt(s1**2 - s2)
                    self.t_c = self.time - 2 * self.t_a
                    self.v_max = accel * self.t_a

            elif v_max and not time:
                self.v_max = v_max
                self.accel = accel
                self.t_a = v_max / accel
                self.t_c = dist / v_max - v_max / accel

                if self.t_c <= 0:
                    self.time = 2 * self.t_a
                    self.t_c = 0
                    self.tri = Triangular(dist, self.time, accel, n)
                    if n:
                        self.d = self.tri.calc(n)
                else:
                    self.time = 2 * self.t_a + self.t_c
                    self.is_trapezoidal = True

        elif time and v_max:
            self.time = time
            self.v_max = v_max
            self.t_a = time / 3
            self.t_c = time / 3
            self.accel = v_max / self.t_a
            self.is_trapezoidal = True

        if self.is_triangular:
            self.t_c = 0
            self.tri = Triangular(dist, time, v_max, accel, n)
            if n:
                self.d = self.tri.calc(n)

        elif self.is_trapezoidal:
            self.v_avg = self.dist / self.time
            self.d['Vmax'] = self.v_max
            self.d['Vavg'] = self.v_avg
            self.d['A'] = self.accel
            self.d['time'] = self.time
            self.d['dist'] = self.dist
            if n:
                # Calculate if the number of points n, are specified.
                self.d = self.calc(self.n)

    def calc(self, n):
        self.n = n
        # print(self.time)
        if self.tri:
            return self.tri.calc(n)

        else:
            self.d['t'] = np.arange(n + 1) * self.time / n
            self.d['j'] = np.zeros(n + 1)
            self.d['a'] = np.zeros(n + 1)
            self.d['v'] = np.zeros(n + 1)
            self.d['p'] = np.zeros(n + 1)

            for i, t in zip(range(n + 1), self.d['t']):
                if 0 <= t < self.t_a:
                    self.d['p'][i] = 1 / 2 * self.d['A'] * t ** 2
                    self.d['v'][i] = self.d['A'] * t
                    self.d['a'][i] = self.d['A']

                elif self.t_a <= t < self.t_a + self.t_c:
                    self.d['p'][i] = self.d['Vmax'] * t - (1 / 2 * self.d['A'] * self.t_a ** 2)
                    self.d['v'][i] = self.d['Vmax']
                    self.d['a'][i] = 0

                elif self.t_a + self.t_c <= t <= self.time:
                    self.d['p'][i] = self.d['Vmax'] * t - (1 / 2 * self.d['A'] * self.t_a ** 2) \
                                  - 1 / 2 * self.d['A'] * (t - (self.t_a + self.t_c)) ** 2
                    self.d['v'][i] = self.d['Vmax'] - (t - (self.t_a + self.t_c)) * self.d['A']
                    self.d['a'][i] = -self.d['A']

            return self.d


class Triangular(MotionProfile):
    def __init__(self, dist, time=None, v_max=None, accel=None, n=None):
        """One of time, v_max or accel must be given."""
        MotionProfile.__init__(self)
        self.dist = dist
        self.n = n

        if not time:
            if v_max and not accel:
                self.time = 2 * dist / v_max
                self.v_max = v_max
                self.accel = self.v_max / (self.time / 2)
                self.v_avg = self.dist / self.time

            elif accel and not v_max:
                self.time = 2 * np.sqrt(dist / accel)
                self.v_max = self.dist / (self.time / 2)
                self.accel = accel
                self.v_avg = self.dist / self.time
        else:
            self.time = time
            self.v_max = self.dist / (self.time / 2)
            self.accel = self.v_max / (self.time / 2)
            self.v_avg = self.dist / self.time

        self.d['Vmax'] = self.v_max
        self.d['Vavg'] = self.v_avg
        self.d['A'] = self.accel
        self.d['time'] = self.time
        self.d['dist'] = self.dist

        # Calculate if the number of points n, are specified
        if n:
            self.d = self.calc(self.n)

    def calc(self, n):
        self.n = n
        self.d['t'] = np.arange(n + 1) * self.time / n
        self.d['j'] = np.zeros(n + 1)
        self.d['a'] = np.zeros(n + 1)
        self.d['v'] = np.zeros(n + 1)
        self.d['p'] = np.zeros(n + 1)

        for i, t in zip(range(n + 1), self.d['t']):
            if 0 <= t < self.time / 2:
                self.d['p'][i] = 1 / 2 * self.d['A'] * t ** 2
                self.d['v'][i] = self.d['A'] * t
                self.d['a'][i] = self.d['A']

            elif self.time / 2 <= t <= self.time:
                self.d['p'][i] = -self.dist / 2 + self.d['Vmax'] * t - 1 / 2 * self.d['A'] * (t - self.time / 2) ** 2
                self.d['v'][i] = self.d['Vmax'] - (t - self.time / 2) * self.d['A']
                self.d['a'][i] = -self.d['A']

        self.t = self.d['t']
        self.pos = self.d['p']
        self.vel = self.d['v']
        self.accel = self.d['a']
        self.jerk = self.d['j']

        return self.d


def scurve(dist, time, n=100):
    """Returns S-curve motion profile, 3rd order polynomial.

    dist -- the displacement distance (movement length)
    time -- the time duration of the displacement, cannot be zero.
    n -- the number of ponts to return
    """

    d = {
        'Vs': dist / (time / 2),
        'A': 0,
        'As': 0,
        'J': 0,
        't': np.arange(n+1) * time / n,
        'jerk': np.zeros(n+1),
        'acc': np.zeros(n+1),
        'vel': np.zeros(n+1),
        'pos': np.zeros(n+1),
        'trivel': np.zeros(n+1)
    }

    d['A'] = d['Vs'] / (time / 2)
    d['As'] = 2 * d['A']
    d['J'] = d['Vs'] / (time / 4) ** 2

    j = d['J']
    v = d['Vs']
    s = d['As']
    a = d['A']

    for i, t in zip(range(n+1), d['t']):
        if 0 <= t < time / 4:
            d['jerk'][i] = j
            d['acc'][i] = j * t
            d['vel'][i] = 1 / 2 * j * t ** 2
            d['pos'][i] = 1 / 6 * j * t ** 3
            d['trivel'][i] = a * t

        elif time / 4 <= t < time / 2:
            d['jerk'][i] = -j
            d['acc'][i] = s - j * (t - time / 4)
            d['vel'][i] = v / 2 + s * (t - time / 4) - 1 / 2 * j * (t - time / 4) ** 2
            d['pos'][i] = 1 / 6 * j * (time / 4) ** 3 + v / 2 * (t - time / 4) + 1 / 2 * s * (
                    t - time / 4) ** 2 - 1 / 6 * j * (t - time / 4) ** 3
            d['trivel'][i] = a * t

        elif time / 2 <= t < 3 / 4 * time:
            d['jerk'][i] = -j
            d['acc'][i] = -j * (t - time / 2)
            d['vel'][i] = v - 1 / 2 * j * (t - time / 2) ** 2
            d['pos'][i] = v / 2 * time / 4 + s / 2 * (time / 4) ** 2 + v * (
                    t - time / 2) - 1 / 6 * j * (t - time / 2) ** 3
            d['trivel'][i] = v - (t - time / 2) * a

        elif 3 / 4 * time <= t <= time:
            d['jerk'][i] = j
            d['acc'][i] = -s + j * (t - 3 / 4 * time)
            d['vel'][i] = v / 2 - s * (t - 3 / 4 * time) + 1 / 2 * j * (t - 3 / 4 * time) ** 2
            d['pos'][i] = 3 / 2 * v * time / 4 + s / 2 * (time / 4) ** 2 - j / 6 * (
                    time / 4) ** 3 + v / 2 * (t - 3 / 4 * time) - 1 / 2 * s * (
                                  t - 3 / 4 * time) ** 2 + 1 / 6 * j * (t - 3 / 4 * time) ** 3
            d['trivel'][i] = v - (t - time / 2) * a

    return d


def const_speed_profile(time, numpoints, speed, filename='constspeed'):
    # time = 0.250 # s
    # numpoints = 2500
    # speed = 250 # mm/s
    t = np.arange(numpoints) * time / numpoints
    pos = speed * t
    save_datapoints(t, pos, filename + '.txt')


def const_move(velocity, time, n=100):
    """returns positon vs time"""
    d = {}
    d['t'] = np.arange(n) * time / n
    d['p'] = velocity * d['t']
    return d


def replace_range(x, y, i0):
    """ Replace part of x with y, beginning at position i0.

    x -- array
    y -- array
    i0 -- initial value

    The values in x at position i0 and above, up to the length of y,
    will get the values of y. The values in x at position i0+len(y),
    will all get the value of y[-1].

    Example:
        x = [1 2 3 4 5 6 7 8 9]
        y = [a b c]
        i0 = 3
        Returns: x = [1 2 3 a b c c c c]
    """

    # For loop example:
    #     x = [1 2 3 4 5 6 7 8 9]
    #     y = [a b c]
    #     i0 = 3
    # Results in: x = [1 2 3 a b c 7 8 9]
    for i, j in zip(range(i0, i0 + len(y)), range(len(y))):
        x[i] = y[j]

    # The iterator i, has reached the last position of the replaced values.
    # The upper part of x, above the replaced values from y, is replaced with
    # the last value of y. Example:
    # Returned result: x = [1 2 3 a b c c c c].
    return np.concatenate((x[0:i], y[-1] * np.ones(len(x) - i)))


def add_scurve_segment(p, t0, time, displace, t, neg=False):
    """ Returns a position array p, with an added S-curve array.

    p -- Position array
    t0 -- The moment in time to start adding the S-curve to p.
    time -- The time it takes to displace.
    displace -- A value giving the displacement range
    t -- Time array
    neg -- Bool, Return a negative S-curve or not.

    The arrays, p and t have the same length.
    The positon values in p take place at time values in t,
    at index i in t and p.
    """
    # The number of points, the S-curve should return
    n = int(time * len(p) / (np.max(t) + t[1]))

    d = scurve(displace, time, n)
    if neg:
        d['p'] = displace - d['p']

    # The index of the start time in the t time array.
    i0 = np.where(t >= t0)[0][0]
    return replace_range(p, d['p'], i0)


def ind(expr):
    """Returns the first index in an array, where  the expression is true"""
    return np.where(expr)[0][0]


def save_datapoints(time, pos, filename):
    """Creates a file containing two columns; time and position"""
    with open(filename, 'w') as f:
        for t, p in zip(time, pos):
            f.write(str(t) + ',' + str(p) + '\n')


def plot_motion_profile(d):
    """Plots pos, vel, acc and jerk.

    The argument is a tuple of dictionaries.
    """
    f, ax = plt.subplots(2, 2)

    # print(d)

    ax[0, 0].set_title('Pos')
    ax[0, 1].set_title('Vel')
    ax[1, 0].set_title('Acc')
    ax[1, 1].set_title('Jerk')

    if isinstance(d, tuple):
        j = 1
        for i in d:
            if i is not None:
                ax[0, 0].plot(i['t'], i['p'], label=str(j))
                ax[0, 1].plot(i['t'], i['v'])
                ax[1, 0].plot(i['t'], i['a'])
                if 'jerk' in i:
                    ax[1, 1].plot(i['t'], i['j'])

                j += 1
                ax[0, 0].legend()
    else:
        if d is not None:
            ax[0, 0].plot(d['t'], d['p'])
            ax[0, 1].plot(d['t'], d['v'])
            ax[1, 0].plot(d['t'], d['a'])
            if 'j' in d:
                ax[1, 1].plot(d['t'], d['j'])

    plt.show()


def shift(x, x0, s=0):
    """Shift the first elements in the array x, to position x0, x0 <= len(x).

    Example:
        x0 = 4
        x = a b c d e f g h i j
            0 1 2 3 4 5 6 7 8 9

        y = s s s s a b c d e f
    """
    # y = s * np.ones(len(x))
    #
    # i = x0
    # j = 0
    # while i < len(x):
    #     y[i] = x[j]
    #     i += 1
    #     j += 1
    #
    # return y
    return np.concatenate((s * np.ones(x0), x[:len(x) - x0]))


if __name__ == '__main__':
    # s = scurve(0.090, 0.1)
    # print('scurve accel:', d['acc'].max())

    # plt.plot(d['t'], d['jerk'])
    # plt.plot(d['t'], d['acc'])
    # plt.plot(d['t'], d['vel'])
    # plt.plot(d['t'], d['pos'])
    # plt.plot(d['t'], d['trivel'])

    # d = const_move(0, 0.1)
    # plt.plot(d['t'], d['p'])

    # z = trapezoidal(0.090, 0.1, 40)
    # plot_motion_profile((z,))

    # d = triangular(0.090, 0.1)
    # print('triangular accel:', d['acc'].max())
    # plot_motion_profile((d,))

    # plot_motion_profile((s, z, d))

    # const_speed_profiles(time=0.250, numpoints=2500, speed=400,
    #         filename='constspeed400')

    # plt.show()  # print(data)

    # shift([1, 2, 3, 4, 5, 6, 7, 8, 9], 4)

    # d1 = triangular(10, 2*0.0155+0.0528, n=1000)
    # d2 = trapezoidal(10, 2*0.0155+0.0528, 10000, n=1000)
    # # Distance and acceleretaion are known: x, a. Time is unknown: t
    # # x = v (t/2), a = v / (t/2) => x = a (t/2) (t/2)
    # # => t = 2 sqrt(x/a)
    # d3 = triangular(10, 2 * np.sqrt(10 / 10000), n=1000)
    # # Distance and velocity is known: x, v. Time is unknown: t
    # # v = x / t => t = x / v
    # d4 = triangular(10, 2 * 10 / 148, n=1000)
    # tri5 = Triangular(10, v_max=148)
    # d5 = tri5.calc(1000)
    #
    # print('d1 vavg', d1['Vavg'])
    # print('d2 vavg', d2['Vavg'])
    # print('d3 vavg', d3['Vavg'])
    # print('d4 vavg', d4['Vavg'])
    # print('d5 vavg', d5['Vavg'])
    #
    # plot_motion_profile((d1, d2, d3, d4, d5))

    # Trapezoidal 1: accel and time
    # d = []
    # for i in range(1000, 16000, 2000):
    #     tra = Trapezoidal(10, accel=i, v_max=200, n=1000)
    #     d.append(tra.d)
    # plot_motion_profile(tuple(d))

    # Trapezoidal 2: accel and v_max
    # d = []
    # for i in range(1000, 50000, 1000):
    #     tra = Trapezoidal(10, accel=i, v_max=200, n=1000)
    #     d.append(tra.d)
    # plot_motion_profile(tuple(d))

    # Trapezoidal 3: time and v_max
    d = []
    # tra = Trapezoidal(10, time=0.05, v_max=200, n=1000)
    # d.append(tra.d)
    # tra = Trapezoidal(10, time=0.05, v_max=100, n=1000)
    # d.append(tra.d)
    # tra = Trapezoidal(10, time=0.05, v_max=100, n=1000)
    # tra.d['p'] = -tra.d['p']
    # tra.d['v'] = -tra.d['v']
    # tra.d['a'] = -tra.d['a']
    # d.append(tra.d)

    tra = Trapezoidal(5, accel=10000, v_max=143, n=3000)
    d.append(tra.d)
    plot_motion_profile(tuple(d))



