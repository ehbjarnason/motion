import numpy as np
import matplotlib.pyplot as plt


class CamProfile:
    """A base class, which other cam profiles should extend.

    Essential format:
        ...
        t: [t0 t1 ... tN],      -- Time Steps
      pos: [[p10 p11 ... p1N],  -- Position steps for each movement
            [p20 p21 ... p2N],     (M pieces)
            ...
            [pM0 pM1 ... pMN]],
      vel: [[v10 v11 ... v1N],  -- Velocity steps for each movement
            [v20 v21 ... v2N],
            ...
            [vM0 vM1 ... vMN]]
        ...
    """
    def __init__(self):
        self.name = ''
        self.n = None  # Number of time steps
        self.dist = None  # Total distance
        self.time = None  # Total time duration
        self.v_max = None  # Max velocity
        self.accel = None  # (Max) acceleration
        self.jerk = None
        self.v_avg = None  # Average velocity, dist/time

        # Arrays
        self.t = None  # Time array, size Nx1
        self.pos = None  # Position array, size MxN
        self.vel = None  # Velocity array, size MxN
        self.acc = None  # Acceleration array, size MxN
        self.jerk = None  # Jerk array, size MxN

    def __call__(self, num):
        """Create motion arrays"""
        raise NotImplementedError


class Quadratic(CamProfile):
    def __init__(self, dist, time=None, v_max=None, accel=None, num=None, name='Quad'):
        """A quatratic (2nd degree polynomial) cam profile object

        dist -- the movement distance. In length-units or angular-units.
        time -- (optional) the time duration of the movement.
        v_max -- (optional) the maximum allowable speed to be reached.
        accel -- (optional) the acceleration and deceleration to be used.
        num -- (optional) the number of time steps.

        One of time, v_max or accel can be given.
        Or two of time, v_max or accel can also be given.

        TODO expand for NxM arrays of dist, v_max, and accel
        """
        if dist > 0:
            CamProfile.__init__(self)
            self.dist = dist
            self.n = num
            self.t_a = None  # The acceleration and deceleration time.
            self.t_c = None  # The time at constant speed.
            self.name = name

            if accel:
                self.accel = accel

                if time and not v_max:
                    # Given: dist, accel, time
                    self.time = time
                    discriminant = time ** 2 - 4 * dist / accel

                    if discriminant > 0:
                        # A trapezoidal cam profile
                        self.t_a = .5 * (time - np.sqrt(discriminant))
                        self.t_c = time - 2 * self.t_a
                        self.v_max = accel * self.t_a
                    else:
                        # A triangular cam profile
                        # Recalculate (increase) accel
                        self.v_max = dist / (time / 2)
                        self.accel = self.v_max / (time / 2)

                elif not time and v_max:
                    # Given: dist, accel, v_max
                    self.v_max = v_max
                    self.t_a = v_max / accel
                    self.t_c = dist / v_max - v_max / accel

                    if self.t_c <= 0:
                        # A triangular cam profile
                        # Keep the acceleration fixed and recalculate (lower) v_max
                        self.v_max = np.sqrt(dist * accel)
                        self.time = 2 * self.v_max / accel
                        self.t_a, self.t_c = None, None
                    else:
                        # A trapezoidal cam profile
                        self.time = 2 * self.t_a + self.t_c

                else:
                    # Given: accel
                    # A triangular cam profile
                    self.time = 2 * np.sqrt(dist / accel)
                    self.v_max = dist / (self.time / 2)

            elif time:
                self.time = time

                if v_max:
                    # Given: dist, time, v_max
                    self.v_max = v_max
                    self.t_a = time - dist / v_max
                    self.t_c = time - 2 * self.t_a

                    if self.t_c <= 0:
                        # A triangular cam profile
                        # The speed, v_max is unnecessary high and will be
                        # recalculated (lowered) based on dist and time.
                        self.v_max = dist / (time / 2)
                        self.accel = self.v_max / (time / 2)
                        self.t_a, self.t_c = None, None
                    else:
                        # A trapezoidal cam profile
                        self.accel = v_max / self.t_a
                else:
                    # Given: dist, time
                    # A triangular cam profile
                    self.v_max = dist / (time / 2)
                    self.accel = self.v_max / (time / 2)

            elif v_max:
                # Given: dist, v_max
                # A triangular cam profile
                self.v_max = v_max
                self.time = 2 * dist / v_max
                self.accel = v_max / (self.time / 2)

            self.v_avg = dist / self.time

            if num:
                # only create arrays if the number of time steps are given
                self.__call__(num)

    def __str__(self):
        return self.name

    def __call__(self, num):
        self.n = num
        self.t = np.arange(num) * self.time / num
        self.pos = np.zeros(num)
        self.vel = np.zeros(num)
        self.acc = np.zeros(num)

        if self.accel is None or self.v_max is None or self.time is None:
            return

        if self.t_a is None and self.t_c is None:
            # A triangular cam profile
            i1 = np.where(np.logical_and(0 <= self.t, self.t < .5 * self.time))[0]
            i2 = np.where(.5 * self.time <= self.t)[0]

            self.pos[i1] = 1 / 2 * self.accel * self.t[i1] ** 2
            self.pos[i2] = -self.dist / 2 + self.v_max * self.t[i2] - 1 / 2 * self.accel * (
                    self.t[i2] - self.time / 2) ** 2

            self.vel[i1] = self.accel * self.t[i1]
            self.vel[i2] = self.v_max - (self.t[i2] - self.time / 2) * self.accel

            self.acc[i1] = self.accel
            self.acc[i2] = -self.accel
        else:
            # A trapezoidal cam profile
            i1 = np.where(np.logical_and(0 <= self.t, self.t < self.t_a))[0]
            i2 = np.where(np.logical_and(self.t_a <= self.t, self.t < self.t_a + self.t_c))[0]
            i3 = np.where(self.t_a + self.t_c <= self.t)[0]

            self.pos[i1] = .5 * self.accel * self.t[i1] ** 2
            self.pos[i2] = self.v_max * self.t[i2] - (1 / 2 * self.accel * self.t_a ** 2)
            self.pos[i3] = self.v_max * self.t[i3] - (1 / 2 * self.accel * self.t_a ** 2) \
                - 1 / 2 * self.accel * (self.t[i3] - (self.t_a + self.t_c)) ** 2

            self.vel[i1] = self.accel * self.t[i1]
            self.vel[i2] = self.v_max
            self.vel[i3] = self.v_max - (self.t[i3] - (self.t_a + self.t_c)) * self.accel

            self.acc[i1] = self.accel
            self.acc[i2] = 0
            self.acc[i3] = -self.accel


class Cubic(CamProfile):
    def __init__(self, dist, time, num=None, name='Cubic'):
        """A cubic (3rd order polynomial) cam profile object.

        dist -- the displacement distance (movement length)
        time -- the time duration of the displacement, cannot be zero.
        num -- the number of time steps
        """
        CamProfile.__init__(self)
        self.name = name
        self.dist = dist
        self.time = time

        self.v_max = dist / (.5 * time)
        self.accel = 2 * self.v_max / (.5 * time)
        self.v_avg = dist / time
        self.jerk_ = self.v_max / (.25 * time) ** 2

        if num:
            self.__call__(num)

    def __str__(self):
        return self.name

    def __call__(self, num):
        if num is None or self.dist is None or self.time is None or self.v_max is None or self.accel is None:
            return

        if num == 0:
            return

        self.num = num
        self.t = np.arange(num) * self.time / num
        self.pos = np.zeros(num)
        self.vel = np.zeros(num)
        self.acc = np.zeros(num)
        self.jerk = np.zeros(num)

        i1 = np.where(np.logical_and(0 <= self.t, self.t < .25 * self.time))[0]
        i2 = np.where(np.logical_and(.25 * self.time <= self.t, self.t < .5 * self.time))[0]
        i3 = np.where(np.logical_and(.5 * self.time <= self.t, self.t < .75 * self.time))[0]
        i4 = np.where(np.logical_and(.75 * self.time <= self.t, self.t <= self.time))[0]

        t, v, a, j = self.t, self.v_max, self.accel, self.jerk_  # For readability

        self.pos[i1] = 1 / 6 * j * t[i1] ** 3
        self.pos[i2] = 1 / 6 * j * (self.time / 4) ** 3 + v / 2 * (t[i2] - self.time / 4) + .5 * a * (
                t[i2] - self.time / 4) ** 2 - 1 / 6 * j * (t[i2] - self.time / 4) ** 3
        self.pos[i3] = v / 2 * self.time / 4 + a / 2 * (self.time / 4) ** 2 + v * (
                t[i3] - self.time / 2) - 1 / 6 * j * (t[i3] - self.time / 2) ** 3
        self.pos[i4] = 1.5 * v * self.time / 4 + a / 2 * (self.time / 4) ** 2 - j / 6 * (
                self.time / 4) ** 3 + v / 2 * (t[i4] - .75 * self.time) - .5 * a * (
                              t[i4] - .75 * self.time) ** 2 + 1 / 6 * j * (t[i4] - .75 * self.time) ** 3

        self.vel[i1] = .5 * j * t[i1] ** 2
        self.vel[i2] = v / 2 + a * (t[i2] - self.time / 4) - .5 * j * (t[i2] - self.time / 4) ** 2
        self.vel[i3] = v - .5 * j * (t[i3] - self.time / 2) ** 2
        self.vel[i4] = v / 2 - a * (t[i4] - .75 * self.time) + .5 * j * (t[i4] - .75 * self.time) ** 2

        self.acc[i1] = j * t[i1]
        self.acc[i2] = a - j * (t[i2] - self.time / 4)
        self.acc[i3] = -j * (t[i3] - self.time / 2)
        self.acc[i4] = -a + j * (t[i4] - .75 * self.time)

        self.jerk[i1] = j
        self.jerk[i2] = -j
        self.jerk[i3] = -j
        self.jerk[i4] = j


def create_plot_cam_axes(cam, fig, form='separate'):
    """Plots pos, vel, acc and jerk from one or more cam profile objects

    :param cam: one or more cam profile objects
    :param fig: A Figure object
    :param form: 'combine' or 'separate'.
        'combine': plots pos, vel, acc and jerk for each cam profile object
            in one plot with multiple vertical axes
        'separate': plots pos, vel, acc and jerk for each cam profile object
            in separate plots, each with one vertical axes.
    """

    if form == 'combine':
        # A single axes containing all cam curves; pos, vel, accel and jerk.
        from mpl_toolkits.axes_grid1 import host_subplot
        from mpl_toolkits import axisartist

        host = host_subplot(111, axes_class=axisartist.Axes)
        plt.subplots_adjust(right=0.65)

        par1 = host.twinx()
        par2 = host.twinx()
        par3 = host.twinx()

        par2.axis["right"] = par2.new_fixed_axis(loc="right", offset=(55, 0))
        par3.axis["right"] = par3.new_fixed_axis(loc="right", offset=(100, 0))

        par1.axis["right"].toggle(all=True)
        par2.axis["right"].toggle(all=True)
        par3.axis["right"].toggle(all=True)

        p0, = host.plot(cam.t, cam.pos, label="Density")
        p1, = par1.plot(cam.t, cam.vel, label="Temperature")
        p2, = par2.plot(cam.t, cam.acc, label="Velocity")
        p3, = par3.plot(cam.t, cam.jerk)

        # host.set_xlim(0, 2)
        # host.set_ylim(0, 2)
        # par1.set_ylim(0, 4)
        # par2.set_ylim(1, 65)

        host.set_xlabel("time")
        host.set_ylabel("pos")
        par1.set_ylabel("vel")
        par2.set_ylabel("accel")
        par3.set_ylabel("jerk")

        # host.legend()

        host.axis["left"].label.set_color(p0.get_color())
        par1.axis["right"].label.set_color(p1.get_color())
        par2.axis["right"].label.set_color(p2.get_color())
        par3.axis["right"].label.set_color(p3.get_color())

        plt.grid()

        return host

    elif form == 'separate':
        # Multiple axes
        if not isinstance(cam, (list, tuple, np.ndarray)):
            cam = (cam,)

        ax00 = fig.add_subplot(221)
        ax01 = fig.add_subplot(222)
        ax10 = fig.add_subplot(223)
        ax11 = fig.add_subplot(224)

        ax00.set_title('Pos')
        ax01.set_title('Vel')
        ax10.set_title('Acc')
        ax11.set_title('Jerk')

        for c, i in zip(cam, range(len(cam))):
            if c is not None:
                ax00.plot(c.t, c.pos, label=str(i) + ' ' + c.__str__())
                ax00.legend()
                ax01.plot(c.t, c.vel)
                ax10.plot(c.t, c.acc)
                if c.jerk is not None:
                    ax11.plot(c.t, c.jerk)

        return [[ax00, ax01], [ax10, ax11]]


def plot_cam(cam, form='separate'):
    fig = plt.figure()
    ax = create_plot_cam_axes(cam, fig, form=form)
    plt.show()


def _test_quad():
    s1 = Quadratic(dist=5, accel=10000, num=3000)
    s2 = Quadratic(dist=5, accel=10000, v_max=143, num=3000)
    plot_cam((s1, s2))


def _test_quad2():
    s1 = Quadratic(dist=0.09, time=0.1, num=1000)
    s2 = Quadratic(dist=0.09, time=0.1, v_max=1.5, num=1000)
    plot_cam((s1, s2))


def _test_quad3():
    s1 = Quadratic(dist=0.09, time=0.1, num=1000)
    s2 = Quadratic(dist=0.09, time=0.1, accel=38, num=1000)
    s3 = Quadratic(dist=0.09, time=0.1, accel=100, num=1000)
    s4 = Quadratic(dist=0.09, time=0.1, accel=10, num=1000)
    plot_cam((s1, s2, s3, s4))


def _test_cubic():
    s1 = Cubic(dist=0.09, time=0.1)
    s2 = Quadratic(dist=0.09, time=0.1, num=1000)
    s1(1000)

    # plot_cam((s1, s2))
    plot_cam(s1, form='combine')


if __name__ == '__main__':
    # _test_quad()
    # _test_quad2()
    # _test_quad3()
    _test_cubic()

