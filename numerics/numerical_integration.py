import numpy as np
from numerics.matte4.ode import ode_adaptive, ode_solver
from collections import deque

# All units are SI, eg meters, seconds etc.

GRAVITY = 9.81

class Ball(object):
    def __init__(self, mass, radius, start_x, start_y):
        """
        :param mass: float, weight of the ball
        :param radius: float, radius of the ball
        :param start_x: float, the position of the ball's start.
        :param start_y: float, the position of the ball's height.
        """
        self.mass = mass
        self.radius = radius
        self.start_x = start_x
        self.rotational_inertia = (2/5)*mass*(radius**2)

def makePolynom(height):
    """Makes a polynomial of degree 4.
    The outer edges are at (0,1) and (2,1).
    The center is at x=1, and has the given height.
    :param height: float, the height of the obstacle.
    """
    #TODO x=0 is too hight at y=1, it will jump.
    points = [(0, 1), (0.5, 0), (1, height), (1.5, 0), (2, 1)]
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return np.polyfit(x, y, len(points)-1)

class _PolynomialValues(object):
  def __init__(self, y, dydx, d2ydx2, alpha, R):
    self.y = y
    self.dydx = dydx
    self.d2ydx2 = d2ydx2
    self.alpha = alpha
    self.R = R

class Track(object):
    def __init__(self, polynomial_coefficients, start_height, obstacle_height, end_height):
        """A track for the ball to roll in.

        :param polynomial_coefficients: array, an array of length N+1 with polynomial coefficients
            for a polynomial of degree N. Coefficients are in decending order.
        :param start_height: float, the height of the first curve attachment.
        :param obstacle_height: float, the height of the center attachment forming a obstacle or bump.
        :param end_height: float, the height of the last curve attachment.
        """
        self.polynomial_coefficients = polynomial_coefficients
        self.start_height = start_height
        self.obstacle_height = obstacle_height
        self.end_height = end_height

        self.attachment_count = 8

    def angle_at_position(self, x):
        info = self.track_info(x)
        return info.alpha

    def y(self, x):
        return np.polyval(self.polynomial_coefficients, x)
    
    def track_info(self, x):
        p = self.polynomial_coefficients
        y=np.polyval(p,x)
        dp=np.polyder(p)
        dydx=np.polyval(dp,x)
        ddp=np.polyder(dp)
        d2ydx2=np.polyval(ddp,x)
        alpha=np.arctan(-dydx)
        R=(1.0+dydx**2)**1.5/d2ydx2
        return _PolynomialValues(y, dydx, d2ydx2, alpha, R)

X = 0
Y = 1


class BallIntegrator(object):

    def __init__(self, ball: Ball, track: Track, braking_factor=0.4):
        """
        Coordinate system is horizontal and vertical.
        x = 0 at the first attachment point.
        y = 0 at the lowest curve height.

        :param ball: Ball, the ball to simulate.
        :param track: Track, the track to test the ball in.
        """
        # Initial values
        self.ball = ball
        self.track = track
        self.start_time = 0
        self.gravity = GRAVITY
        self.braking_factor = braking_factor  # FIND THIS EXPERIMENTALLY!

        # Integration settings
        self.current_time = self.start_time
        self.end_time = np.inf # TODO: not needed?
        self.timestep = 0.1  # Defines resolution of integration
        self.initial_integration_stepsize = 0.1
        self.time = [0]
        self.position = [ball.start_x]
        self.velocity = [0]
        self.acceleration = [self._get_parallell_acceleration(self.time[-1], self.velocity[-1])]

    def step(self):
        """Run a step forward with integration
        """
        # NOTE: Use very small step-size for velocity.
        # Look at the ipynb for implementation of this.
        # This code is unfinished and possibly incorrect.
        self.time.append(self.time[-1] + self.timestep)


        # Find velocity from acceleration
        previous_velocity = self.velocity[-1]
        _, v_num = ode_adaptive(self._get_parallell_acceleration, x0=0, xend=self.timestep, y0=previous_velocity, h0=self.initial_integration_stepsize)
        #_, v_num = ode_solver(self._get_parallell_acceleration, x0=0, xend=self.timestep, y0=previous_velocity, h=self.initial_integration_stepsize)
        self.velocity.append(v_num[-1])

        # Find position from velocity
        previous_x = self.position[-1]
        _, x_num = ode_adaptive(self._get_horizontal_velocity, x0=0, xend=self.timestep, y0=previous_x, h0=self.initial_integration_stepsize)
        #_, x_num = ode_solver(self._get_horizontal_velocity, x0=0, xend=self.timestep, y0=previous_x, h=self.initial_integration_stepsize)
        self.position.append(x_num[-1])

        # Store acceleration for the new position
        current_acceleration = self._get_parallell_acceleration(self.time[-1], self.velocity[-1])
        self.acceleration.append(current_acceleration)

    def _get_horizontal_velocity(self, time, position):
        """
        :param time: float, passed by the ODE solver
        :param position: position along the parallell tangent, passed by the ODE solver
        """
        angle = self.track.angle_at_position(position)
        return self.velocity[-1]*np.cos(angle)

    def _get_parallell_acceleration(self, time, velocity):
        """
        :param time: float, passed by the ODE solver
        :param velocity: float, passed by the ODE solver
        :return: float
        """
        last_x = self.position[-1]
        track_angle = self.track.angle_at_position(last_x)

        braking_force = self.braking_factor * velocity
        rolling_force = (5/7)*self.gravity*np.sin(track_angle)
        return rolling_force - braking_force

class ObstaclePassChecker(object):
    def __init__(self, obstacle_x):
        self.obstacle_x = obstacle_x

    def _contains(self, a, b, obstacle):
        return (a <= obstacle <= b) or (b <= obstacle <= a)

    def detectPasses(self, times, x, velocities):
        """Groups velocities in chunks where they have the same sign.
        Then scans the x-values for the chunks with the same timestamp to 
        find passes of the obstacle.
        :param times: array<float>, timestamps. Same size as x and velocities
        :param x: array<float>, positions.
        :param velocities: array<float>, velocities.
        """
        obstacle_x = self.obstacle_x
        to_chunk = deque(enumerate(velocities))  # [(index, velocity)]
        chunks = [[]]
        current_sign = np.sign([vel[1] for vel in to_chunk if vel[1] != 0][0])
        while len(to_chunk) > 0:
            i, v = to_chunk.popleft()
            sign = np.sign(v)
            if sign != current_sign and sign != 0:
                current_sign = sign
                chunks.append([])
            chunks[-1].append((i, v, times[i]))
        #print(f"Got {len(chunks)} chunks where speed is the same direction")

        obstacle_passes = [chunk for chunk in chunks if len(
            chunk) > 2 and self._contains(x[chunk[0][0]], x[chunk[-1][0]], obstacle_x)]
        return obstacle_passes

class ForceCalculator(object):
    def __init__(self, track: Track, ball: Ball, time, x, velocity, acceleration):
        """
        :param track: Track
        :param ball: Ball
        :param time: array<float>, positions
        :param x: array<float>, positions
        :param velocity: array<float>, velocities
        :param acceleration: array<float>, accelerations
        """
        self.track = track
        self.ball = ball
        self.gravity = GRAVITY
        
        if any([len(param) != len(time) for param in (x, velocity, acceleration)]):
            raise Exception("All lists should have same dimensions")
        
        self.time = time
        self.x = x
        self.velocity = velocity
        self.acceleration = acceleration

    def calculate(self):
        """
        :return: tuple, (frictions, normal forces)
        """
        frictions = []
        normalForces = []

        for i, (time, x, vel, accel) in enumerate(zip(self.time, self.x, self.velocity, self.acceleration)):
            info = self.track.track_info(x)
            f = self._calculate_friction(accel)
            N = self._calculate_normal_force(x, vel, info.R, track_angle=info.alpha)
            frictions.append(f)
            normalForces.append(N)

        return (frictions, normalForces)

    def _calculate_friction(self, acceleration):
        """
        :param acceleration: float, the acceleration of the ball
        """
        f = (2/5)*self.ball.mass*acceleration
        return f
    
    def _calculate_normal_force(self, x, velocity, curvature_radius, track_angle):
        """
        :param x: float, the x position of the ball
        :param velocity: float, velocity of the ball
        :param curvature_radius: float, curvature of the track at x
        :param track_angle: float, the angle of the track at x
        """
        sentripental_force = self.ball.mass * (velocity**2) / curvature_radius
        gravitational_force = self.ball.mass * self.gravity*np.cos(track_angle)
        N = sentripental_force + gravitational_force
        return N
