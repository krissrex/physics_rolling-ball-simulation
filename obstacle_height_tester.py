import matplotlib.pyplot as plt
import numpy as np
import random

import numerics.numerical_integration as num_integ
from numerics.tracker.interpolate_tracker_data import iptrack as trackerData, iptrack_p1100902 as tracker_1, load_p1100902
import numerics.tracker.filter_values as filter_values

def makePolynom(obstacle_height: float):
    """Make a polynomial for numpy starting at 
    :return: tuple, (list, float), (polynomial coefficients, obstacle_x)
    """
    # A challenge is that the polynomials will stretch down below 0 differently.
    # An alternative algorithm would be a 3-part polynomial,
    # where the start and end are the same,
    # joined by a 2nd degree parabola in the middle, interpolating 3 points (left, middle, right).
    # Could do this with a function, generating many values, and using numpy to interpolate.
    obstacle_x = 0.5
    end_height = 0.3
    track_length = 1.13
    points = [(0, end_height), (0.4, 0.05), (obstacle_x, obstacle_height), (0.8, 0.05), (track_length, end_height)]
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return np.polyfit(x, y, len(points)-1), obstacle_x

def find_suitable_heights():
    # Use the measured data for comparison
    measured_data = load_p1100902() 
    measured_data = filter_values.calculate_track_values(measured_data)
    (measured_time, measured_x, measured_y, measured_v, measured_a) = np.hsplit(measured_data, 5)

    # Load values for the track to use
    (ball_x, ball_y), polynom, track_end_x = tracker_1()
    current_obstacle_height = 0.1
    height_step_size = 0.005
    polynom, obstacle_x = makePolynom(current_obstacle_height)
    
    #print(f"Roots at {np.roots(np.polyder(polynom))}") # Used to find obstacle_x candidates
    print(f"Obstacle at {obstacle_x}")
    
    # Values here are measured during experiment
    ball = num_integ.Ball(mass=0.05, radius=0.0189/2, start_x=ball_x, start_y=ball_y)
    
    # braking_factor was found with 'find_best_braking_factor.py'
    timestep = 0.01
    steps = 1000
    print(f"Calculating to time {steps*timestep}s")

    # Array of (height, passes)
    passes_list = []

    too_high = False
    while not too_high:
        print(f"Calculating for height {current_obstacle_height}")
        # Set up new round
        track = num_integ.Track(polynom, start_height=0.3, obstacle_height=current_obstacle_height, end_height=0.3)
        integrator = num_integ.BallIntegrator(ball, track, braking_factor=0.11097230149082478)
        integrator.timestep = timestep

        # Calculate
        for step in range(steps):
            integrator.step()
        
        passChecker = num_integ.ObstaclePassChecker(obstacle_x=obstacle_x)
        forceCalculator = num_integ.ForceCalculator(track, ball, integrator.time, integrator.position, integrator.velocity, integrator.acceleration)
        frictions, normalForces = forceCalculator.calculate()
        
        # Detect fligh/takeoff, where the ball loses contact with the track.
        # If this happens, all subsequent calculations are wrong.
        flight = None
        for i, n in enumerate(normalForces):
            if n < 0:
                flight = integrator.position[i]
                break
        
        # Check how many times the obstacle was passed.
        # Our experiment demands exactly 2.
        if flight == None:
            passes = len(passChecker.detectPasses(integrator.time, integrator.position, integrator.velocity))
            print(f"Found {passes} passes with height {current_obstacle_height}")
            passes_list.append((current_obstacle_height, passes))
            if passes == 0:
                too_high = True
        
        current_obstacle_height += height_step_size
        polynom, obstacle_x = makePolynom(current_obstacle_height)

    passes_list = np.array(passes_list)
    plt.plot(passes_list[:,0], passes_list[:,1], label="Passes per height")
    plt.xlabel("height [m]")
    plt.ylabel("passes")
    plt.legend()
    plt.show(block=True)
    

if __name__ == '__main__':
  find_suitable_heights()
