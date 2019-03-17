import matplotlib.pyplot as plt
import numpy as np
import random

import numerics.numerical_integration as num_integ
from numerics.tracker.interpolate_tracker_data import iptrack as trackerData, iptrack_p1100902 as tracker_1, load_p1100902
import numerics.tracker.filter_values as filter_values


def plot_measured_numerical_model():
    # Use the measured data for comparison
    measured_data = load_p1100902() 
    measured_data = filter_values.calculate_track_values(measured_data)
    (measured_time, measured_x, measured_y, measured_v, measured_a) = np.hsplit(measured_data, 5)

    # Load values for the track to use
    (ball_x, ball_y), polynom, track_end_x = tracker_1()
    obstacle_x = 0.59890243
    #print(f"Roots at {np.roots(np.polyder(polynom))}") # Used to find obstacle_x candidates
    print(f"Obstacle at {obstacle_x}")
    
    # Values here are measured during experiment
    ball = num_integ.Ball(mass=0.05, radius=0.0189/2, start_x=ball_x, start_y=ball_y)
    track = num_integ.Track(polynom, start_height=0.3, obstacle_height=0.238, end_height=0.3)
    
    # braking_factor was found with 'find_best_braking_factor.py'
    integrator = num_integ.BallIntegrator(ball, track, braking_factor=0.11097230149082478)
    integrator.timestep = 0.005
    
    steps = 2000
    print(f"Calculating to time {steps*integrator.timestep}s")
    for step in range(steps):
        integrator.step()
    
    passChecker = num_integ.ObstaclePassChecker(obstacle_x=obstacle_x)
    forceCalculator = num_integ.ForceCalculator(track, ball, integrator.time, integrator.position, integrator.velocity, integrator.acceleration)
    frictions, normalForces = forceCalculator.calculate()
    
    measuredForceCalculator = num_integ.ForceCalculator(track, ball, measured_time, measured_x, measured_v, measured_a)
    measured_frictions, measured_normalForces = measuredForceCalculator.calculate()

    # Detect fligh/takeoff, where the ball loses contact with the track.
    # If this happens, all subsequent calculations are wrong.
    flight = None
    for i, n in enumerate(normalForces):
        if n < 0:
            flight = integrator.position[i]
            break
    
    # Check how many times the obstacle was passed.
    # Our experiment demands exactly 2.
    passes = passChecker.detectPasses(integrator.time, integrator.position, integrator.velocity)
    print(f"Found {len(passes)} passes")
    

    # Plot stuff
    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, figsize=(17, 12))
    
    measured_index = min(int(steps*integrator.timestep/0.01), len(measured_time))

    # Plot the track and obstacle position, and mark where flight/lifoff happened (if any)
    x = np.linspace(0, track_end_x, 100)
    ax0.plot(x, np.polyval(polynom, x), label="Track")
    ax0.axvline(obstacle_x, label="Obstacle", color="red")
    ax0.legend()
    if flight:
        ax0.axvline(x=flight, label="Error! Liftoff/flight here")
    
    # Plot the calculated values and show a horizontal line for the obstacle
    ax1.axhline(obstacle_x, label="x_obstacle", color="darkred", linestyle="dotted", alpha=0.8)
    ax1.plot(integrator.time, integrator.position, label="Position", color="red")
    ax1.plot(integrator.time, integrator.velocity, label="Velocity")
    ax1.plot(integrator.time, integrator.acceleration, label="Acceleration")
    ax1.legend()
    
    # Plot the calculated forces
    ax2.plot(integrator.time, frictions, label="Friction")
    ax2.plot(integrator.time, normalForces, label="Normal force")
    ax2.legend()
    
    # Plot comparisons to the "real" values, measured with Tracker
    ax3.plot(measured_data[:,0], measured_data[:,1], label="Measured position", color="black")
    ax3.plot(integrator.time, integrator.position, label="Calulated Position", color="red")
    ax3.legend()

    ax4.plot(integrator.time, normalForces, label="Calculated Normal force")
    ax4.plot(measured_time[:measured_index], measured_normalForces[:measured_index], label="Measured Normal force")
    ax4.legend()

    print("Showing plots")
    figure_path = "out/figure_graphs_all.png"
    fig.savefig(figure_path)
    print("Saved figure of all plots to " + figure_path)
    plt.show(block=True)

if __name__ == '__main__':
  plot_measured_numerical_model()
