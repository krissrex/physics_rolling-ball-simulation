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
    exponents = [len(polynom)-i-1 for i in range(len(polynom))]
    pstr = ("x^{{{}}} + ".join([str(round(x, 3)) for x in polynom])).format(*exponents)
    pstr = pstr.replace("+ -", "- ")
    print("Polynom: " + pstr)
    
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
    
    figure_size = (8, 7)
    plt.rc('axes', titlesize=32, labelsize=32)

    # Error calculations
    delta_mass = 0.0001
    delta_gravity = 0.004
    def plotError():
      acc = np.array(integrator.acceleration)
      v = np.array(integrator.velocity)
      g = 9.81
      m = 0.05
      def R_alpha(x):
        info = integrator.track.track_info(x)
        return info.R, info.alpha
      r_a = np.array([R_alpha(x) for x in integrator.position])
      r = r_a[:,0]
      alpha = r_a[:,1]

      friction_error = np.abs(acc)*(2/5)*delta_mass
      normalForce_error = np.sqrt(np.square(\
            ((v*v)/(r) + g*np.cos(alpha))*delta_mass
          )\
        + np.square(\
            m*np.cos(alpha)*delta_gravity\
          ))
          
      fig, ax0 = plt.subplots(1, figsize=figure_size)
      ax0.set_xlim(0, 6)
      ax0.plot(integrator.time, friction_error, label="Delta f")
      ax0.plot(integrator.time, normalForce_error, label="Delta N")
      ax0.set_xlabel("Time [s]")
      ax0.set_ylabel("Error [N]")
      ax0.legend()
      figure_path = "out/figure_graphs_simulated_delta_f_N.png"
      fig.savefig(figure_path, dpi=96, bbox_inches="tight")
    
    plotError()

    # Plot stuff
    fig, (ax0) = plt.subplots(1, figsize=(8, 5))
    
    measured_index = min(int(steps*integrator.timestep/0.01), len(measured_time))

    # Plot the track and obstacle position, and mark where flight/lifoff happened (if any)
    x = np.linspace(0, track_end_x, 100)
    ax0.set_xlim(0, track_end_x)
    ax0.plot(x, np.polyval(polynom, x), label="Track")
    ax0.set_xlabel("x [m]")
    ax0.set_ylabel("y [m]")
    ax0.axvline(obstacle_x, label="Obstacle x=" + str(round(obstacle_x, 5)), color="red")
    ax0.legend()
    if flight:
        ax0.axvline(x=flight, label="Error! Liftoff/flight here")
    figure_path = "out/figure_graphs_simulated_track.png"
    fig.savefig(figure_path, dpi=96, bbox_inches="tight")

    # Plot the calculated values and show a horizontal line for the obstacle
    fig, (ax1) = plt.subplots(1, figsize=figure_size)
    ax1.axhline(obstacle_x, label="Obstacle", color="darkred", linestyle="dotted", alpha=0.8)
    ax1.plot(integrator.time, integrator.position, label="x position", color="red")
    ax1.set_xlim(0, 6)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("x [m]")
    ax1.legend(loc="lower right")
    figure_path = "out/figure_graphs_simulated_x.png"
    fig.savefig(figure_path, dpi=96, bbox_inches="tight")
    
    fig, (ax1) = plt.subplots(1, figsize=figure_size)
    ax1.plot(integrator.time, integrator.velocity, label="Velocity")
    ax1.set_xlim(0, 6)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Parallell velocity [m/s]")
    ax1.legend(loc="lower right")
    figure_path = "out/figure_graphs_simulated_v.png"
    fig.savefig(figure_path, dpi=96, bbox_inches="tight")

    fig, (ax1) = plt.subplots(1, figsize=figure_size)
    ax1.plot(integrator.time, integrator.acceleration, label="Acceleration")
    ax1.set_xlim(0, 6)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Parallell acceleration [m/s^2]")
    ax1.legend(loc="lower right")
    figure_path = "out/figure_graphs_simulated_a.png"
    fig.savefig(figure_path, dpi=96, bbox_inches="tight")

    # Plot the calculated forces
    fig, (ax2) = plt.subplots(1, figsize=figure_size)
    ax2.set_xlim(0, 6)
    ax2.plot(integrator.time, np.abs(frictions), label="Friction")
    ax2.plot(integrator.time, normalForces, label="Normal force")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Force [N]")
    ax2.legend()
    figure_path = "out/figure_graphs_simulated_force.png"
    fig.savefig(figure_path, dpi=96, bbox_inches="tight")

    # Plot comparisons to the "real" values, measured with Tracker
    fig, (ax3) = plt.subplots(1, figsize=(12, 6))
    ax3.set_xlim(0, 6)
    ax3.axhline(obstacle_x, label="Obstacle", color="darkred", linestyle="dotted", alpha=0.8)
    ax3.plot(measured_data[:,0], measured_data[:,1], label="Measured position", color="black")
    ax3.plot(integrator.time, integrator.position, label="Calulated Position", color="red")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("x [m]")
    ax3.legend()

    figure_path = "out/figure_graphs_comparison_x.png"
    fig.savefig(figure_path, dpi=96, bbox_inches="tight")


    fig, (ax4) = plt.subplots(1, figsize=(6, 6))
    ax4.set_xlim(0, 6)    
    ax4.plot(integrator.time, normalForces, label="Calculated Normal force")
    ax4.plot(measured_time[:measured_index], measured_normalForces[:measured_index], label="Measured Normal force")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Force [N]")
    ax4.legend()
    figure_path = "out/figure_graphs_comparison_n.png"
    fig.savefig(figure_path, dpi=96, bbox_inches="tight")


    fig, (ax5) = plt.subplots(1, figsize=(6, 6))
    ax5.set_xlim(0, 6)
    ax5.plot(integrator.time, integrator.velocity, label="Calculated velocity")
    ax5.plot(measured_time, measured_v, label="Measured velocity")
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Track parallell velocity [m/s]")
    ax5.legend()
    figure_path = "out/figure_graphs_comparison_v.png"
    fig.savefig(figure_path, dpi=96, bbox_inches="tight")

if __name__ == '__main__':
  plot_measured_numerical_model()
