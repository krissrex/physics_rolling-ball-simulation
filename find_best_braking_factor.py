import matplotlib.pyplot as plt
import numpy as np
import numerics.numerical_integration as num_integ
import random

## For animations
from matplotlib import animation, rc
from IPython.display import HTML
##

from numerics.tracker.interpolate_tracker_data import iptrack as trackerData, iptrack_p1100902 as tracker_1, load_p1100902
import numerics.tracker.filter_values as filter_values


def calculate_measured_values(obstacle_x = 0.59890243) -> tuple:
    data = load_p1100902()
    velocities = [0]
    for i, (t, x, y) in enumerate(data[1:]):
        (prev_x, prev_y) = data[i-1][1:]
        distance = (x-prev_x)
        delta_time = t - data[i-1][0]
        velocities.append(distance/delta_time)
    
    t = data[:,0]
    x = data[:,1]
    
    pass_detecter = num_integ.ObstaclePassChecker(obstacle_x)
    passes = pass_detecter.detectPasses(t, x, velocities)

    return t, x, velocities

def simulate_with_braking(braking_factor=0.1119893, obstacle_x = 0.59890243, steps = 600, timestep=0.016):
    """
    :return: False|int|tuple, int if passes are invalid
    """
    (ball_x, ball_y), polynom, track_end_x = tracker_1()
    
    ball = num_integ.Ball(mass=0.05, radius=0.0189/2, start_x=ball_x, start_y=ball_y)
    track = num_integ.Track(polynom, start_height=0.3, obstacle_height=0.238, end_height=0.3)
    
    integrator = num_integ.BallIntegrator(ball, track, braking_factor=braking_factor)
    integrator.timestep = timestep
    
    for step in range(steps):
        integrator.step()
    
    forceCalculator = num_integ.ForceCalculator(track, ball, integrator.time, integrator.position, integrator.velocity, integrator.acceleration)
    passChecker = num_integ.ObstaclePassChecker(obstacle_x)
    pass_detecter = num_integ.ObstaclePassChecker(obstacle_x)
    passes = pass_detecter.detectPasses(integrator.time, integrator.position, integrator.velocity)

    if len(passes) != 2:
        print(f"Invalid braking factor {braking_factor}, {len(passes)} passes")
        return len(passes)
    
    frictions, normalForces = forceCalculator.calculate()    
    for i, n in enumerate(normalForces):
        if n < 0:
            print(f"Invalid setup. Ball flies at force index {i}")
            return False
    
    return (integrator.time, integrator.position, integrator.velocity)

def find_error(measured_t: list, measured_x: list, calc_t: list, calc_x: list) -> float:
    """
    :return: float, the error
    """
    # 1: interpolate to match t values
    # 2: diff x values at interpolated values
    if len(measured_x) != len(calc_x):
        raise Exception(f"Must be same size! {len(measured_x)} vs {len(calc_x)}")
    return np.square(np.sum(np.square(calc_x - measured_x))/len(measured_x))
    

def find_close_match():
    (meas_t, meas_x, meas_v) = calculate_measured_values()
    braking_factor = 0.11097230149082478
    low_factor = 0.095
    high_factor = 0.4
    
    last_error = np.inf
    smallest_error = np.inf
    best_factor = braking_factor
    
    compare_size = 800
    frame_rate_seconds=0.01
    
    errors = []

    """
    for run_number in range(40):
        print(f"Run {run_number}. Factor {braking_factor}" + "-"*20)
        simulated = simulate_with_braking(braking_factor=braking_factor, steps=compare_size-1, timestep=frame_rate_seconds)
        if isinstance(simulated, bool) and simulated == False:
            # Flight
            low_factor = braking_factor
            braking_factor = (high_factor + low_factor) / 2
        elif isinstance(simulated, int):
            # Invalid pass number
            passes = simulated
            print(f"Found {passes} passes with factor {braking_factor}. Range ({low_factor}, {high_factor})")
            if passes < 2:
                high_factor = braking_factor
            elif passes > 2:
                low_factor = braking_factor
            braking_factor = (high_factor + low_factor)/2
            print(f"Adjusted factors: {braking_factor} in range ({low_factor}, {high_factor})")
        else:
            # Normal case
            (calc_t, calc_x, calc_v) = simulated
            error = find_error(meas_t[:compare_size], meas_x[:compare_size], calc_t, calc_x)
            errors.append((braking_factor, error))
            
            if error < smallest_error:
                # New best case
                smallest_error = error
                best_factor = braking_factor
                print(f"Found better factor: {braking_factor} with error {error}")
            
            # Assuming error is in a valley, from previous graphed runs
            # Some sort of gradient descent with annealing
            if error > last_error:
                # Error too far from best case, go back towards best
                braking_factor = (best_factor + braking_factor) / 2
            else:
                if random.randint(0, 10) <= 2:
                    # Randomly try another value, annealing
                    braking_factor = random.uniform(low_factor, high_factor)
                else:
                    # Look around best case
                    braking_factor = random.uniform(0.8, 1.2)*best_factor
            last_error = error
    """

    # Local search step sizes
    microsteps = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]
    microstep_index = 0
    braking_factor = best_factor

    simulated = simulate_with_braking(braking_factor=braking_factor, steps=compare_size-1, timestep=frame_rate_seconds)
    (calc_t, calc_x, calc_v) = simulated    
    last_error = find_error(meas_t[:compare_size], meas_x[:compare_size], calc_t, calc_x)
    error = last_error
    
    run_number = 0
    print(f"Starting iterations with {braking_factor}:")
    while microstep_index < len(microsteps):
        run_number += 1
        last_error = error
        braking_factor += microsteps[microstep_index]
        print(f"Iterative run {run_number}. Factor {braking_factor}. Microstep {microsteps[microstep_index]} " + "-"*20)
        simulated = simulate_with_braking(braking_factor=braking_factor, steps=compare_size-1, timestep=frame_rate_seconds)
        if isinstance(simulated, bool) and simulated == False:
            # Flight
            low_factor = braking_factor
            braking_factor = (high_factor + low_factor) / 2
            braking_factor -= microsteps[microstep_index]
            microstep_index += 1
        elif isinstance(simulated, int):
            # Invalid pass number
            passes = simulated
            print(f"Found {passes} passes with factor {braking_factor}. Range ({low_factor}, {high_factor})")
            braking_factor -= microsteps[microstep_index]
            microstep_index += 1
        else:
            # Normal case
            (calc_t, calc_x, calc_v) = simulated
            error = find_error(meas_t[:compare_size], meas_x[:compare_size], calc_t, calc_x)
            errors.append((braking_factor, error))
            if error < smallest_error:
                print(f"\tFound better factor: {braking_factor} with error {error}. Microstep {microsteps[microstep_index]} ({microstep_index}/{len(microsteps)}")
                best_factor = braking_factor
                smallest_error = error
            if error > last_error:
                braking_factor -= microsteps[microstep_index]
                microstep_index += 1

    errors = np.array(sorted(errors, key=lambda err: err[0]))
    plt.plot(errors[:,0], errors[:,1], label="Error")
    plt.legend()
    plt.show()
    
    print("-"*100)
    print(f"Best value found: {best_factor} (error={smallest_error})")
    # 0.11097230149082478 (error=4.3536731384222186e-07)
    return best_factor

if __name__ == '__main__':
    find_close_match()