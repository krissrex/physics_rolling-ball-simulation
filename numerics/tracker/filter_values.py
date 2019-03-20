import numpy as np
from scipy.signal import savgol_filter

def calculate_track_values(data: np.ndarray) -> np.ndarray:
  """Calculate velocity and acceleration. 
  Assumes 100fps data with 0.01s between each point
  :param data: ndarray, [(t, x, y)]
  :return: ndarray, [(t, x, y, v, a)]
  """
  velocities = [0]
  delta_time = 0.01 #100fps, avoid floating point errors
  for i, (t, x, y) in enumerate(data[1:-1]):
    i += 1
    (prev_x, prev_y) = data[i-1][1:]
    (next_x, next_y) = data[i+1][1:]
    distance = np.sign(next_x-prev_x)*np.sqrt((next_x-prev_x)**2 + (next_y - prev_y)**2)
    #delta_time = t - data[i-1][0]
    velocity = distance/(2*delta_time)
    velocities.append(velocity)
  velocities.append((velocities[-1] - velocities[-2])/delta_time)
  #velocities = smooth_data(velocities)

  accelerations = [0]
  for i, velocity in enumerate(velocities[1:]):
    i += 1
    #delta_time = t - data[i-1][0]
    acceleration = (velocity - velocities[i-1]) / delta_time
    accelerations.append(acceleration)

  # np.c_[M, c] adds a column c to the right of a matrix M
  return np.c_[data, velocities, accelerations]

def smooth_data(data_points: list) -> list:
  window_size = 17
  polynomial_degree = 3
  smoothed = savgol_filter(data_points, window_size, polynomial_degree)
  return smoothed
