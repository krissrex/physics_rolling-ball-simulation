# iptrack - interpolate track
#
# SYNTAX
# p=iptrack(filename)
#
# INPUT
# filename: data file containing exported tracking data on the standard
# Tracker export format
#
# mass_A
# t	x	y
# 0.0	-1.0686477620876644	42.80071293284619
# 0.04	-0.714777136706708	42.62727536827738
# ...
#
# OUTPUT
# p=iptrack(filename) returns the coefficients of a polynomial of degree 15
# that is the least square fit to the data y(x). Coefficients are given in
# descending powers.

import numpy as np
import matplotlib.pyplot as plt

def iptrack(filename):
  data = np.loadtxt(filename, skiprows=2, delimiter=";")
  return np.polyfit(data[:,1],data[:,2],15)

def find_track_end(data):
  for i, (t,x,y) in enumerate(data):
   if data[i+1][1] < x:
     return i

def load_p1100902() -> np.ndarray:
  """Loads the raw track data for recording p1100902.
  Use iptrack_p1100902() for the polynomial.
  
  Use `numerics.tracker.filter_values.calculate_track_values(data)` to get velocity
  and acceleration. The raw data is a bit inaccurate and noisy.

  :return: np.ndarray, (time, x, y)
  """
  data = np.loadtxt("test_data/P1100902-data.csv", skiprows=2, delimiter=";")
  data = data[33:]
  t_0 = data[0][0]
  for row in data:
    row[0] = row[0] - t_0
  return data

def iptrack_p1100902():
  """
  Returns track data for recording p1100902 that ends after 
  after the ball has traveled to its max(x).
  :return: ((float, float), [float], float), (ball_x, ball_y), polynomial, track_end
  """
  data = load_p1100902()
  ball_start_x_y = (data[0][1], data[0][2])
  point = find_track_end(data)
  data = data[:point]
  polynom = np.polyfit(data[:,1], data[:,2], 15)

  """
  plt.plot(data[:,1], data[:,2], linewidth=2, label="Test")
  plt.legend()

  x = np.linspace(0, 1.13, 100)
  plt.plot(x, np.polyval(polynom, x), label="Polynom")
  plt.legend()

  plt.show() """

  return ball_start_x_y, polynom, data[-1][1]


if __name__ == '__main__':
  iptrack_p1100902()