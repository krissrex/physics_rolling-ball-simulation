# trvalues - track values
#
# SYNTAX
# [y,dydx,d2ydx2,alpha,R]=trvalues(p,x)
#
# INPUT
# p: the n+1 coefficients of a polynomial of degree n, given in descending
# order. (For instance the output from p=iptrack(filename).)
# x: ordinate value at which the polynomial is evaluated.
#
# OUTPUT
# [y,dydx,d2ydx2,alpha,R]=trvalues(p,x) returns the value y of the
# polynomial at x, the derivative dydx and the second derivative d2ydx2 in
# that point, as well as the slope alpha(x) and the radius of the
# osculating circle. 
# The slope angle alpha is positive for a curve with a negative derivative. 
# The sign of the radius of the osculating circle is the same as that of 
# the second derivative.

import numpy as np

class PolynomialValues(object):
  def __init__(self, y, dydx, d2ydx2, alpha, R):
    self.y = y
    self.dydx = dydx
    self.d2ydx2 = d2ydx2
    self.alpha = alpha
    self.R = R

def y(p, x):
  return np.polyval(p, x)

def calculate(p,x):
  """
  :param p: polynomial function.
  :param x: float, x-value for polynomial.
  :return: PolynomialValues, the values of the polynomial at point x.
  """
  y=np.polyval(p,x)
  dp=np.polyder(p)
  dydx=np.polyval(dp,x)
  ddp=np.polyder(dp)
  d2ydx2=np.polyval(ddp,x)
  alpha=np.arctan(-dydx)
  R=(1.0+dydx**2)**1.5/d2ydx2
  return PolynomialValues(y, dydx, d2ydx2, alpha, R)
