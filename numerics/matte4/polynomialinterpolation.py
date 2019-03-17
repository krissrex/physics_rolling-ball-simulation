from numpy import *
from matplotlib.pyplot import *
ion()
clf()

def cardinal(xdata, x):
    """
    cardinal(xdata, x): 
    In: xdata, array with the nodes x_i.
        x, array or a scalar of values in which the cardinal functions are evaluated.
    Return: l: a list of arrays of the cardinal functions evaluated in x. 
    """
    n = len(xdata)              # Number of evaluation points x
    l = []
    for i in range(n):          # Loop over the cardinal functions
        li = ones(len(x))
        for j in range(n):      # Loop to make the product for l_i
            if i is not j:
                li = li*(x-xdata[j])/(xdata[i]-xdata[j])
        l.append(li)            # Append the array to the list            
    return l

def lagrange(ydata, l):
    """
    lagrange(ydata, l):
    In: ydata, array of the y-values of the interpolation points.
         l, a list of the cardinal functions, given by cardinal(xdata, x)
    Return: An array with the interpolation polynomial. 
    """
    poly = 0                        
    for i in range(len(ydata)):
        poly = poly + ydata[i]*l[i]  
    return poly
# end of lagrange

def divdiff(xdata,ydata):
    # Create the table of divided differences based
    # on the data in the arrays x_data and y_data. 
    n = len(xdata)
    F = zeros((n,n))
    F[:,0] = ydata             # Array for the divided differences
    for j in range(n):
        for i in range(n-j-1):
            F[i,j+1] = (F[i+1,j]-F[i,j])/(xdata[i+j+1]-xdata[i])
    return F                    # Return all of F for inspection. 
                                # Only the first row is necessary for the
                                # polynomial.

def newton_interpolation(F, xdata, x):
    # The Newton interpolation polynomial evaluated in x. 
    n, m = shape(F)
    xpoly = ones(len(x))               # (x-x[0])(x-x[1])...
    newton_poly = F[0,0]*ones(len(x))  # The Newton polynomial
    for j in range(n-1):
        xpoly = xpoly*(x-xdata[j])
        newton_poly = newton_poly + F[0,j+1]*xpoly
    return newton_poly
# end of newton_interpolation


def equidistributed_bound(n, M, a, b):
    # Return the bound for error in equidistributed nodes
    print(n)
    h = (b-a)/n
    return 0.25*h**(n+1)/(n+1)*M;
# end of equidistributed_bound

def omega(xdata, x):
    # compute omega(x) for the nodes in xdata
    n1 = len(xdata)
    omega_value = ones(len(x))             
    for j in range(n1):
        omega_value = omega_value*(x-xdata[j])  # (x-x_0)(x-x_1)...(x-x_n)
    return omega_value
# end of omega

def plot_omega():
    # Plot omega(x) 
    n = 8                           # Number of interpolation points is n+1
    a, b = -1, 1                    # The interval
    x = linspace(a, b, 501)        
    xdata = linspace(a, b, n) 
    plot(x, omega(xdata, x))
    grid(True)
    xlabel('x')
    ylabel('omega(x)')
    print("n = {:2d}, max|omega(x)| = {:.2e}".format(n, max(abs(omega(xdata, x)))))
# end of plot_omega
 
def chebyshev_nodes(a, b, n):
    # n Chebyshev nodes in the interval [a, b] 
    i = array(range(n))                 # i = [0,1,2,3, ....n-1]
    x = cos((2*i+1)*pi/(2*(n)))         # nodes over the interval [-1,1]
    return 0.5*(b-a)*x+0.5*(b+a)        # nodes over the interval [a,b]
# end of chebyshev_nodes

def example3():
    # Example 3
    xdata = [0, 1, 3]           # The interpolation points
    ydata = [3, 8, 6]
    x = linspace(0, 3, 101)     # The x-values in which the polynomial is evaluated
    l = cardinal(xdata, x)      # Find the cardinal functions evaluated in x
    p = lagrange(ydata, l)      # Compute the polynomial evaluated in x
    plot(x, p)                  # Plot the polynomial
    plot(xdata, ydata, 'o')     # Plot the interpolation points 
    title('The interpolation polynomial p(x)')
    xlabel('x');
# end of example 3

def example4():
    # Example 4

    # Define the function
    def f(x):
        return sin(x)

    # Set the interval 
    a, b = 0, 2*pi                  # The interpolation interval
    x = linspace(a, b, 101)         # The 'x-axis' 

    # Set the interpolation points
    n = 8                           # Interpolation points
    xdata = linspace(a, b, n+1)     # Equidistributed nodes (can be changed)
    ydata = f(xdata)                

    # Evaluate the interpolation polynomial in the x-values
    l = cardinal(xdata, x)  
    p = lagrange(ydata, l)

    # Plot f(x) og p(x) and the interpolation points
    subplot(2,1,1)                  
    plot(x, f(x), x, p, xdata, ydata, 'o')
    legend(['f(x)','p(x)'])
    grid(True)

    # Plot the interpolation error
    subplot(2,1,2)
    plot(x, (f(x)-p))
    xlabel('x')
    ylabel('Error: f(x)-p(x)')
    grid(True)
    print("Max error is {:.2e}".format(max(abs(p-f(x)))))
# end of example 4

def example_divided_differences():
    # Example: Use of divided differences and the Newton interpolation
    # formula. 
    xdata = [0, 2/3, 1]
    ydata = [1, 1/2, 0]
    F = divdiff(xdata, ydata)      # The table of divided differences
    print('The table of divided differences:\n',F)

    x = linspace(0, 1, 101)     # The x-values in which the polynomial is evaluated
    p = newton_interpolation(F, xdata, x)
    plot(x, p)                  # Plot the polynomial
    plot(xdata, ydata, 'o')     # Plot the interpolation points 
    title('The interpolation polynomial p(x)')
    grid(True)
    xlabel('x');
# end of example_divided_differences








