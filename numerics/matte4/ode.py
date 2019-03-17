from numpy import *
from numpy.linalg import norm, solve
from matplotlib.pyplot import *
rcParams['axes.grid'] = True
ion()


## We copied this from Matte 4 curriculum.
## Made by Anne Kværnø, teacher in Matte 4
## 
## Jupyter notebooks - wiki.math.ntnu.no. (2019). Wiki.math.ntnu.no. 
## Retrieved 20 February 2019, from https://wiki.math.ntnu.no/tma4135/2018h/jupyter_notes
##
## Ordinary differential equations	
## http://www.math.ntnu.no/emner/TMA4135/2018h/numerics/python/ode.py

## We only use ode_adaptive with euler (and heun to get the error)

def euler(f, x, y, h):
    # One step of the Euler method
    y_next = y + h*f(x, y)
    x_next = x + h
    return x_next, y_next
# end of euler


def heun(f, x, y, h):
    # One step of Heun's method
    k1 = f(x, y)
    k2 = f(x+h, y+h*k1)
    y_next = y + 0.5*h*(k1+k2)
    x_next = x + h
    return x_next, y_next
# end of heun


def ode_solver(f, x0, xend, y0, h, method=euler):
    # Generic solver for ODEs
    #    y' = f(x,y), y(a)=y0
    # Input: f, the integration interval x0 and xend, 
    #        the stepsize h and the method of choice.  
    #   
    # Output: Arrays with the x- and the corresponding y-values. 
    
    # Initializing:
    y_num = array([y0])    # Array for the solution y 
    x_num = array([x0])    # Array for the x-values

    xn = x0                # Running values for x and y
    yn = y0 

    # Main loop
    while xn < xend - 1.e-10:            # Buffer for truncation errors        
        xn, yn = method(f, xn, yn, h)    # Do one step by the method of choice
        
        # Extend the arrays for x and y
        y_num = concatenate((y_num, array([yn])))
        x_num = append(x_num,xn)
        
    return x_num, y_num
# end of ode_solver

#=============

def heun_euler(f, x, y, h):
    # One step with the pair Heun/Euler
    # Input: the function f, the present state xn and yn  and the stepsize h
    # Output: the solution x and y in the next step, error estimate, and the
    #         order p of Eulers method (the lowest order) 
    
    k1 = f(x, y)
    k2 = f(x+h, y+h*k1)
    y_next = y + 0.5*h*(k1+k2)      # Heuns metode (lokal ekstrapolasjon)
    x_next = x + h
    error_estimate = 0.5*h*norm(k2-k1)   # The 2-norm or the error estimate
    p = 1
    return x_next, y_next, error_estimate, p
# end of heun_euler



def ode_adaptive(f, x0, xend, y0, h0, tol = 1.e-6, method=heun_euler):
    # Adaptive solver for ODEs
    #    y' = f(x,y), y(x0)=y0
    # 
    # Input: the function f, x0, xend, and the initial value y0
    #        intial stepsize h, the tolerance tol, 
    #         and a function (method) implementing one step of a pair.
    # Ut: Array med x- og y- verdier. 
    
    y_num = array([y0])    # Array for the solutions y
    x_num = array([x0])    # Array for the x-values

    xn = x0                # Running values for  x, y and the stepsize h
    yn = y0 
    h = h0
    Maxcall = 100000        # Maximum allowed calls of method
    ncall = 0
    
    # Main loop
    while xn < xend - 1.e-10:               # Buffer for truncation error
        # Adjust the stepsize for the last step
        if xn + h > xend:                   
            h = xend - xn 
        
        # Gjør et steg med valgt metode
        x_try, y_try, error_estimate, p = method(f, xn, yn, h)
        ncall = ncall + 1
        
        if error_estimate <= tol:   
            # Solution accepted, update x and y
            xn = x_try    
            yn = y_try
            # Store the solutions 
            y_num = concatenate((y_num, array([yn])))
            x_num = append(x_num, xn)
        
        # else: The step rejectes and nothing is updated. 
        
        # Adjust the stepsize
        h = 0.8*(tol/error_estimate)**(1/(p+1))*h
        
        # Stop with a warning in the case of max calls to method
        if ncall > Maxcall:
            print('Maximum number of method calls')
            return x_num, y_num

    # Some diagnostic output
    #print('Number of accepted steps = ', len(x_num)-1)
    #print('Number of rejected steps = ', ncall - len(x_num)+1)
    return x_num, y_num
# end of ode_adaptive



def num_ex1():
    # Numerical experiment 1
    
    # The right hand side of the ODE
    def f(x, y):
        return -2*x*y

    # The exact solution, for verification
    def y_exact(x):
        return exp(-x**2)

    x0, xend = 0, 1               # Integration interval
    y0 = 1                        # Initial value for y
    h = 0.1                       # Stepsize

    # Solve the equation
    x_num, y_num = ode_solver(f, x0, xend, y0, h)

    # Plot of the exact solution
    x = linspace(x0, xend, 101)
    plot(x, y_exact(x))

    # Plot of the numerical solution
    plot(x_num, y_num, '.-')

    xlabel('x')
    ylabel('y(x)')
    legend(['Exact', 'Euler']);

    figure(2) # only for the python file
    # Calculate and plot the error in the x-values
    error = y_exact(x_num)-y_num
    plot(x_num, error, '.-')
    xlabel('x')
    ylabel('Error in Eulers metode')
    print('Max error = ', max(abs(error)))  # Print the maximum error
# end of num_ex1


def num_ex2():
    # Numerical example 2, system of equations.

    # The right hand side of the ODE
    # NB! y is an array of dimension 2, and so is dy. 
    def lotka_volterra(x, y):
        alpha, beta, delta, gamma = 2, 1, 0.5, 1     # Set the parameters
        dy = array([alpha*y[0]-beta*y[0]*y[1],       # 
                    delta*y[0]*y[1]-gamma*y[1]])
        return dy

    x0, xend = 0, 20            # Integration interval
    y0 = array([2, 0.5])        # Initital values

    # Solve the equation
    x_lv, y_lv = ode_solver(lotka_volterra, x0, xend, y0, h=0.02) 

    # Plot the solution
    plot(x_lv,y_lv);
    xlabel('x')
    title('Lotka-Volterra equation')
    legend(['y1','y2'],loc=1);
# end of num_ex2


def num_ex3():
    # Numerical example 3

    # Define the ODE
    def van_der_pol(x, y):
        mu = 2
        dy = array([y[1],
                    mu*(1-y[0]**2)*y[1]-y[0] ])
        return dy
    
    # Solve the equation
    x_vdp, y_vdp = ode_solver(van_der_pol, x0=0, xend=20, y0=array([2,0]), h=0.1)

    # Plot the solution
    plot(x_vdp,y_vdp);
    xlabel('x')
    title('Van der Pols ligning')
    legend(['y1','y2'],loc=1);
# end of num_ex3


def num_ex4():
    # Numerical example 4
    def f(x, y):                # The right hand side of the ODE
        return -2*x*y

    def y_exact(x):            # The exact solution
        return exp(-x**2)

    h = 0.1                     # The stepsize
    x0, xend = 0, 1             # Integration interval
    y0 = 1                      # Initial value

    print('h           error\n---------------------')
    
    # Main loop
    for n in range(10):
        x_num, y_num = ode_solver(f, x0, xend, y0, h)   # Solve the equation 
        error = abs(y_exact(xend)-y_num[-1])            # Error at the end point
        print(format('{:.3e}   {:.3e}'.format( h, error)))   
        h = 0.5*h                                       # Reduce the stepsize
# end of num_ex4


def num_ex5():
    # Numerical experiment 5
    
    def f(x, y):            # The right hand side of the ODE
        return -2*x*y

    def y_exact(x):         # The exact solution
        return exp(-x**2)

    h = 0.1                 # The stepsize
    x0, xend = 0, 1         # Integration interval             
    y0 = 1                  # Initial value

    # Solve the equations
    xn_euler, yn_euler = ode_solver(f, x0, xend, y0, h, method=euler)
    xn_heun, yn_heun = ode_solver(f, x0, xend, y0, 2*h, method=heun)     

    # Plot the solution
    x = linspace(x0, xend, 101)
    plot(xn_euler, yn_euler, 'o') 
    plot(xn_heun, yn_heun, 'd')
    plot(x, y_exact(x))
    legend(['Euler','Heun','Exact']);
    xlabel('x')
    ylabel('y');
    
    figure(2) # Only for the python code 5

    # Plot the error of the two methods
    semilogy(xn_euler, abs(y_exact(xn_euler)- yn_euler), 'o');
    semilogy(xn_heun, abs(y_exact(xn_heun)- yn_heun), 'd');
    xlabel('x')
    ylabel('Error')
    legend(['Euler', 'Heun'],loc=3)

    # Print the error as a function of h. 
    print('Error in Euler and Heun\n')
    print('h           Euler       Heun')
    print('---------------------------------')
    for n in range(10):
        x_euler, y_euler = ode_solver(f, x0, xend, y0, h, method=euler)
        x_heun, y_heun = ode_solver(f, x0, xend, y0, 2*h, method=heun)
        error_euler = abs(y_exact(xend)-y_euler[-1])
        error_heun = abs(y_exact(xend)-y_heun[-1])
        print(format('{:.3e}   {:.3e}   {:.3e}'.format( h, error_euler, error_heun)))
        h = 0.5*h

# end of num_ex5

def num_ex6():
    # Numerical example 6

    def lotka_volterra(x, y):       # The Lotka-Volterra equation
        alpha, beta, delta, gamma = 2, 1, 0.5, 1        # Parameters
        dy = array([alpha*y[0]-beta*y[0]*y[1],  
                    delta*y[0]*y[1]-gamma*y[1]])
        return dy

    x0, xend = 0, 20
    y0 = array([2, 0.5])
    h = 0.01

    x_euler, y_euler = ode_solver(lotka_volterra, x0, xend, y0, h, method=euler)
    x_heun, y_heun = ode_solver(lotka_volterra, x0, xend, y0, 2*h, method=heun)

    plot(x_euler,y_euler)
    plot(x_heun, y_heun, '--')
    xlabel('x')
    title('Lotka-Volterra ligningen')
    legend(['y1 (Euler)','y2', 'y1 (Heun)', 'y2'],loc=2)
# end of num_ex6

def num_ex7():
    # Numerical example 7
    def f(x, y):
        return -2*x*y

    def y_exact(x):
        return exp(-x**2)

    h0 = 100
    x0, xend = 0, 1
    y0 = 1

    x_num, y_num = ode_adaptive(f, x0, xend, y0, h0, tol=1.e-3)

    plot(x_num, y_num, '.-', x_num, y_exact(x_num))
    title('Adaptive Heun-Euler')
    xlabel('x')
    ylabel('y')
    legend(['Numerical', 'Exact']);

    # split 71
    figure(2) 

    # Plot the error from the adaptive method
    error = abs(y_exact(x_num) - y_num)
    semilogy(x_num, error, '.-')
    title('Error in Heun-Euler for dy/dt=-2xy')
    xlabel('x');

    # split 72
    figure(3)

    # Plot the step size sequence
    h_n = diff(x_num)            # array with the stepsizes h_n = x_{n+1} 
    x_n = x_num[0:-1]            # array with x_num[n], n=0..N-1
    semilogy(x_n, h_n, '.-')
    xlabel('x')
    ylabel('h')
    title('Stepsize variations');
# end of num_ex7


#---------------------------
# For stiff ODEs
#---------------------------

def implicit_euler(rhs, x, y, h):
    # One step of the implicit Euler's method on the problem 
    #              y' = Ay + g(x)
    # The function rhs should return A and g for each x 
    #     A, gx = rhs(x)
    A, gx = rhs(x+h)
    d = len(gx)                  # The dimension of the system
    M = eye(d)-h*A               # M = I-hA
    b = y + h*gx                 # b = y + hf(x)
    y_next = solve(M, b)         # Solve M y_next = b
    x_next = x+h
    return x_next, y_next
# end of implicit_euler


def trapezoidal_ieuler(rhs, x, y, h):
    # One step with the combination of implicit Euler and the trapezoidal rule
    # for ODEs on the form
    #              y' = Ay + f(x)
    # The function rhs should return A and g for each x:
    #     A, fx = rhs(x)
    A, gx1 = rhs(x+h)
    A, gx0 = rhs(x)
    d = len(gx1)
    
    # One step with implicit Euler
    M = eye(d)-h*A
    b = y + h*gx1
    y_ie = solve(M, b)
    
    # One step with the trapezoidal rule
    M = eye(d)-0.5*h*A
    b = y + 0.5*h*dot(A,y) + 0.5*h*(gx0+gx1)
    y_next = solve(M, b)                       # The solution in the next step
    
    error_estimate = norm(y_next-y_ie)
    x_next = x + h
    p = 1                                       # The order
    return x_next, y_next, error_estimate, p
# end of trapezoidal_ieuler


def num_ex1_s():
    # Numerical example 1s
    # Define the function
    def f(x, y):
        a = 2
        dy = array([-2*y[0]+y[1]+2*sin(x),
                    (a-1)*y[0]-a*y[1]+a*(cos(x)-sin(x))])
        return dy

    # Initial values and integration interval 
    y0 = array([2, 3])
    x0, xend = 0, 10
    h0 = 0.1

    tol = 1.e-2
    # Solve the ODE using different tolerances 
    for n in range(3):
        print('\nTol = {:.1e}'.format(tol)) 
        x_num, y_num = ode_adaptive(f, x0, xend, y0, h0, tol, method=heun_euler)
        
        if n==0:
            # Plot the solution
            subplot(2,1,1)
            plot(x_num, y_num)
            ylabel('y')
            subplot(2,1,2)

        # Plot the step size control
        semilogy(x_num[0:-1], diff(x_num), label='Tol={:.1e}'.format(tol));
        
        tol = 1.e-2*tol         # Reduce the tolerance by a factor 0.01.
    xlabel('x')
    ylabel('h')
    legend(loc='center left', bbox_to_anchor=(1, 0.5));
# end of num_ex1_s

def num_ex2_s():
    # Numerical example 2s
    def f(x, y):
        # y' = f(x,y) = A*y+g(x)
        a = 9
        dy = array([-2*y[0]+y[1]+2*sin(x),
                    (a-1)*y[0]-a*y[1]+a*(cos(x)-sin(x))])
        return dy

    # Startverdier og integrasjonsintervall 
    y0 = array([2, 3])
    x0, xend = 0, 10
    h = 0.19

    x_num, y_num = ode_solver(f, x0, xend, y0, h, method=euler)
    plot(x_num, y_num);
# end of num_ex2_s


def num_ex3_s():
    # Numerical example 3s
    def rhs(x):
        # The right hand side (rhs) of y' = Ay + g(x)
        a = 9
        A = array([[-2, 1],[a-1, -a]])
        gx = array([2*sin(x), a*(cos(x)-sin(x))])
        return A, gx

    # Initial values and integration interval 
    y0 = array([2, 3])
    x0, xend = 0, 10
    h = 0.2             # Initial stepsize

    x_num, y_num = ode_solver(rhs, x0, xend, y0, h, method=implicit_euler)
    plot(x_num, y_num);
# end of num_ex3_s


def num_ex4_s():
    # Numerical example 4s
    def rhs(x):
        # The right hand side of the ODE y' = Ay+g(x)
        a = 9
        A = array([[-2, 1],[a-1, -a]])
        gx = array([2*sin(x), a*(cos(x)-sin(x))])
        return A, gx

    # Initial values and integration interval
    y0 = array([2, 3])
    x0, xend  = 0, 10
    h0 = 0.1                    # Initial stepsize

    tol = 1.e-2                 # Tolerance

    rcParams['figure.figsize'] = 8, 8
    # Solve the equation by different stepsizes. 
    for n in range(3):
        print('\nTol = {:.1e}'.format(tol)) 
        x_num, y_num = ode_adaptive(rhs, x0, xend, y0, h0, tol, method=trapezoidal_ieuler)
        
        if n==0:
            # Plot the solution
            subplot(2,1,1)
            plot(x_num, y_num)
            ylabel('y')
            subplot(2,1,2)

        # Plot the step size sequence
        semilogy(x_num[0:-1], diff(x_num), label='Tol={:.1e}'.format(tol));
        
        tol = 1.e-2*tol         # Reduce the tolerance by a factor 1/100

    # Decorations
    xlabel('x')
    ylabel('h')
    legend(loc='center left', bbox_to_anchor=(1, 0.5));
# end of num_ex4_s

