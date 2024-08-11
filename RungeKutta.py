import numpy as np
import pandas as pd

def fistOrderRungeKutta(dy, xmin:float, xmax:float, y0:float, n_steps:int) -> np.array:
    """
    dy is a function of 2 variables x and y. dy/dx = dy(x, y)
    Runge Kutta 4 factors method for first order ordinary differential equation
    Euler method for comparison
    returns 2 arrays with all y values for the x range selected for both methods

    dy= lambda x,y: y
    xmin=0
    xmax=5
    y0=1
    n_steps=100
    x = np.linspace(xmin, xmax, n_steps + 1)
    rk, euler = fistOrderRungeKutta(dy, xmin, xmax, y0, n_steps)
    df = pd.DataFrame({'rk':rk, 'euler':euler,'exact':np.exp(x)}, index =np.linspace(xmin, xmax, n_steps + 1))
    """
    x = np.linspace(xmin, xmax, n_steps + 1)
    h = x[1]-x[0]
    y = np.empty(n_steps + 1)
    e = np.empty(n_steps + 1)
    y[0] = y0
    e[0] = y0
    for i in range(n_steps):
        K1 = h * dy(x[i], y[i])
        K2 = h * dy(x[i]+h/2, y[i]+K1/2)
        K3 = h * dy(x[i]+h/2, y[i]+K2/2)
        K4 = h * dy(x[i]+h, y[i]+K3)
        y[i+1] = y[i] + (K1+2*K2+2*K3+K4)/6
        e[i+1] = e[i] + h * dy(x[i], e[i])
    return y,e

def secondOrderRungeKutta(d2y, xmin:float, xmax:float, y0:float, dy0:float, n_steps:int) -> np.array:
    """
    d2y is a function of 3 variables x, y and d^2y/dx^2 = d2y(x, y, u) with u = dy/dx
    Runge Kutta 4 factors method for second order ordinary differential equation
    returns 2 arrays with all y and dy/dx values for the x range selected

    xmin=0
    xmax=5
    y0=1
    n_steps=100
    x = np.linspace(xmin, xmax, n_steps + 1)
    dy0 = 0
    d2y = lambda x,y,u: -y 
    y, u = secondOrderRungeKutta(d2y, xmin, xmax, y0, dy0, n_steps)      
    df = pd.DataFrame({'y':y, 'exact':np.cos(x)}, index = x)
    ddf = pd.DataFrame({'dy_on_dx':u, 'exact':-np.sin(x)}, index = x)
    print(df)
    print(ddf)
    """
    x = np.linspace(xmin, xmax, n_steps + 1)
    h = x[1]-x[0]
    y = np.empty(n_steps + 1)
    u = np.empty(n_steps + 1)  # u =dy
    y[0] = y0
    u[0] = dy0
    dy = lambda x, y, u: u
    for i in range(n_steps):
        K1 = h * dy(x[i], y[i], u[i])
        L1 = h * d2y(x[i], y[i], u[i])
        K2 = h * dy(x[i]+h/2, y[i]+K1/2, u[i]+L1/2)
        L2 = h * d2y(x[i]+h/2, y[i]+K1/2, u[i]+L1/2)
        K3 = h * dy(x[i]+h/2, y[i]+K2/2, u[i]+L2/2)
        L3 = h * d2y(x[i]+h/2, y[i]+K2/2, u[i]+L2/2)
        K4 = h * dy(x[i]+h, y[i]+K3, u[i]+L2)
        L4 = h * d2y(x[i]+h, y[i]+K3, u[i]+L2)
        
        y[i+1] = y[i] + (K1+2*K2+2*K3+K4)/6
        u[i+1] = u[i] + (L1+2*L2+2*L3+L4)/6
    return y,u


