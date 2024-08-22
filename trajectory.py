from RungeKutta import secondOrderRungeKutta
import pandas as pd
import numpy as np 

g = 9.81
m = 1
alpha = 0.1
n_steps = 40
t0 = 0
t1 = 10
vertical_acceleration = lambda t, y , vy: - g - alpha * vy / m 
horizontal_acceleration = lambda t, x , vx:  - alpha * vx / m 

y, dy = secondOrderRungeKutta(vertical_acceleration, t0, t1, 0, 50, n_steps)
x, dx = secondOrderRungeKutta(horizontal_acceleration, t0, t1, 0, 50, n_steps)

data = pd.DataFrame({
    't': np.linspace(t0, t1, n_steps+1),
    'x':x,
    'y':y,
    'v_x':dx,
    'v_y':dy
    })


print(data)


