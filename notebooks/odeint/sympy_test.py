import numpy as np
import sympy as sp

x, y, z, t = sp.symbols('x y z t')

eqs = (x*y + 3*y + np.sqrt(3), x + 4 + y)

sp.solve((x**2 - y**2 - t), x, y)


