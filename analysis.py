import numpy as np
from scipy.integrate import cumulative_trapezoid

def analysis(error, time):
    error = np.asarray(error, dtype=float)
    time  = np.asarray(time,  dtype=float)

    ie = cumulative_trapezoid(error, x=time)

    abs_error = abs(error)
    iae = cumulative_trapezoid(abs_error, x=time)

    sqrd_error = error**2
    ise = cumulative_trapezoid(sqrd_error, x=time)

    time_abs_error = abs_error*time
    itae = cumulative_trapezoid(time_abs_error, x=time)

    return ie, iae, ise, itae