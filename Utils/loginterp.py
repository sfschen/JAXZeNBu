import numpy as np


def loginterp(x, y, yint = None, side = "both",  lp = 1, rp = -2):
    '''
    Extrapolate function by evaluating a log-index of left & right side.
    
    From Chirag Modi's CLEFT code at
    https://github.com/modichirag/CLEFT/blob/master/qfuncpool.py
    
    The warning for divergent power laws on both ends is turned off. To turn back on uncomment lines 26-33.
    '''
    
    if side == "both":
        side = "lr"
    
    # Make sure there is no zero crossing between the edge points
    # If so assume there can't be another crossing nearby
    
    if np.sign(y[lp]) == np.sign(y[lp-1]) and np.sign(y[lp]) == np.sign(y[lp+1]):
        l = lp
    else:
        l = lp + 2
        
    if np.sign(y[rp]) == np.sign(y[rp-1]) and np.sign(y[rp]) == np.sign(y[rp+1]):
        r = rp
    else:
        r = rp - 2
    
    lneff = np.gradient(y,x)[l]*x[l]/y[l]
    rneff = np.gradient(y,x)[r]*x[r]/y[r]
    #lneff = derivative(yint, x[l], dx = x[l]*ldx, order = lorder)*x[l]/y[l]
    #rneff = derivative(yint, x[r], dx = x[r]*rdx, order = rorder)*x[r]/y[r]

    yint2 = lambda xx:   (xx <= x[l]) * y[l]* np.nan_to_num((xx/x[l])**lneff) \
                   + (xx >= x[r]) * y[r]* np.nan_to_num((xx/x[r])**rneff) \
                   + (xx > x[l]) * (xx < x[r]) * np.interp(xx,x,y)

    return yint2
