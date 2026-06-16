def delete_spike(prev_cv, current_cv, threshold):
    """
    The purpose of this function is to eliminate the peak noise.
    Therefore, we can consider two cases: 
        1. when tank is not detected (ie, current_data is None), and 
        2. when water is not detected (ie, current_data = 0)
    logic assumes that if two consecutive peak noise (ie, None and 0) is received, the received data is true.
    If not, the function returns the boolean False, that will be used to skip the controller action.
    """
    if threshold <= 0 or threshold >= 0.5:
        raise ValueError("Threshold for \"delete_spike\" function must be positive and small (0,0.5).")
    
    proceed = True

    if current_cv is None:
        proceed = False
    elif current_cv <= threshold:
        if prev_cv <= threshold:
            pass
        else:
            proceed = False

    return proceed