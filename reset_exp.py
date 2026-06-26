import time
from AILevelingRL.PyArduino import PyArduino

UPPER = 5
LOWER = 6
DISTURBANCE_PUMP = 7

pa = PyArduino("minima")

if __name__ == "__main__":
    pa.run_digital_write(UPPER, True)
    pa.run_digital_write(LOWER, True)
    pa.run_digital_write(DISTURBANCE_PUMP, False)
    pa.run_pump_speed(0.0)