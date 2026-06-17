import time
from AILevelingRL.PyArduino import PyArduino

UPPER = 5
LOWER = 6

pa = PyArduino("minima")

if __name__ == "__main__":
    pa.run_digital_write(UPPER, True)
    pa.run_digital_write(LOWER, True)
    pa.run_pump_speed(0.0)

    time.sleep(120.0)