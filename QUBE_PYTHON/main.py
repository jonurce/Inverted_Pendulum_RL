# ------------------------------------- AVAILABLE FUNCTIONS --------------------------------#
# qube.setRGB(r, g, b) - Sets the LED color of the QUBE. Color values range from [0, 999].
# qube.setMotorSpeed(speed) - Sets the motor speed. Speed ranges from [-999, 999].
# qube.setMotorVoltage(volts) - Applies the given voltage to the motor. Volts range from (-24, 24).
# qube.resetMotorEncoder() - Resets the motor encoder in the current position.
# qube.resetPendulumEncoder() - Resets the pendulum encoder in the current position.

# qube.getMotorAngle() - Returns the cumulative angular positon of the motor.
# qube.getPendulumAngle() - Returns the cumulative angular position of the pendulum.
# qube.getMotorRPM() - Returns the newest rpm reading of the motor.
# qube.getMotorCurrent() - Returns the newest reading of the motor's current.
# ------------------------------------- AVAILABLE FUNCTIONS --------------------------------#

from QUBE import *
from logger import *
from com import *
from liveplot import *
from control import *
from time import time
import threading

# Replace with the Arduino port. Can be found in the Arduino IDE (Tools -> Port:)
port = COM_PORT
baudrate = 115200
qube = QUBE(port, baudrate)

# Resets the encoders in their current position.
qube.resetMotorEncoder()
qube.resetPendulumEncoder()

# Enables logging - comment out to remove
enableLogging()

t_last = time()

motor_target = 0
pendulum_target = 0
rpm_target = 0
pid = PID()


def control(data, lock):
    global motor_target, pendulum_target, rpm_target, pid
    while True:
        motor_target = MOTOR_TARGET_ANGLE
        pendulum_target = PENDULUM_TARGET_ANGLE
        rpm_target = MOTOR_TARGET_RPM

        # Updates the qube - Sends and receives data
        qube.update()
        qube.setRGB(0, 999, 0)

        # Gets the logdata and writes it to the log file
        logdata = qube.getLogData(motor_target, pendulum_target, rpm_target)
        save_data(logdata)

        # Multithreading stuff that must happen. Dont mind it.
        with lock:
            doMTStuff(data)

        # Get deltatime
        dt = getDT()

        # Set pid parameters using GUI
        setPidParams(pid)

        # Get states
        motor_degrees = qube.getMotorAngle()
        pendulum_degrees = qube.getPendulumAngle()
        rpm = qube.getMotorRPM()

        # Get control signal
        u = control_system(dt, motor_degrees, pendulum_degrees, rpm)
        
        # Apply control signal
        qube.setMotorVoltage(u)


def getDT():
    global t_last
    t_now = time()
    dt = t_now - t_last
    t_last += dt
    return dt


def doMTStuff(data):
    packet = data[9]
    pid.copy(packet.pid)
    if packet.resetEncoders:
        qube.resetMotorEncoder()
        qube.resetPendulumEncoder()
        packet.resetEncoders = False

    new_data = qube.getPlotData(motor_target, pendulum_target, rpm_target)
    for i, item in enumerate(new_data):
        data[i].append(item)


if __name__ == "__main__":
    try:
        _data = [[], [], [], [], [], [], [], [], [], Packet()]
        lock = threading.Lock()

        if not USING_MAC:
            thread1 = threading.Thread(target=startPlot, args=(_data, lock))
            thread2 = threading.Thread(target=control, args=(_data, lock))
            thread1.start()
            thread2.start()
            thread1.join()
            thread2.join()
            
            
            print("Plot closed. Exiting program.")
            if not thread1.is_alive():
                qube.setMotorVoltage(0)
                exit()
        else:
            thread1 = threading.Thread(target=control, args=(_data, lock))
            thread1.start()
            thread1.join()

    except:
        print("UNKNOWN ERROR OCCURRED")
        qube.setMotorVoltage(0)
