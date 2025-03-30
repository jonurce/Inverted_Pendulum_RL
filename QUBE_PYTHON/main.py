# ------------------------------------- AVAILABLE FUNCTIONS --------------------------------#
# qube.setRGB(r, g, b) - Sets the LED color of the QUBE. Color values range from [0, 999].
# qube.setMotorSpeed(speed) - Sets the motor speed. Speed ranges from [-999, 999].
# qube.setMotorVoltage(volts) - Applies the given voltage to the motor. Volts range from (-24, 24).
# qube.resetMotorEncoder() - Resets the motor encoder in the current position.
# qube.resetPendulumEncoder() - Resets the pendulum encoder in the current position.
from numpy.ma.core import arctan2
from sympy.physics.units import length

# qube.getMotorAngle() - Returns the cumulative angular position of the motor.
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

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math

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

# Real QUBE data
real_time_data = []
real_motor_angle_data = []
real_pendulum_angle_data = []
real_rpm_data = []
real_voltage_data = []

# Simulated data
sim_time_data = []
sim_motor_angle_data = []
sim_pendulum_angle_data = []
sim_rpm_data = []
sim_voltage_time_data = []
sim_voltage_data = []

run_time = 5


def control(data, lock):
    global motor_target, pendulum_target, rpm_target, pid
    start_time = time()  # Record start time
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

        #Get current time
        current_time = time() - start_time

        # Get control signal
        u = control_system(current_time, dt, motor_degrees, pendulum_degrees, rpm)

        # Store data
        real_time_data.append(current_time)
        real_motor_angle_data.append(motor_degrees)
        real_pendulum_angle_data.append(pendulum_degrees)
        real_rpm_data.append(rpm)
        real_voltage_data.append(u)

        # Stop after 20 seconds
        if current_time >= run_time:
            qube.setMotorVoltage(0)  # Stop the motor
            break

        # Apply control signal
        qube.setMotorVoltage(u)

def simulate():
    global sim_time_data, sim_motor_angle_data, sim_pendulum_angle_data, sim_rpm_data
    # Parameters (approximate values from QUBE-Servo 2 specs)
    m_1 = 0.024  # Pendulum mass (kg)
    l_1 = 0.128  # Pendulum length to CoM (m)
    I_1 = 0.000131  # Pendulum inertia about pivot (kg·m²)
    m_0 = 0.053  # Arm mass (kg)
    L_0 = 0.086  # Arm length (m)
    I_0 = 0.0000572  # Arm inertia about pivot (kg·m²)
    g = 9.81  # Gravity (m/s²)
    b_0 = 0.0003  # Viscous friction coefficient for arm (N·m·s/rad)
    b_1 = 0.0005  # Viscous friction coefficient for pendulum (N·m·s/rad)
    K_m = 0.0431
    R_m = 8.94

    dt = 0.001
    current_time = 0.0

    # System dynamics
    def dynamics(t, x):
        s0, c0, d0, s1, c1, d1 = x
        voltage = control_system(t,dt,np.degrees(arctan2(s0,c0)),np.degrees(arctan2(s1,c1)),d0*60/(2*np.pi))
        torque = K_m * (voltage - K_m * d0) / R_m

        # Based on Euler-Lagrange differential equation
        # theta_0 is arm angle
        # theta_1 is pendulum angle (0 upwards)
        # Mass matrix (left-hand side)
        alpha = I_0 + m_1 * L_0 ** 2 + m_1 * l_1 ** 2 * s1 ** 2
        beta = m_1 * l_1 ** 2 * (2 * s1 * c1)
        gamma = m_1 * L_0 * l_1 * c1
        sigma = m_1 * L_0 * l_1 * s1

        M = np.array([
            [alpha, gamma],
            [gamma, I_1 + m_1 * l_1 ** 2],
        ])

        # Right-hand side
        f = np.array([
            torque - b_0 * d0 + sigma * d1 ** 2 - beta * d0 * d1,
            -b_1 * d1 + m_1 * g * l_1 * s1 + 0.5 * beta * d0 ** 2,
        ])

        # Solve for accelerations: M * [theta_ddot, alpha_ddot] = f
        acc = np.linalg.solve(M, f)

        # State derivatives
        return [d0 * c0, -d0 * s0, acc[0], d1 * c1, -d1 * s1, acc[1]]

    # Initial conditions: [s0, c0, d0, s1, c1, d1]
    # Start with pendulum near upright (theta_1 = 0) and small perturbation
    theta_0 = 0.0
    theta_1 = np.pi
    x0 = [np.sin(theta_0), np.cos(theta_0), 0.0, np.sin(theta_1), np.cos(theta_1), 0.0]

    # Time span
    t_span = (0, run_time)
    t_eval = np.linspace(0, run_time, math.ceil(run_time/dt))

    # Solve ODE
    sol = solve_ivp(dynamics, t_span, x0, t_eval=t_eval, method='RK45')

    sim_time_data = sol.t
    sim_motor_angle_data = np.degrees(np.arctan2(sol.y[0], sol.y[1]))
    sim_pendulum_angle_data = np.degrees(np.arctan2(sol.y[3], sol.y[4]))
    sim_rpm_data = sol.y[2] * 60 / (2*np.pi)

    # Compute voltage for each t_eval point after solving
    for i in range(len(sol.t)):
        s0, c0, d0, s1, c1, d1 = sol.y[:, i]
        voltage = control_system(sol.t[i],dt, np.degrees(np.arctan2(s0, c0)), np.degrees(np.arctan2(s1, c1)),
                                 d0 * 60 / (2 * np.pi))
        sim_voltage_data.append(voltage)
        sim_voltage_time_data.append(dt*i)


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

def plot_results():
    # Create subplots for each variable
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # Motor angle
    axs[0].scatter(real_time_data, real_motor_angle_data, label="Real Motor Angle", color="blue", marker="o", s=1)
    axs[0].scatter(sim_time_data, sim_motor_angle_data, label="Sim Motor Angle", color="cyan", marker="x", s=1)
    axs[0].axhline(y=motor_target, color='r', linestyle='--', label=r"$\theta_0$ ="+f"{motor_target}")
    axs[0].set_ylabel("Motor Angle (deg)")
    axs[0].legend()
    axs[0].grid(True)

    # Pendulum angle
    axs[1].scatter(real_time_data, real_pendulum_angle_data, label="Real Pendulum Angle", color="blue", marker="o", s=1)
    axs[1].scatter(sim_time_data, sim_pendulum_angle_data, label="Sim Pendulum Angle", color="cyan", marker="x", s=1)
    axs[1].axhline(y=pendulum_target, color='r', linestyle='--', label=r"$\theta_1$ ="+f"{pendulum_target}")
    axs[1].set_ylabel("Pendulum Angle (deg)")
    axs[1].legend()
    axs[1].grid(True)

    # RPM
    axs[2].scatter(real_time_data, real_rpm_data, label="Real Motor RPM", color="blue", marker="o", s=1)
    axs[2].scatter(sim_time_data, sim_rpm_data, label="Sim Motor RPM", color="cyan", marker="x", s=1)
    axs[2].set_ylabel("Motor RPM")
    axs[2].legend()
    axs[2].grid(True)

    # Voltage
    axs[3].scatter(real_time_data, real_voltage_data, label="Real Voltage", color="blue", marker="o", s=1)
    axs[3].scatter(sim_voltage_time_data, sim_voltage_data, label="Sim Voltage", color="cyan", marker="x", s=1)
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Voltage (V)")
    axs[3].legend()
    axs[3].grid(True)

    plt.suptitle(f"QUBE: Real vs Simulated Performance (First {run_time} Seconds)")
    plt.show()


if __name__ == "__main__":
    try:
        _data = [[], [], [], [], [], [], [], [], [], Packet()]
        lock = threading.Lock()

        print("Running simulation...")
        simulate()
        print("Simulation finished.")

        if not USING_MAC:
            #thread1 = threading.Thread(target=startPlot, args=(_data, lock))
            thread2 = threading.Thread(target=control, args=(_data, lock))
            #thread1.start()
            thread2.start()
            #thread1.join()
            thread2.join()
            print("Real control finished.")

            #print("Plot closed. Exiting program.")
            #if not thread1.is_alive():
            #    qube.setMotorVoltage(0)
            #    exit()

        else:
            thread1 = threading.Thread(target=control, args=(_data, lock))
            thread1.start()
            thread1.join()
            print("Real control finished.")

        print("Plotting results...")
        plot_results()

    except:
        print("UNKNOWN ERROR OCCURRED")
        qube.setMotorVoltage(0)
