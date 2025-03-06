from QUBE import QUBE
from time import sleep, time
import numpy as np

# Initialize QUBE
port = "COM3"  # Replace with the actual port
baudrate = 115200
qube = QUBE(port, baudrate)

# Physical parameters
m_p = 0.1  # Pendulum stick mass (kg)
l = 0.095  # Length of pendulum (m)
l_com = l / 2  # Distance to center of mass (m)
J = (1/3) * m_p * l * l  # Inertia (kg*m^2)
g = 9.81  # Gravitational constant (m/s^2)

# Swingup parameters
Er = 0.015  # Reference energy (Joules)
ke = 50  # Tunable gain for swingup voltage (m/s/J)
u_max = 2.5  # Max voltage (m/s^2)
balance_range = 20.0  # Range where mode switches to balancing (degrees)

# Balance parameters
s = 0.33
kp_theta = 1 * s
kd_theta = 0.125 * s
kp_pos = 0.07 * s
kd_pos = 0.06 * s

# Filter parameters
twopi = 3.141592 * 2
wc = 500 / twopi
wc2 = 500 / twopi
wc3 = 500 / twopi
y_k_last = 0
y2_k_last = 0
y3_k_last = 0

# Control loop parameters
freq = 300  # Frequency (Hz)
dt = 1.0 / freq  # Timestep (sec)

# Program variables
targetPos = 0
targetAngle = 0
t_balance = 0
prevAngle = 0
prevPos = 0
last = time()
t_reset = time()
mode = 0
lastMode = 0
reset = False

def setup():
    global qube
    qube.setRGB(999, 999, 999)
    qube.resetMotorEncoder()
    sleep(1)
    qube.resetPendulumEncoder()
    qube.update()
    sleep(1)

def swingup(angle):
    global prevAngle
    angularV = (angle - prevAngle) / dt
    print(angularV)
    prevAngle = angle
    
    E = 0.5 * J * angularV**2 + m_p * g * l_com * (1 - np.cos(angle))
    u = ke * (E - Er) * (-angularV * np.cos(angle))
    u_sat = max(-u_max, min(u, u_max))
    
    voltage = u_sat * (8.4 * 0.095 * 0.085) / 0.042
    qube.setMotorVoltage(voltage)

def settle(angle):
    angle = angle + 360 - 2 * angle if angle < 0 else -360 + 2 * angle
    swingup(angle)

def balance(position, angle):
    global prevAngle, prevPos, y_k_last, y2_k_last
    
    u_dot = (angle - prevAngle) / dt
    y_k = y_k_last + wc * dt * (u_dot - y_k_last)
    y_k_last = y_k
    
    v = (position - prevPos) / dt
    y2_k = y2_k_last + wc2 * dt * (v - y2_k_last)
    y2_k_last = y2_k
    
    u_pos = kp_pos * position + kd_pos * y2_k
    u_ang = kp_theta * angle + kd_theta * y_k
    u = u_pos + u_ang
    
    qube.setMotorVoltage(u)
    prevAngle = angle
    prevPos = position

def loop():
    global last, mode, t_balance, reset, t_reset, lastMode
    
    now = time()
    if (now - last) < dt:
        return
    last = now
    
    position = qube.getMotorAngle()
    angle = qube.getPendulumAngle() % 360
    angle = -180 + angle if angle > 0 else 180 + angle
    
    if mode == 0 and -balance_range < angle < balance_range:
        mode = 1
        t_balance = now
    elif mode == 1 and not (-balance_range < angle < balance_range):
        mode = 0
        if now - t_balance > 1:
            reset = True
            t_reset = now
    
    if reset:
        while time() - t_reset < 2:
            settle(angle)
            qube.update()
            angle = qube.getMotorAngle()
            sleep(dt)
        
        while time() - t_reset < 2:
            settle(angle)
            qube.update()
            angle = qube.getMotorAngle()
            sleep(dt)
        
        reset = False
        return
    
    if mode == 0:
        swingup(angle)
    elif mode == 1:
        balance(position, angle)
    
    qube.update()
    lastMode = mode

if __name__ == "__main__":
    setup()
    while True:
        loop()
