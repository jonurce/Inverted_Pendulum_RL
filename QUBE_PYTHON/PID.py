import sys


class PID:
    def __init__(self):
        self.kp = 0
        self.ki = 0
        self.kd = 0
        self.windup = 0
        self.lastIntegral = 0
        self.lastError = 0
        self.useWindup = False

    def control(self, target, current, dt):

        e = target-current
        P = e*self.kp
        I = self.lastIntegral + e*dt*self.ki
        D = (e-self.lastError)/dt * self.kd

        self.lastIntegral = I
        self.lastError = e

        return P + I + D

    def copy(self, pid):
        self.kp = pid.kp
        self.ki = pid.ki
        self.kd = pid.kd
        self.windup = pid.windup
        self.useWindup = pid.useWindup
