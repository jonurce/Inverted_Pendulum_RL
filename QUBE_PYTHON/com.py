from PID import *

class Packet:
    def __init__(self):
        self.pid = PID()
        self.plot_data = [[] * 9]
        self.resetEncoders = False

    def unpack(self):
        return [
            self.pid,
            self.resetEncoders,
        ]