import tkinter as tk
from tkinter import Scale, Entry
from QUBE import *
from time import sleep
from control import COM_PORT


class MotorControlGUI:
    def __init__(self, master, qube):
        self.master = master
        self.qube = qube
        master.title("Motor Control GUI")

        # Frame for encoder buttons
        encoder_frame = tk.Frame(master)
        encoder_frame.pack(pady=10)
        self.reset_encoder1_button = tk.Button(
            encoder_frame, text="Reset Motor Encoder", command=self.reset_encoder1
        )
        self.reset_encoder1_button.grid(row=0, column=0, padx=5)
        self.reset_encoder2_button = tk.Button(
            encoder_frame, text="Reset Pendulum Encoder", command=self.reset_encoder2
        )
        self.reset_encoder2_button.grid(row=0, column=1, padx=5)

        # Frame for RGB sliders
        rgb_frame = tk.Frame(master)
        rgb_frame.pack(pady=10)
        self.r_slider = Scale(rgb_frame, from_=999, to=0, label="Red")
        self.r_slider.grid(row=0, column=0, padx=5)
        self.g_slider = Scale(rgb_frame, from_=999, to=0, label="Green")
        self.g_slider.grid(row=0, column=1, padx=5)
        self.b_slider = Scale(rgb_frame, from_=999, to=0, label="Blue")
        self.b_slider.grid(row=0, column=2, padx=5)

        # Frame for motor speed and voltage sliders
        motor_frame = tk.Frame(master)
        motor_frame.pack(pady=10)
        self.speed_voltage_slider = Scale(
            motor_frame,
            from_=-24,
            to=24,
            orient=tk.HORIZONTAL,
            label="Motor Voltage",
            length=500,
            resolution=0.1,
        )
        self.speed_voltage_slider.grid(row=0, column=0, padx=20)

        # Frame for textfields
        text_frame = tk.Frame(master)
        text_frame.pack(pady=10)

        tk.Label(text_frame, text="Motor Position").grid(row=1, column=0, padx=5)
        self.position_textfield = Entry(text_frame, width=10, justify=tk.CENTER)
        self.position_textfield.grid(row=1, column=1, padx=5)

        tk.Label(text_frame, text="Absolute Position").grid(row=2, column=0, padx=5)
        self.position_absolute_textfield = Entry(
            text_frame, width=10, justify=tk.CENTER
        )
        self.position_absolute_textfield.grid(row=2, column=1, padx=5)

        tk.Label(text_frame, text="Pendulum Position").grid(row=1, column=2, padx=5)
        self.pendulum_position_textfield = Entry(
            text_frame, width=10, justify=tk.CENTER
        )
        self.pendulum_position_textfield.grid(row=1, column=3, padx=5)

        tk.Label(text_frame, text="Absolute Position").grid(row=2, column=2, padx=5)
        self.pendulum_position_absolute_textfield = Entry(
            text_frame, width=10, justify=tk.CENTER
        )
        self.pendulum_position_absolute_textfield.grid(row=2, column=3, padx=5)

        tk.Label(text_frame, text="RPM").grid(row=3, column=0, padx=5)
        self.rpm_textfield = Entry(text_frame, width=20, justify=tk.CENTER)
        self.rpm_textfield.grid(row=3, column=1, padx=5)

        tk.Label(text_frame, text="Current (mA)").grid(row=4, column=0, padx=5)
        self.current_textfield = Entry(text_frame, width=20, justify=tk.CENTER)
        self.current_textfield.grid(row=4, column=1, padx=5)

    def reset_encoder1(self):
        self.qube.resetMotorEncoder()

    def reset_encoder2(self):
        self.qube.resetPendulumEncoder()

    def update_gui(self):
        # Update GUI components with QUBE information
        self.qube.setRGB(self.r_slider.get(), self.g_slider.get(), self.b_slider.get())
        self.qube.setMotorVoltage(self.speed_voltage_slider.get())

        self.rpm_textfield.delete(0, tk.END)
        self.rpm_textfield.insert(0, str(self.qube.getMotorRPM()))

        self.current_textfield.delete(0, tk.END)
        self.current_textfield.insert(0, str(self.qube.getMotorCurrent()))

        motorPos = self.qube.getMotorAngle()
        self.position_textfield.delete(0, tk.END)
        self.position_textfield.insert(0, str(round(motorPos, 2)))

        motorPos = motorPos % 360
        self.position_absolute_textfield.delete(0, tk.END)
        self.position_absolute_textfield.insert(0, str(round(motorPos, 2)))

        pendulumPos = self.qube.getPendulumAngle()
        self.pendulum_position_textfield.delete(0, tk.END)
        self.pendulum_position_textfield.insert(0, str(round(pendulumPos, 2)))

        pendulumPos = pendulumPos % 360
        self.pendulum_position_absolute_textfield.delete(0, tk.END)
        self.pendulum_position_absolute_textfield.insert(0, str(round(pendulumPos, 2)))


def main():
    port = COM_PORT
    baudrate = 115200
    qube = QUBE(port, baudrate)
    qube.resetMotorEncoder()
    qube.resetPendulumEncoder()
    qube.setMotorSpeed(0)

    root = tk.Tk()
    gui = MotorControlGUI(root, qube)

    while True:
        qube.update()
        gui.update_gui()
        root.update()


if __name__ == "__main__":
    main()
