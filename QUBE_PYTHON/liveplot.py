import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QLineEdit,
    QLabel,
    QGridLayout,
    QCheckBox,
    QHBoxLayout,
)
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
from com import Packet
from time import time
from config import *

keys = {
    "NONE": -1,
    "MOTOR_ANGLE": 0,
    "MOTOR_TARGET_ANGLE": 1,
    "PENDULUM_ANGLE": 2,
    "PENDULUM_TARGET_ANGLE": 3,
    "RPM": 4,
    "RPM_TARGET": 5,
    "VOLTAGE": 6,
    "CURRENT": 7,
}

class LivePlotter(QMainWindow):
    def __init__(self, data, lock):
        super(LivePlotter, self).__init__()

        self.startTime = time()
        self.timeStamps = []
        self.setWindowTitle("QUBE PID Tuner")
        self.setGeometry(100, 100, 1080, 720)

        # Set the background color to black
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(9, 3, 38))
        self.setPalette(palette)

        # Create the main widget and layout
        central_widget = QWidget(self)
        central_widget.setAutoFillBackground(True)
        central_widget.setPalette(palette)

        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)  # Set vertical spacing between widgets

        # Create a grid layout for the plots
        canvas_layout = QGridLayout()

        self.titles = [
            PLOT1_TITLE,
            PLOT2_TITLE,
            PLOT3_TITLE,
            PLOT4_TITLE
        ]
        self.axisTitles = [
            PLOT1_AXISTITLE,
            PLOT2_AXISTITLE,
            PLOT3_AXISTITLE,
            PLOT4_AXISTITLE
        ]
        # Create the PyQtGraph plots
        self.plots = []
        for i in range(4):
            plot = pg.PlotWidget()
            plot.setBackground((9, 0, 48))  # Set the plot background color to black
            plot.getAxis("bottom").setLabel(
                text="Time (s)", color="#F30A49"
            )  # Red color for the x-axis label
            plot.getAxis("bottom").setPen(pg.mkPen(color=(255, 255, 255), width=2))

            plot.getAxis("left").setLabel(
                text=self.axisTitles[i], color="#F30A49"
            )  # Green color for the y-axis label
            plot.getAxis("left").setPen(pg.mkPen(color=(255, 255, 255), width=2))

            plot.setTitle(
                self.titles[i], color=(255, 255, 255)
            )  # Blue color for the title
            self.plots.append(plot)
            canvas_layout.addWidget(plot, int(i / 2), i % 2)

        # Create the update timer for plots
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_plot)
        self.update_interval = int(1000/UPDATE_FREQUENCY)  # Update interval in milliseconds
        self.dataPointsLimit = MAX_DATA_POINTS
        # Initialize data for each plot
        self.plot_data = data
        self.lock = lock
        self.packet = Packet()

        # Create labels and input fields for PID values
        pid_labels = [
            "Proportional Gain (KP):",
            "Integral Gain (KI):",
            "Derivative Gain (KD):",
            "Windup Limit:",
        ]
        self.pid_inputs = []
        self.windup_checkbox = QCheckBox("Enable Windup", self)  # Add checkbox

        rows = []
        textPalette = QPalette()
        textPalette.setColor(QPalette.WindowText, QColor(207, 45, 83))
        fieldPalette = QPalette()
        fieldPalette.setColor(QPalette.WindowText, QColor(255, 2, 255))
        self.windup_checkbox.setPalette(textPalette)
        for i, label in enumerate(pid_labels):
            box = QHBoxLayout()
            rows.append(box)
            _label = QLabel(label)
            _label.setPalette(textPalette)
            _label.setFixedWidth(100)
            rows[i].addWidget(_label, 0)

            # QLineEdit
            input_field = QLineEdit(self)
            input_field.setMaximumWidth(50)
            input_field.setText("0")
            input_field.setPalette(fieldPalette)
            rows[i].addWidget(input_field, 1)

            self.pid_inputs.append(input_field)

        spacer = QLabel("")
        rows[0].addWidget(spacer, 2)

        # Create the "Reset Encoders" button
        reset_button = QPushButton("Reset Encoders", self)
        reset_button.clicked.connect(self.reset_encoders)
        reset_button.setMaximumWidth(100)
        rows[1].addWidget(reset_button, 2)
        rows[1].addWidget(spacer, 3)

        # Create the "Set PID" button
        set_pid = QPushButton("Set PIDW", self)
        set_pid.clicked.connect(self.set_pid_params)
        set_pid.setMaximumWidth(100)
        rows[2].addWidget(set_pid, 2)
        rows[2].addWidget(spacer, 3)

        # Add the windup checkbox next to the "Windup Limit" label and input field
        rows[3].addWidget(self.windup_checkbox, 3)
        self.windup_checkbox.stateChanged.connect(self.check_windup)

        # Add the grid layout to the main layout
        layout.addLayout(canvas_layout)
        for row in rows:
            layout.addLayout(row)

        # Connect the update timer
        self.update_timer.start(self.update_interval)

    def set_pid_params(self):
        try:
            self.packet.pid.kp = float(self.pid_inputs[0].text())
            self.packet.pid.ki = float(self.pid_inputs[1].text())
            self.packet.pid.kd = float(self.pid_inputs[2].text())
            self.packet.pid.windup = float(self.pid_inputs[3].text())
        except ValueError:
            ...

    def reset_encoders(self):
        self.packet.resetEncoders = True

    def check_windup(self):
        self.packet.pid.useWindup = self.windup_checkbox.isChecked()

    def plotGraph1(self):
        times = self.plot_data[8]

        if (keys[PLOT1_VALUE_1] >= 0):
            self.plots[0].plot(
                times, self.plot_data[keys[PLOT1_VALUE_1]], name=PLOT1_LEGENDS[0], pen=pg.mkPen(color="m", width=1)
            )
        if (keys[PLOT1_VALUE_2] >= 0):
            self.plots[0].plot(
                times, self.plot_data[keys[PLOT1_VALUE_2]], name=PLOT1_LEGENDS[1], pen=pg.mkPen(color="w", width=1)
            )
        self.plots[0].addLegend(offset=(-10, -140))

    def plotGraph2(self):
        times = self.plot_data[8]
        if (keys[PLOT2_VALUE_1] >= 0):
            self.plots[1].plot(
                times, self.plot_data[keys[PLOT2_VALUE_1]], name=PLOT2_LEGENDS[0], pen=pg.mkPen(color="m", width=1)
            )
        if (keys[PLOT2_VALUE_2] >= 0):
            self.plots[1].plot(
                times, self.plot_data[keys[PLOT2_VALUE_2]], name=PLOT2_LEGENDS[1], pen=pg.mkPen(color="w", width=1)
            )
        self.plots[1].addLegend(offset=(-10, -140))

    def plotGraph3(self):
        times = self.plot_data[8]
        if (keys[PLOT3_VALUE_1] >= 0):
            self.plots[2].plot(
                times, self.plot_data[keys[PLOT3_VALUE_1]], name=PLOT3_LEGENDS[0], pen=pg.mkPen(color="m", width=1)
            )
        if (keys[PLOT3_VALUE_2] >= 0):
            self.plots[2].plot(
                times, self.plot_data[keys[PLOT3_VALUE_2]], name=PLOT3_LEGENDS[1], pen=pg.mkPen(color="w", width=1)
            )
        self.plots[2].addLegend(offset=(-10, -140))

    def plotGraph4(self):
        times = self.plot_data[8]
        if (keys[PLOT4_VALUE_1] >= 0):
            self.plots[3].plot(
                times, self.plot_data[keys[PLOT4_VALUE_1]], name=PLOT4_LEGENDS[0], pen=pg.mkPen(color="m", width=1)
            )
        if (keys[PLOT4_VALUE_2] >= 0):
            self.plots[3].plot(
                times, self.plot_data[keys[PLOT4_VALUE_2]], name=PLOT4_LEGENDS[1], pen=pg.mkPen(color="w", width=1)
            )
        self.plots[3].addLegend(offset=(-10, -140))

    def update_plot(self):
        # Update input
        with self.lock:
            self.plot_data[9] = self.packet

        datapoints = len(self.plot_data[0])
        for i in range(len(self.plot_data) - 1):
            if datapoints > self.dataPointsLimit:
                self.plot_data[i] = self.plot_data[i][
                    datapoints - self.dataPointsLimit :
                ]

        for plot in self.plots:
            plot.clear()
        self.plotGraph1()
        self.plotGraph2()
        self.plotGraph3()
        self.plotGraph4()


def startPlot(data, lock):
    app = QApplication(sys.argv)
    window = LivePlotter(data, lock)
    window.show()
    app.exec_()