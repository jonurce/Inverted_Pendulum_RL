# Plotting config settings

# Performance settings. Adjust if needed
UPDATE_FREQUENCY = 10 # How many times per second to update the plotview
MAX_DATA_POINTS = 500 # How many datapoints to show

# Set plot titles
PLOT1_TITLE = "Motor Angle"
PLOT2_TITLE = "Pendulum Angle"
PLOT3_TITLE = "RPM"
PLOT4_TITLE = "Voltage"

# Set axis titles
PLOT1_AXISTITLE = "Degrees"
PLOT2_AXISTITLE = "Degrees"
PLOT3_AXISTITLE = "RPM"
PLOT4_AXISTITLE = "Volts"

# Set plot legends
PLOT1_LEGENDS = ("Actual", "Target")
PLOT2_LEGENDS = ("Actual", "Target")
PLOT3_LEGENDS = ("Actual", "Target")
PLOT4_LEGENDS = ("Actual", "Target")

# Set which values to plot

# 1. MOTOR_ANGLE - Plots the motor angle
# 2. MOTOR_TARGET_ANGLE - Plots the motor target angle (User defined)
# 3. PENDULUM_ANGLE - Plots the pendulum angle
# 4. PENDULUM_TARGET_ANGLE - Plots the pendulum target angle (User defined)
# 5. RPM - Plots the motor rpm
# 6. RPM_TARGET - Plots the rpm target (User defined)
# 7. VOLTAGE - Plots the voltage applied to the mototr
# 8. CURRENT - Plots the measured motor current
# 9. NONE - Plots nothing

PLOT1_VALUE_1 = "MOTOR_ANGLE"
PLOT1_VALUE_2 = "MOTOR_TARGET_ANGLE"

PLOT2_VALUE_1 = "PENDULUM_ANGLE"
PLOT2_VALUE_2 = "PENDULUM_TARGET_ANGLE"

PLOT3_VALUE_1 = "RPM"
PLOT3_VALUE_2 = "RPM_TARGET"

PLOT4_VALUE_1 = "VOLTAGE"
PLOT4_VALUE_2 = "NONE"

