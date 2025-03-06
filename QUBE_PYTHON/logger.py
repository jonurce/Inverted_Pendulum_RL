import csv
import os
import time

LOGGING = False

fieldnames = [
    "time",
    "motor_angle",
    "motor_target",
    "pendulum_angle",
    "pendulum_target",
    "rpm",
    "rpm_target",
    "voltage",
    "current",
]

files = 0
directory = os.path.join(os.curdir, "Data")

# Check if the directory exists
if os.path.exists(directory) and os.path.isdir(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            files += 1

    print(f"Number of files in 'Data': {files}")
else:
    print("The 'Data' directory does not exist.")

filename = f"Data/log{files}.csv"

counter = 0
startTime = time.time()


def enableLogging():
    with open(filename, "w", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    global LOGGING
    LOGGING = True


def save_data(data):
    if not LOGGING:
        return
    global counter
    global filename
    elapsedTime = round(time.time() - startTime, 3)

    with open(filename, "a", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        info = {
            "time": elapsedTime,
            "motor_angle": data[0],
            "motor_target": data[1],
            "pendulum_angle": data[2],
            "pendulum_target": data[3],
            "rpm": data[4],
            "rpm_target": data[5],
            "voltage": data[6],
            "current": data[7],
        }
        csv_writer.writerow(info)
