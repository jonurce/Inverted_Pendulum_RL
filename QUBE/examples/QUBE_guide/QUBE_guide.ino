/* Start by including the library. If you are not able to find the library, make sure to check
 * that you have placed the QUBE folder containing QUBE.cpp and QUBE.hpp inside your /Arduino/libraries
 * folder. Your arduino sketches (programs) should be saved in a folder that is located in /Arduino.
 */
#include "QUBE.hpp" // Includes the library

QUBE qube; // Creates a qube object

void setup()
{

  // Start serial data communication for printing to serial monitor.
  Serial.begin(115200);

  // Setup for the qube, initializes communication with the QUBE.
  qube.begin();

  // Zeros the motor encoder position at the current position.
  qube.resetMotorEncoder();

  // Zeros the pendulum encoder position at the current position.
  qube.resetPendulumEncoder();

  // Sets the LED to RGB colors (max is 999)
  qube.setRGB(0, 500, 999);

  // Sets the motor voltage. Ranges from -24V to 24V.
  qube.setMotorVoltage(0);

  // Sets the motor speed. Ranges from -999 to 999. The number indicates the PWM duty cycle multiplied by 10. I.e. a speed of 100 = 10% pwm duty cycle
  qube.setMotorSpeed(999);

  /* qube.update() - The function reads and writes data to the qube, and must be ran every time any changes are applied.
   * E.g: Running qube.setMotorSpeed() wont change the motor speed until qube.update() has been ran.
   * The communication takes on average 32.2 microseconds.
   */
  qube.update();

  // Prints out sensor data to the console. qube.update() should have been ran recently when using this.
  qube.print();
}

void loop()
{

  // The qube should be updated as often as needed, typically this is every loop.
  qube.update();

  /* getMotorAngle(bool absolute = true) returns the position of the motor shaft relative to a zero-position.
   * This zero-position can be set by physically rotating the shaft to a wanted position, and then running
   * qube.resetMotorEncoder().
   * The output can be either cumulative angle, or the absolute position. To retrieve the cumulative angle the parameter
   * must be set to false. I.E. qube.getMotorAngle(false);
   */
  float motorPosition = qube.getMotorAngle();

  // Works exactly like qube.getMotorAngle() but returns the pendulum's position.
  float pendulumPosition = qube.getPendulumAngle();

  // Returns the rpm of the motor. rpm comes in increments of 0.35rpm ranging from 0rpm to approximately 3500rpm.
  float rpm = qube.getRPM();

  // Returns the last current measurement in mA.
  float current = qube.getMotorCurrent();

  // Returns the applied motor voltage in V
  float voltage = qube.getMotorVoltage();
}
