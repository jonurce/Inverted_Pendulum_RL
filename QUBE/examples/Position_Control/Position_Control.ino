/* Start by including the library. If you are not able to find the library, make sure to check
 * that you have placed the QUBE folder containing QUBE.cpp and QUBE.hpp inside your /Arduino/libraries
 * folder. Your arduino sketches (programs) should be saved in a folder that is located in /Arduino.
 */
#include "QUBE.hpp"

QUBE qube;

class PID
{
public:
  float kp;
  float ki;
  float kd;

  PID(float _kp, float _ki, float _kd, float _windup = 0)
  {
    kp = _kp;
    ki = _ki;
    kd = _kd;
    windup = _windup;
  };

  float regulate(float dt, float target, float current)
  {
    float e = target - current;
    float P = kp * e;
    float I = lastIntegral + ki * e * dt;
    I = constrain(I, -windup, windup);
    float D = ((e - lastError) / dt) * kd;

    lastIntegral = I;
    lastError = e;
    return P + I + D;
  };

private:
  float lastIntegral;
  float lastError;
  float windup;
};

void setup()
{
  Serial.begin(115200);
  qube.begin();
  qube.resetMotorEncoder();
  qube.setRGB(999, 999, 999);
  qube.update();
  delay(1000);
}

// Target is the desired angle with 0.17 degrees per encoder count
float targetPos = 0;
float currentPos = qube.getMotorAngle(false);

/* The PID parameters were tested on a microcontroller with a 240MHz clock frequency.
 * They might have to be retuned for a different frequency as the update loop will be slower/faster.
 * Open the Serial Plotter to see the rpm of the disk.
 */
PID pid(1, 12, 0.02, 0.3);

// Used to calculate dt
long t_last = micros();

void loop()
{
  // Update QUBE and get current position
  qube.update();
  currentPos = qube.getMotorAngle(false);

  // Calculate loop time
  long t_now = micros();
  float dt = (t_now - t_last) * 1e-6;
  t_last = t_now;

  // Get regulator output
  float regulation = pid.regulate(dt, targetPos, currentPos);
  qube.setMotorVoltage(regulation);
}