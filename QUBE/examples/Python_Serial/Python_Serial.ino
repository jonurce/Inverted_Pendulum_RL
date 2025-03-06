#include "QUBE.hpp"

QUBE qube;

byte inputBuffer[10];

void setup() {
  Serial.begin(115200);
  qube.begin();
  qube.resetMotorEncoder();
  qube.resetPendulumEncoder();
  qube.setMotorVoltage(0);
  qube.setRGB(0, 999, 999);
  qube.update();
}

bool receiveData() {
  if (Serial.available() >= 10) {    
    bool resetMotorEncoder = Serial.read();
    bool resetPendulumEncoder = Serial.read();

    int r_MSB = Serial.read();
    int r_LSB = Serial.read();
    int r = (r_MSB << 8) + r_LSB;

    int g_MSB = Serial.read();
    int g_LSB = Serial.read();
    int g = (g_MSB << 8) + g_LSB;

    int b_MSB = Serial.read();
    int b_LSB = Serial.read();
    int b = (b_MSB << 8) + b_LSB;

    int motorCommand_MSB = Serial.read();
    int motorCommand_LSB = Serial.read();
    int motorCommand = (motorCommand_MSB << 8) + motorCommand_LSB - 999;

    if (resetMotorEncoder) {
      qube.resetMotorEncoder();
    }
    if (resetPendulumEncoder) {
      qube.resetPendulumEncoder();
    }
    qube.setRGB(r, g, b);
    qube.setMotorSpeed(motorCommand);
    return true;
  }
  return false;
}

void sendEncoderData(bool encoder) {

  float encoderAngle = 0;

  if (encoder == 1) {
    encoderAngle = qube.getPendulumAngle(false);
  } else {
    encoderAngle = qube.getMotorAngle(false);
  }
  
  long revolutions = (long)encoderAngle/360.0;

  float _angle = encoderAngle - revolutions*360.0;
  long angle = (long)_angle; // represent the angle in integer value
  long angleDecimal = (_angle - angle) * 100; // represents the two decimal digits .xx

  if (encoderAngle < 0) { 
    revolutions = abs(revolutions);
    angle = abs(angle);
    angleDecimal = abs(angleDecimal);

    revolutions |= (1<<15); // 0bx000 0000 0000 0000 - MSB is sign (1 means negative), bits (14-0) are revolution count
    
  }
  angle = (angle << 7) | angleDecimal; // angle range is from 0-360 an requires 9 bits, angleDecimal is from 0-100 and requires 7 bits. Bits (15-7) represent int, bits (6-0) represent dec.

  byte rev_MSB = revolutions >> 8;
  byte rev_LSB = revolutions & 0xFF;
  byte ang_MSB = angle >> 8;
  byte ang_LSB = angle & 0xFF;


  Serial.write(highByte(revolutions));
  Serial.write(lowByte(revolutions));
  Serial.write(ang_MSB);
  Serial.write(ang_LSB);
}

void sendRPMData() {
  long rpm = (long)qube.getRPM();
  bool dir = rpm < 0;

  if (dir) {
    rpm = abs(rpm);
    rpm |= 1 << 15;
  }

  byte rpm_MSB = rpm >> 8;
  byte rpm_LSB = rpm & 0xFF;

  Serial.write(rpm_MSB);
  Serial.write(rpm_LSB);
}

void sendCurrentData() {
  long current = (long)qube.getMotorCurrent();
  
  current = abs(current);

  long current_MSB = current >> 8;
  long current_LSB = current & 0xFF;

  Serial.write(current_MSB);
  Serial.write(current_LSB);
}

void sendData() {
  sendEncoderData(0);
  sendEncoderData(1);
  sendRPMData();
  sendCurrentData();
}

void loop() {
  qube.update();

  bool received = false;
  while (!received) {
    received = receiveData();
  }
  sendData();
}
