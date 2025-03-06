#include "QUBE.hpp"

void QUBE::print()
{
    Serial.print("Motor: ");
    Serial.print(getMotorAngle());
    Serial.print("    Pendulum: ");
    Serial.print(getPendulumAngle());
    Serial.print("    RPM: ");
    Serial.print(getRPM());
    Serial.print("    Current (mA): ");
    Serial.print(getMotorCurrent());
    Serial.print("    Motor voltage (V):");
    Serial.print(getMotorVoltage());
    Serial.print("    Watts (W):");
    Serial.print(getMotorVoltage() * getMotorCurrent() * 0.001);
    Serial.print("    Amplifier error: ");
    Serial.print(amplifierFault);
    Serial.print("    Stall detected: ");
    Serial.print(stallDetected);
    Serial.print("    Stall Error: ");
    Serial.print(stallError);
    Serial.println();
}

QUBE::QUBE(int cs)
{
    CS = cs; // default is pin 10
    pinMode(CS, OUTPUT);
    digitalWrite(CS, HIGH);

    output[0] = 1; // Mode is always 1
    output[1] = 0; // Padding is always 0
    output[2] = 0; // Mask is set during the other functions
}

void QUBE::setRGB(int r, int g, int b)
{
    r = constrain(r, 0, 999);
    byte R_MSB = r >> 8;
    byte R_LSB = r;

    g = constrain(g, 0, 999);
    byte G_MSB = g >> 8;
    byte G_LSB = g;

    b = constrain(b, 0, 999);
    byte B_MSB = b >> 8;
    byte B_LSB = b;

    lastColor[0] = r;
    lastColor[1] = g;
    lastColor[2] = b;

    output[2] |= B00011100; // Adds RGB to write mask
    output[3] = R_MSB;
    output[4] = R_LSB;
    output[5] = G_MSB;
    output[6] = G_LSB;
    output[7] = B_MSB;
    output[8] = B_LSB;
}

void QUBE::setErrorLight()
{

    long now = micros();
    if (now - LEDBlinkTimer > 1e6 * 0.25)
    {
        LEDBlinkTimer += 1e6 * 0.25;
        LED_ON = !LED_ON;
    }

    int r, g, b = 0;

    if (stallDetected)
    {
        r = 999;
    }

    if (stallError)
    {
        r = 999 * LED_ON;
    }

    if (amplifierFault)
    {
        r = (now % 1000000) * 1e-3;
        r = constrain(r, 0, 999);
    }

    output[2] |= B00011100; // Adds RGB to write mask
    output[3] = r >> 8;
    output[4] = r;
    output[5] = g >> 8;
    output[6] = g;
    output[7] = b >> 8;
    output[8] = b;
}

void QUBE::setMotorSpeed(int v)
{
    bool dir = v >= 0;

    v = constrain(v, -999, 999);
    voltage = 24.0 * v / 999.0;

    v += (1 << (16 - dir)); // Activates motor

    byte v_MSB = v >> 8;
    byte v_LSB = v;

    output[2] |= B00000011; // Adds Motor commands to write mask
    output[15] = v_MSB;
    output[16] = v_LSB;
}

void QUBE::setMotorVoltage(float V)
{
    int pwm_duty_cycle_10x = (V / 24.0) * 999.0; // Map voltage to motor command range (actually the number is 10x the motors pwm duty cycle)
    setMotorSpeed(pwm_duty_cycle_10x);
}

void QUBE::resetMotorEncoder()
{
    setMotorEncoder(0);
}

void QUBE::resetPendulumEncoder()
{
    setPendulumEncoder(0);
}

void QUBE::setMotorEncoder(int count)
{
    output[2] |= B00100000; // Adds setEncoder0 to write mask
    output[9] = count >> 16;
    output[10] = (count << 16) >> 24;
    output[11] = (count << 24) >> 32;
}

void QUBE::setPendulumEncoder(int count)
{
    output[2] |= B01000000; // Adds setEncoder1 to write mask
    output[12] = count >> 16;
    output[13] = (count << 16) >> 24;
    output[14] = (count << 24) >> 32;
}

void QUBE::begin()
{
    SPI.begin();
}

long QUBE::getMotorEncoder()
{

    long data = input[1];
    bool negative = input[1] >> 23;
    if (negative)
        data -= 0xFFFFFF;
    return data;
}

long QUBE::getPendulumEncoder()
{
    long data = input[2];
    bool negative = input[2] >> 23;
    if (negative)
        data -= 0xFFFFFF;
    return data;
}

float QUBE::getMotorAngle(bool absolute)
{
    long count = getMotorEncoder();

    if (absolute)
    {
        count %= 2048;
        if (count <= -1024)
        {
            count += 2048;
        }

        if (count > 1024)
        {
            count -= 2048;
        }
    }

    float angle = ((float)count / 2048.0) * 360.0;
    return angle;
}

float QUBE::getPendulumAngle(bool absolute)
{

    long count = getPendulumEncoder();
    if (absolute)
    {
        count %= 2048;
        if (count <= -1024)
        {
            count += 2048;
        }

        if (count > 1024)
        {
            count -= 2048;
        }
    }

    float angle = ((float)count / 2048.0) * 360.0;
    return angle;
}

float QUBE::getRPM()
{
    uint32_t tach = input[3];

    if (tach == 0x7FFFFF || tach == 0xFFFFFF)
    {
        return 0; // Defined in datasheet to be 0
    }

    bool tachDir = (tach >> 23); // Direction is the MSB (1 is positive, 0 is negative)
    tach = tach & 0x7FFFFF;      // Left shift by 9 to remove MSB
    float rpm = (4.0 / (tach * 25.0 * pow(10, -9))) * 60 / 512 / 4;
    tachDir ? rpm = rpm : rpm = -rpm;
    return rpm;
}

float QUBE::getMotorVoltage()
{
    return voltage;
}

void QUBE::checkStatus()
{
    byte status = input[4];

    amplifierFault = status & B00000001; // Unsure, seems to activate just like stallDetected?
    stallDetected = status & B00000010;  // activates when motor is close to or has stalled
    stallError = status & B00000100;     // activates if stallDetected has been true for about 5 seconds
}

float QUBE::getMotorCurrent()
{
    float current = (input[5] - 8190.0) / 9828.0;
    return current * 1000; // Huh? Formula seems to give amps not milliamps. Also, website specifies 0.54A nominal
    // however, maximum amperage I was able to reach was 0.43A at stall.
}

void QUBE::update()
{
    checkStatus();
    if (stallError || amplifierFault || stallDetected)
    {
        setErrorLight();
    }
    else
    {
        setRGB(lastColor[0], lastColor[1], lastColor[2]);
    }

    SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE2));
    digitalWrite(CS, LOW);

    int ID = SPI.transfer(output[0]) << 8; // Mode bit
    ID |= SPI.transfer(output[1]);         // Padding bit

    byte encoder01 = SPI.transfer(output[2]); // Write mask: | empty bit | Set encoder 1 | Set encoder 0 | Write Blue LED | Write Green LED | Write Red LED | Write Motor Enable | Write Motor |
    byte encoder02 = SPI.transfer(output[3]); // Red MSB (scale is 0-999)
    byte encoder03 = SPI.transfer(output[4]); // Red LSB

    byte encoder11 = SPI.transfer(output[5]); // Green MSB
    byte encoder12 = SPI.transfer(output[6]); // Green LSB
    byte encoder13 = SPI.transfer(output[7]); // Blue MSB

    byte tach1 = SPI.transfer(output[8]);   // Blue LSB
    byte tach2 = SPI.transfer(output[9]);   // Set Encoder0 (23-16)
    byte tach3 = SPI.transfer(output[10]);  // Set Encoder0 (15-8)
    byte status = SPI.transfer(output[11]); // Set Encoder0 (7-0)

    int currentSense = SPI.transfer(output[12]) << 8; // Set Encoder1 (23-16)
    currentSense |= SPI.transfer(output[13]);         // Set Encoder1 (15-8)

    SPI.transfer(output[14]); // Set Encoder1 (7-0)
    SPI.transfer(output[15]); // Motor Command (15-8)
    SPI.transfer(output[16]); // Motor Command (7-0)

    digitalWrite(CS, HIGH);
    SPI.endTransaction();

    input[0] = ID;
    input[1] = ((uint32_t)encoder01) << 16 | ((uint32_t)encoder02) << 8 | encoder03;
    input[2] = ((uint32_t)encoder11) << 16 | ((uint32_t)encoder12) << 8 | encoder13;
    input[3] = ((uint32_t)tach1) << 16 | ((uint32_t)tach2) << 8 | tach3;
    input[4] = status;
    input[5] = currentSense;

    output[2] = 0;
}