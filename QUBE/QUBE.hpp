#ifndef QUBE_HPP
#define QUBE_HPP
#define CW 1
#define CCW 0

#include <Arduino.h>
#include <SPI.h>

class QUBE
{
public:
    QUBE(int cs = 10);

    byte output[17];
    uint32_t input[6];
    int CS;

    /* @brief Prints all data to the serial monitor.
     *
     */
    void print();

    /* @brief Sets the LED's RGB values.
     *
     * @param int r (0-999)
     * @param int g (0-999)
     * @param int b (0-999)
     */
    void setRGB(int r, int g, int b);

    /* @brief Sets the motor speed and direction.
     *
     * @param int v (0-999)
     */
    void setMotorSpeed(int v);

    /* @brief Sets the motor voltage and direction
     *
     * @param int V (0.0-24.0)
     */
    void setMotorVoltage(float V);

    /* @brief Resets the motor encoder to 0.
     *
     */
    void resetMotorEncoder();

    /* @brief Resets the pendulum encoder to 0.
     *
     */
    void resetPendulumEncoder();

    /* @brief Sets the encoder position.
     *
     * This will typically be 0.
     *
     * @param count
     */
    void setMotorEncoder(int count);

    /* @brief Sets the encoder position.
     *
     * This will typically be 0.
     *
     * @param count
     */
    void setPendulumEncoder(int count);

    /* @brief Initializes the communication with the QFLEX2 EMBEDDED board.
     *
     */
    void begin();

    /* @brief Sends and receives data once.
     *
     */
    void update();

    /* @brief Gets the motor current.
     *
     */
    float getMotorCurrent();

    /* @brief Gets the rpm of the motor.
     *
     */
    float getRPM();

    /* @brief Gets the motor encoder count.
     *
     */
    long getMotorEncoder();

    /* @brief Gets the pendulum encoder count.
     *
     */
    long getPendulumEncoder();

    /* @brief Gets the motor's angular position.
     *
     * Standardly the position is given as an absolute position. Changing the parameter to false yields cumulative position.
     * @param bool absolute = true
     */
    float getMotorAngle(bool absolute = true);

    /* @brief Gets the pendulums's angular position.
     *
     * Standardly the position is given as an absolute position. Changing the parameter to false yields cumulative position.
     * @param bool absolute = true
     */

    float getPendulumAngle(bool absolute = true);

    /* @brief Gets the applied motor voltage.
     *
     */
    float getMotorVoltage();

private:
    int lastColor[3];
    bool amplifierFault;
    bool stallDetected;
    bool stallError;
    long LEDBlinkTimer = 0;
    bool LED_ON;

    float lastMotorCount = 0;
    bool spinDir;

    float voltage;

    /* @brief Checks for any error messages in the last update from the QUBE.
     *
     */
    void checkStatus();

    /* @brief Updates the LED to indicate any error
     *
     */
    void setErrorLight();
};

#endif