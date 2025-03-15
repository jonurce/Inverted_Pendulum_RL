// Include required libraries
#include <LiquidCrystal.h>
#include <Wire.h>

// Initialize the library by associating LCD pins with Arduino pins
const int rs = 13, en = 12, d4 = 11, d5 = 10, d6 = 9, d7 = 8;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

// Define pin connections
const int stepPin = 6;     // Connected to PUL+ on DM556
const int dirPin = 7;      // Connected to DIR+ on DM556
const int BUTTON_Up = 5;    // Button for clockwise rotation
const int BUTTON_Down = 4;   // Button for counterclockwise rotation
const int BUTTON_RESET = 26; // Button to reset angle to 0
const int sensor1Pin = 30;   //Start timer
const int sensor2Pin = 31;   //Stop Timer
const int ENC_A = 27;        // Encoder Channel A (interrupt pin)
const int ENC_B = 28;        // Encoder Channel B
const int ENC_INDEX = 29;    // Encoder Index pin (optional)

// Variables to store timing
unsigned long s1Time = 0;  // Time when S1 is triggered
unsigned long s2Time = 0;  // Time when S2 is triggered
unsigned long timeDiffMillis = 0; // Time difference in milliseconds

// State tracking
bool s1Triggered = false;  // Flag for S1
bool s2Triggered = false;  // Flag for S2

void setup() {
  // Start serial communication
  Serial.begin(9600);

  // Set up the LCD's number of columns and rows:
  lcd.begin(16, 2);

  // Set sensor pins as inputs with pull-up resistors (invert logic)
  pinMode(sensor1Pin, INPUT_PULLUP);
  pinMode(sensor2Pin, INPUT_PULLUP);

  pinMode(BUTTON_Up, INPUT_PULLUP);
  pinMode(BUTTON_Down, INPUT_PULLUP);

  // Declare pins as output:
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
}

void loop() {

  // SERVO MOTOR CODE
  int Up_State = !digitalRead(BUTTON_Up);
  int Down_State = !digitalRead(BUTTON_Down);

  if (Up_State == 1){
    digitalWrite(dirPin, HIGH);
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(200);
    digitalWrite(stepPin, LOW);
  }
  else if (Down_State == 1){
    digitalWrite(dirPin, LOW);
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(200);
    digitalWrite(stepPin, LOW);
  }
  
  // LASER SENSOR CODE
  int sensor1State = !digitalRead(sensor1Pin);
  int sensor2State = !digitalRead(sensor2Pin);

  // If S1 is triggered, store the timestamp
  if (sensor1State == 1 && !s1Triggered) {
    s1Time = millis();
    s1Triggered = true;
    s2Triggered = false;  // Reset S2 flag for a new measurement
    Serial.println("S1 triggered! Time recorded.");
  }

  // If S2 is triggered after S1, calculate the time difference
  if (sensor2State == 1 && s1Triggered && !s2Triggered) {
    s2Time = millis();
    timeDiffMillis = s2Time - s1Time; // Keep milliseconds
    s2Triggered = true;  // Mark S2 as triggered

    // Print to Serial Monitor
    Serial.print("S2 triggered! Time difference: ");
    Serial.print(timeDiffMillis / 1000); // Whole seconds
    Serial.print(".");
    Serial.print(timeDiffMillis % 1000); // Milliseconds
    Serial.println(" sec");
  }

  // DISPLAY CODE
  lcd.clear();
  // Print time difference when available
  lcd.setCursor(0, 0);
  if (s2Triggered) {
    lcd.print("Time: ");
    lcd.print(timeDiffMillis / 1000); // Whole seconds
    lcd.print(".");
    if (timeDiffMillis % 1000 < 100) lcd.print("0"); // Add leading zero if < 100ms
    if (timeDiffMillis % 1000 < 10) lcd.print("0");  // Add extra leading zero if < 10ms
    lcd.print(timeDiffMillis % 1000); // Milliseconds
    lcd.print("s");
  } else {
    lcd.print("Waiting...");
  }

  delayMicroseconds(600);
}
