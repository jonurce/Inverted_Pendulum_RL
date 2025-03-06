
#include <SD.h>
#include <SPI.h>

const int chipSelect = BUILTIN_SDCARD;   
File dataFile;

void setup()
{

  // Open serial communications and wait for port to open:
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect.
  }


  Serial.print("Initializing SD card...");

  // see if the card is present and can be initialized:
  if (!SD.begin(chipSelect)) {
    Serial.println("Card failed, or not present");
    while (1) {
      // No SD card, so don't do anything more - stay stuck here
    }
  }
  Serial.println("card initialized.");
  dataFile = SD.open("datalog.txt", FILE_WRITE);
}

void loop()
{
  // make a string for assembling the data to log:
  long t = micros();
  String dataString = "111111111111111111111111111111";

  for (int i = 0; i < 500; i++) {

    // if the file is available, write to it:
    if (dataFile) {
      dataFile.println(dataString);
    } else {
      Serial.println("error opening datalog.txt");
    }
  }
  long t2 = micros();
  dataFile.println(String(t2 - t));
  dataFile.close();
  delay(10000);
}