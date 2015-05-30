/*
  Analog input, analog output, serial output
 
 Reads an analog input pin, maps the result to a range from 0 to 255
 and uses the result to set the pulsewidth modulation (PWM) of an output pin.
 Also prints the results to the serial monitor.
 
 The circuit:
 * potentiometer connected to analog pin 0.
   Center pin of the potentiometer goes to the analog pin.
   side pins of the potentiometer go to +5V and ground
 * LED connected from digital pin 9 to ground
 
 created 29 Dec. 2008
 modified 9 Apr 2012
 by Tom Igoe
 
 This example code is in the public domain.
 
 */

// These constants won't change.  They're used to give names
// to the pins used:
const int analogInPinRight = A0;  // Analog input pin that the potentiometer is attached to
const int analogInPinLeft = A1;  // Analog input pin that the potentiometer is attached to

int sensorValueRight = 0;        // value read from the pot
int sensorValueLeft = 0;        // value read from the pot


int outputValueRight = 0;
int outputValueLeft = 0;

void setup() {
  // initialize serial communications at 9600 bps:
  Serial.begin(9600); 
}

void loop() {
  // read the analog in value:
  sensorValueRight = analogRead(analogInPinRight);            
  sensorValueLeft = analogRead(analogInPinLeft);            
  // map it to the range of the analog out:
  sensorValueRight = sensorValueRight - 371;
  outputValueRight = map(sensorValueRight, -1023, 1023, -16384, 16384);  
  // change the analog out value:
  

  // print the results to the serial monitor:                    
  Serial.println(sensorValueRight);     

  // wait 2 milliseconds before the next loop
  // for the analog-to-digital converter to settle
  // after the last reading:
  delayMicroseconds(125); //8000 hz sample rate
}
