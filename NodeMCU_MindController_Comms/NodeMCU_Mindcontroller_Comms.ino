/* 
*  Mindcontroller Firmware
*  AJ Casapulla
*  Uses ESP8266 on a NodeMCU to communicate with the ADS1299 on the MindController board via SPI.
*  
*  Pinout:
*  MindController | NodeMCU
*  DRDY           | D1 / GPIO5
*  DOUT           | D6 / GPIO12 / MISO
*  SCLK           | D5 / GPIO14 / SCLK 
*  CS             | D8 / GPIO15 / CS      This is predefined as SS
*  START          | D2 / GPIO4
*  RST            | D3 / GPIO0
*  DIN            | D7 / GPIO13 / MOSI
*  
*  ADS1299 SPI info:
*  CPOL = 0
*  CPHA = 1
*  Minimum speed is determined by sample rate
*  Maxiumum speed is 20Mhz
*  
 */

#include <SPI.h>

uint8_t mc_drdy = 5;
uint8_t mc_start = 4;
uint8_t mc_reset = 0;

SPISettings settings(4000000, MSBFIRST, SPI_MODE1); //CPOL 0, CPHA 1

byte readRegister(int r){  
  byte b1 = (0x02 << 4) + r;
  byte b2 = 0x00;

  SPI.beginTransaction(settings);
  digitalWrite(SS, LOW);
  SPI.transfer(b1);
  SPI.transfer(b2);
  byte response = SPI.transfer(0x00);  
  digitalWrite(SS, HIGH);
  SPI.endTransaction();
  
  return response;
}

// RDC is 'Read Data Continuously'
// The ADS starts up already reading data. This has to be stopped to read/write any registers.
void stopRDC(){
  SPI.beginTransaction(settings);
  digitalWrite(SS, LOW);
  SPI.transfer(17);
  digitalWrite(SS, HIGH);
  SPI.endTransaction();
}

void setup() {
  SPI.begin();

  // Pin setups
  pinMode(SS, OUTPUT);
  digitalWrite(SS, HIGH);

  pinMode(mc_drdy, INPUT);
  
  pinMode(mc_start, OUTPUT);
  digitalWrite(mc_start, LOW);
  
  pinMode(mc_reset, OUTPUT);
  digitalWrite(mc_reset, HIGH);
  // End pin setup

  Serial.begin(115200);
  Serial.println();
  Serial.println("-------------------------------------------------------------------------------------------");
  Serial.println();
  Serial.println("Resetting ADS1299...");
  digitalWrite(mc_reset, LOW);
  delay(2); // = 2 / 2,048,000
  digitalWrite(mc_reset, HIGH);
  delay(100);
  
  Serial.println("Starting in 3..");
  delay(1000);
  Serial.println("2...");
  delay(1000);
  Serial.println("1...");
  delay(1000);
  Serial.println("Reading ID.");

  stopRDC(); //SPI won't respond unless you tell it to stop reading data continuously first

  // Try to get some data
  Serial.print("Response: ");
  Serial.println(readRegister(1), BIN);
}

void loop() {
  // put your main code here, to run repeatedly:

}
