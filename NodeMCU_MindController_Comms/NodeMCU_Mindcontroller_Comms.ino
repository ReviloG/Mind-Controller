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

// ADS1299 registers
#define ID 0x0
#define CONFIG1 0x1
#define CONFIG2 0x2
#define CONFIG3 0x3
#define LOFF 0x4
#define CH1SET 0x5
#define CH2SET 0x6
#define CH3SET 0x7
#define CH4SET 0x8
#define CH5SET 0x9
#define CH6SET 0xA
#define CH7SET 0xB
#define CH8SET 0xC
#define BIAS_SENSP 0xD
#define BIAS_SENSN 0xE
#define LOFF_SENSP 0xF
#define LOFF_SENSN 0x10
#define LOFF_FLIP 0x11
#define LOFF_STATP 0x12
#define LOFF_STATN 0x13
#define GPIO 0x14
#define MISC1 0x15
#define MISC2 0x16
#define CONFIG4 0x17

// ADS1299 commands
#define WAKEUP 0x02
#define STANDBY 0x04
#define RESET 0x06
#define START 0x08
#define STOP 0x0A
#define RDATAC 0x10 //Start continuous data read
#define SDATAC 0x11 //Stop continuous data read
#define RDATA 0x12

uint8_t mc_drdy = 5;
uint8_t mc_start = 4;
uint8_t mc_reset = 0;

bool dataReady = false;

SPISettings settings(4000000, MSBFIRST, SPI_MODE1); //CPOL 0, CPHA 1

typedef struct {
  uint8_t statusReg[3];
  uint8_t ch1[3];
  uint8_t ch2[3];
  uint8_t ch3[3];
  uint8_t ch4[3];
  uint8_t ch5[3];
  uint8_t ch6[3];
  uint8_t ch7[3];
  uint8_t ch8[3];
} rawADSData;

typedef struct {
  int32_t ch1;
  double ch1p;
  int32_t ch2;
  double ch2p;
  int32_t ch3;
  double ch3p;
  int32_t ch4;
  double ch4p;
  int32_t ch5;
  double ch5p;
  int32_t ch6;
  double ch6p;
  int32_t ch7;
  double ch7p;
  int32_t ch8;
  double ch8p;
} ADSData;

ADSData processADSData(rawADSData raw){
  ADSData d;
  
  int32_t ch1temp = (raw.ch1[0] << 16) + (raw.ch1[1] << 8) + raw.ch1[2];
  if (raw.ch1[0] < 0x80){
    d.ch1 = ch1temp;    
  } else {
    d.ch1 = ch1temp - 0x1000000;
  }
  d.ch1p = (double)d.ch1 / 0xFFFFFF;
  
  return d;
}

byte readRegister(uint8_t r){  
  // b1 = b001r rrrr where r rrrr is the register address
  byte b1 = 0x20 + r;
  // b2 = b000n nnnn where n nnnn is the number of registers to read - 1
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

void writeRegister(uint8_t r, uint8_t data){
  // b1 = b010r rrrr where r rrrr is the register address
  byte b1 = 0x80 + r;
  // b2 = b000n nnnn where n nnnn is the number of registers to write - 1
  byte b2 = 0x00;

  SPI.beginTransaction(settings);
  digitalWrite(SS, LOW);
  SPI.transfer(b1);
  SPI.transfer(b2);
  SPI.transfer(data);
  digitalWrite(SS, HIGH);
  SPI.endTransaction();
}

void sendCmd(uint8_t b){
  SPI.beginTransaction(settings);
  digitalWrite(SS,LOW);
  SPI.transfer(b);
  digitalWrite(SS, HIGH);
  SPI.endTransaction();
}

ICACHE_RAM_ATTR void drdy_handler(){
  dataReady = true;
}

rawADSData readData(){
  dataReady = false;
  uint8_t dbuffer[27] = {0};
  SPI.beginTransaction(settings);
  digitalWrite(SS,LOW);
  SPI.transfer(dbuffer, 27);

//  Serial.println("dbuffer:");
//  for (int i=0;i<9;i++){
//    Serial.print(i);
//    Serial.print(": ");
//    Serial.print(dbuffer[i], HEX);
//    Serial.print(' ');
//    Serial.print(dbuffer[i+1], HEX);
//    Serial.print(' ');
//    Serial.print(dbuffer[i+2], HEX);
//    Serial.println();
//  }
//  Serial.println();

  digitalWrite(SS,HIGH);
  SPI.endTransaction();
  
  return *(rawADSData*)dbuffer;
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

  Serial.begin(500000);
  Serial.println();
  Serial.println("-------------------------------------------------------------------------------------------");
  Serial.println();
  Serial.println("Started.");

  // Reset the ADS1299
  digitalWrite(mc_reset, LOW);
  delay(2);
  digitalWrite(mc_reset, HIGH);
  delay(100);

  // Stop continuous data reads
  sendCmd(SDATAC);

  // Get ID register info, just to make sure the ADS is alive.
  int id = readRegister(ID);
  Serial.print("ID Register: ");
  Serial.println(id, BIN);

  // Set to internal VREF
  writeRegister(CONFIG3, 0xE0);

  // Set data rate to fmod / 4096 = 250Samples / second
  writeRegister(CONFIG1, 0x96);

  // Ensure the test signal is turned off
  writeRegister(CONFIG2, 0xC0);

  // Set all channels to input short
  writeRegister(CH1SET, 0x01);
  writeRegister(CH2SET, 0x01);
  writeRegister(CH3SET, 0x01);
  writeRegister(CH4SET, 0x01);
  writeRegister(CH5SET, 0x01);
  writeRegister(CH6SET, 0x01);
  writeRegister(CH7SET, 0x01);
  writeRegister(CH8SET, 0x01);

  // start data conversion
  sendCmd(START);

  // read data continuous
  Serial.println("Starting data collection.");
  attachInterrupt(digitalPinToInterrupt(mc_drdy), drdy_handler, FALLING);
  sendCmd(RDATAC);

  rawADSData dRaw;
  ADSData data;
  uint32_t startTime = millis();
  while (millis()-startTime < 100){
    if(dataReady){
      dRaw = readData();
      //Serial.print(dRaw.ch1[0], HEX);
      //Serial.print(' ');
      //Serial.print(dRaw.ch1[1], HEX);
      //Serial.print(' ');
      //Serial.println(dRaw.ch1[2], HEX);
      data = processADSData(dRaw);
      Serial.println(data.ch1p, 4);
    }
    yield();
  }
  Serial.println("Done collecting");

  detachInterrupt(digitalPinToInterrupt(mc_drdy));

    
}

void loop() {
  // put your main code here, to run repeatedly:

}
