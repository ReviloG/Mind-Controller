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

#include <ESP8266WiFi.h>
#include <SPI.h>
#include <Plotter.h>

#define PLOTMODE true

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

// Pin assignments
uint8_t mc_drdy = 5;
uint8_t mc_start = 4;
uint8_t mc_reset = 0;

SPISettings settings(4000000, MSBFIRST, SPI_MODE1); //CPOL 0, CPHA 1

struct int24_t {
  signed int data : 24;
} __attribute__((packed));

typedef struct {
  int24_t statusReg;
  int24_t data[8];
} rawADSData;

typedef struct {
  int32_t statusReg;
  int32_t ch[8]; 
} ADSData;

union paddedADSData{
  int32_t data32;
  int24_t data24;
  uint8_t dataAry[4];
};

bool dataReady = false;
ADSData data;
Plotter p;
int32_t plotCh[8];
int32_t graphMin = 0;
int32_t graphMax = 2000000;

// Need to convert the 24bit data to 32bit for usability
ADSData processADSData(rawADSData raw){
  ADSData data;
  uint8_t temp;
  
  if (!PLOTMODE) Serial.print("Raw: ");
  
  for (int i=0; i<8; i++){
  
    paddedADSData a;
    a.data32 = 0;
    a.data24 = raw.data[i];
    temp = a.dataAry[0];
    a.dataAry[0] = a.dataAry[2];
    a.dataAry[2] = temp;
    if (!PLOTMODE){
      Serial.print(a.data32,HEX);
      Serial.print(' ');
    }
    
    data.ch[i] = ((a.data32 << 8) >> 8);
  }
  if (!PLOTMODE) Serial.println();
  return data;
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
  byte b1 = 0x40 + r;
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
  digitalWrite(SS,HIGH);
  SPI.endTransaction();

  if(!PLOTMODE){
    for (int i=0;i<27;i++){
      Serial.print(dbuffer[i],HEX);
      Serial.print(' ');
    }
    Serial.println();
  }
  
  return *(rawADSData*)dbuffer;
}

void readLoop(){
  if(dataReady){
      data = processADSData(readData());
      if(PLOTMODE){
        p.Plot();
      }
    }
    yield();
}

void setup() {
  SPI.begin();

  if (PLOTMODE){
    p.Begin();
    p.AddTimeGraph("MindController", 2000, "Ch1", data.ch[0]); //, "Ch2", data.ch[1], "Ch3", data.ch[2], "Ch4", data.ch[3]); //, "Ch5", data.ch[4], "Ch6", data.ch[5]);
  } else {
    Serial.begin(115200);
    Serial.println();
    Serial.println("-------------------------------------------------------------------------------------------");
    Serial.println();
    Serial.println("Started.");
  }
  // Pin setups
  pinMode(SS, OUTPUT);
  digitalWrite(SS, HIGH);

  pinMode(mc_drdy, INPUT);
    
  pinMode(mc_start, OUTPUT);
  digitalWrite(mc_start, LOW);
  
  pinMode(mc_reset, OUTPUT);
  digitalWrite(mc_reset, HIGH);
  // End pin setup

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

  // Set to internal BIASREF
  writeRegister(CONFIG3, 0xE0);

  // Set data rate to fmod / 4098 = 250Samples / second
  writeRegister(CONFIG1, 0x96);
  Serial.print("CONFIG1 (0x96): ");
  Serial.println(readRegister(CONFIG1), HEX);

  // Ensure the test signal is turned off
  writeRegister(CONFIG2, 0xC0);
  Serial.print("CONFIG2 (0xC0): ");
  Serial.println(readRegister(CONFIG2), HEX);

  // Set all channels to input short for noise measurements
  for (int i=CH1SET; i<=CH8SET; i++){
    writeRegister(i, 0x01);
  }
  Serial.print("CH1SET (0x51): ");
  Serial.println(readRegister(CH1SET), HEX);

  // start data conversion
  sendCmd(START);

  // read data continuous
  attachInterrupt(digitalPinToInterrupt(mc_drdy), drdy_handler, FALLING);
  sendCmd(RDATAC);

  // read data for 5 seconds to see baseline noise/offset
  uint32_t startTime = millis();
  while (millis()-startTime < 5000) {
      readLoop();
  }

  // Stop reading data, turn on test signal for all channels, start again.
  sendCmd(SDATAC);
  if (dataReady) readData();

  writeRegister(CONFIG2, 0xD4);
  for (int i=CH1SET; i<=CH8SET; i++){
    writeRegister(i, 0x05);
  }
 
  sendCmd(RDATAC);

  // Run test signal for 5 seconds
  startTime = millis();
  while(millis()-startTime < 5000){
    readLoop();
  }

  sendCmd(SDATAC);
  if (dataReady) readData();

  // SET UP FOR REAL READINGS 
  
  writeRegister(CONFIG2, 0xC0); // Turn off test signal 
  writeRegister(CONFIG3, 0xEC); // Turn on the internal reference buffer & bias buffer

  for (int i=CH1SET; i<=CH1SET; i++){
    writeRegister(i, 0x50);  // Set ch1 to normal operation, gain = 12
  }

  for (int i=CH2SET; i<=CH8SET; i++){
    writeRegister(i, 0x81); // Turn off channels 2-8
  }

  writeRegister(BIAS_SENSP, 0x01);  // Connect 1-4 to the bias derivation
  writeRegister(BIAS_SENSN, 0x01);

  writeRegister(MISC1, 0x20); // Ties all the negative electrode inputs to the reference

  sendCmd(RDATAC);
  
}

void loop() {
  readLoop();
}
