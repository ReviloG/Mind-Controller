Code for using a NodeMCU to talk to the MindController. Made & compiled in Arduino IDE.

MCU used: https://www.amazon.com/gp/product/B07HF44GBT

Configure Adruino IDE to use esp8266 by ESP8266 Community from:
http://arduino.esp8266.com/stable/package_esp8266com_index.json
Current version is 2.6.3

Set up your chip:

Tools -> Board -> NodeMCU 1.0 (ESP-12E Module)

Tools -> Flash Size -> 4M (3M SPIFFS)

Tools -> CPU Frequency -> 80 Mhz

Tools -> Upload Speed -> 921600

Flashed the MCU with 64 bit flasher exe at: github.com/nodemcu/nodemcu-flasher/tree/master/Win64/Release
