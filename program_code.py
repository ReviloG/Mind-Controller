from tkinter import *
import numpy as np
import random

#Define an array of 100 values, all of them are 0 for now.
#Later in the code, n_tests will be used for a for loop. This loop will run that many times.
n_tests = 100  

# Define the function that will determine how the slider and button function.
# Depending on the value of x it will change the sliders position or switch the button between on and off.
def send_to_program(x):
	slide = w.get()
	
	#If the value inputted is "U", shift the slider up and print the new value.
	if(x == 'U'):		
		if(slide < 10):
			slide = slide + 1
			w.set(slide)
		print('slider value: ', slide,'. Letter entered: ', x)
		
	#If the value inputted is "D", shift the slider down and print the new value.	
	elif(x == 'D'):
		if(slide > -10):
			slide = slide - 1
			w.set(slide)
		print('slider value: ', slide,'. Letter entered: ', x)
	
	#If the value inputted is "B", switch the button between on and off depending on it's current value.
	elif(x == 'B'):
		#If the current value of the button is off, change it to on and print its current status (on).
		if(b['text'] == 'Off'):
			b['text'] = 'On'
			print('button status: On')
		else:
			#If the current value of the button is on, change it to off and print its current status (off).
			b['text'] = 'Off'
			print('button status: Off')
			
	#If the value inputted is invalid (not "B", "D", or "U") then print "Invalid input: nothing was done".
	else:
		print('Nothing was done')

#make a tkinter element named master
#master will hold elements w (scale) and b (button)
#Scale will range between -10 and 10, the button will start set to off
master = Tk()		
w = Scale(master, from_=10, to=-10, width=15)
b = Button(master, text = 'Off')
master.title("slider")

#A for loop that will run 100 times
for i in range(n_tests):
	#If i == 0, tell the program that it is not a valid input
	if(i == 0):
		send_to_program('Invalid input')
	#If the value of i is evenly divisable by 4, then select a random number between 1 and 3
	elif(i%4 == 0):
		rannum = random.randint(1,3)
		#If the random number selected is 1, send "U" to the program
		if(rannum == 1):
			send_to_program('U')
		#If the random number selected is 2, send "D" to the program	
		elif(rannum == 2):
			send_to_program('D')
		#If the random number selected is 3, send "B" to the program	
		elif(rannum == 3):
			send_to_program('B')
	#If i%4 not equal to 0, tell the program that the input is invalid
	else:
		send_to_program('Invalid input')
		
w.pack()
b.pack()
mainloop()