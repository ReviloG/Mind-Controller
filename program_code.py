from tkinter import *
import numpy as np
import random
    
n_tests = 20  #The number of tests you can run.
#zero_prob = 0.5  #The probability of getting a zero in the list
#other_prob = (1-zero_prob) / 3  #The other probabilities for the non-zeros
#test_commands = np.random.choice([0,1,2,3], n_tests,
#p = [zero_prob, other_prob, other_prob, other_prob])
test_commands = [0] * n_tests

def send_to_program(x):
	slide = w.get()
	if(x == 'U'):		
		if(slide < 100):
			slide = slide + 1
			w.set(slide)
		print('slider value: ', slide,'. Letter entered: ', x)
	elif(x == 'D'):
		if(slide > 0):
			slide = slide - 1
			w.set(slide)
		print('slider value: ', slide,'. Letter entered: ', x)
	elif(x == 'B'):
		if(b['text'] == 'Off'):
			b['text'] = 'On'
			print('button status: On')
		else:
			b['text'] = 'Off'
			print('button status: Off')
	else:
		print('Do Nothing')

master = Tk()		
w = Scale(master, from_=100, to=0)
b = Button(master, text = 'Off')
master.title("slider")

for i in range(n_tests):
	if(i == 0):
		send_to_program('Do Nothing')
	elif(i%4 == 0):
		rannum = random.randint(1,3)
		if(rannum == 1):
			send_to_program('U')
		elif(rannum == 2):
			send_to_program('D')
		elif(rannum == 3):
			send_to_program('B')
	else:
		send_to_program('Do Nothing')
		
w.pack()
b.pack()
mainloop()

