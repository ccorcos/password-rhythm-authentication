#!/usr/bin/env python
# coding=utf-8

# Chet Corcos on 2/28/14.

# import argparse
import curses
import os
import sys
import signal

import time
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import colorsys

import itertools
import math

tol = 0.95


window = curses.initscr() # initialize key capture
def initCurses():
	curses.cbreak() # dont wait for <enter>
	window.erase()
	window.refresh()
def nextLine():
	y,x = window.getyx()
	window.move(y+1,0)
def println(string):
	y,x = window.getyx()
	window.addstr(y,0, string)
	nextLine()
def reprint(string):
	y,x = window.getyx()
	window.move(y,0)
	window.clrtoeol()
	window.addstr(y,0, string)
def clearln():
	y,x = window.getyx()
	window.move(y,0)
	window.clrtoeol()
def moveUp():
	y,x = window.getyx()
	window.move(y-1,0)
def moveDown():
	y,x = window.getyx()
	window.move(y+1,0)
def clearUp(n):
	clearln()
	for i in range(n-1):
		moveUp()
		clearln()
def clearUpTo(y):
	clearln()
	yi,x = window.getyx()
	while yi > y:
		moveUp()
		clearln()
		yi = yi-1
		window.move(yi,0)
def debugPrint(a, s=''):
	y,x = window.getyx()
	nextLine()
	println("debug: "+s)
	println(str(a))
	println("<enter> to continue")
	nextLine()
	while True:
		key = window.getch() 	# wait for a key press
		if key == 27 or key == 10: # escape to finish
			break
	clearUpTo(y)
def wait():
	println("<enter> to continue...")
	while True:
		key = window.getch() 	# wait for a key press
		if key == 27 or key == 10: # escape to finish
			clearUp(2)
			break
def yesno(a):
	println(str(a))
	reprint("[y/n]:")
	while True:
		key = window.getch() 	# wait for a key press
		if key == ord('y'):
			clearUp(2)
			return True
		elif key == ord('n'):
			clearUp(2)
			return False
		else:
			reprint("[y/n]:")


def record():
	println("Recording:")
	println("Type a password 10+ times with the same rhythm.")
	println("    Press <enter> after each.")
	println("    Press <delete> to reset your typing.")
	println("    Press <escape> when youre finished.")
	nextLine()
	println("Password: ???")
	println("Training Samples: ???")
	nextLine()

	password = None
	timings = []

	def updateTrainInfo():
		moveUp()
		moveUp()
		moveUp()
		reprint("Password: " + password)
		moveDown()
		reprint("Training Samples: " + str(len(timings)))
		moveDown()
		moveDown()

	t = []
	p = ''

	plt.ion()
	while True:
		key = window.getch() 	# wait for a key press

		if key == 27: # escape to finish
			clearln()
			break

		elif key == 127: # delete
			clearln()
			p = ''
			t = []

		elif key == 10: # return
			clearln()
			if password == None:
				password = p

			if password == p:
				timings.append([ti-t[0] for ti in t]) # remove the 0 time
				updateTrainInfo()

				plt.clf()
				plt.grid()
				for i in range(len(timings)):
					plt.scatter(timings[i],[i for z in timings[i]])
				plt.draw()

			t = []
			p = ''

		else:
			t.append(time.time())
			p += chr(key)

	plt.ioff()
	plt.close()
	clearUp(10)
	return password, timings


def train(password, timings, name):
	data = np.array(timings)
	data = np.diff(data)

	n,d = data.shape

	m = np.mean(data,0)								# compute the mean
	cdata = data-m 									# centralize the data
	s = np.cov(cdata, rowvar=0) 					# compute the convariance

	w,v= np.linalg.eig(s) 							# compute eigenvalues and eigenvectors
	T = np.real(np.transpose(v)) 					# transfromation matrix into orthoganol dimensions
	w = abs(w)										# abs(w) for numerical error on VERY fast typing
	stdev = np.sqrt(w)								# compute standard deviation of of each orthoganol basis

	Ti, si = independantFeatures(d,T,stdev)			# compute features based on stdev contribution
	return m, Ti, si
def independantFeatures(d,T,stdev):
	stdevp = stdev/sum(stdev)
	stdevpmax = 3.0/(d*2.0)
	stdevpmin = 1.0/(d*2.0)

	Ti = []
	si = []
	for i in range(d):
		if stdevp[i] < stdevpmax and stdevp[i] > stdevpmin:
			Ti.append(T[i])
			si.append(stdev[i])
	Ti = np.array(Ti)
	si = np.array(si)

	return Ti, si


def plotTraining(timings, m, T, stdev):	
	reprint("Plotting training data on the model. Exit plot to continue...")
	
	tdata = transformData(timings, m, T)

	plt.ion()
	plotGuassians(T, stdev)
	plotData(T,stdev,tdata)

	plt.grid()										# add a grid to the plot
	plt.title('Training Data')
	plt.draw()

	wait()

	plt.ioff()
	plt.close()
	clearln()
def plotGuassians(T,stdev):
	di = T.shape[0]
	x = xSpace(stdev)

	color = get_color(di) 							# generate different colors for plotting each dimension
	for j in range(di):								# plot a gaussian for each orthoganol basis
		acolor = next(color)
		plt.plot(x,stats.norm.pdf(x,loc=0,scale=stdev[j]), color=acolor, zorder=1)
def plotData(T,stdev,tdata):
	di = T.shape[0]
	x = xSpace(stdev)

	# plot one sample
	if len(tdata.shape) == 1:
		d = tdata.shape[0]
		
		color = get_color(di)
		for j in range(di):								# plot a gaussian for each orthoganol basis
			acolor = next(color)

			# check if each dimension passes the test, plot an o or x for each dimension
			sampleDimension = tdata[j]
			sampleCdf = stats.norm.cdf(sampleDimension,loc=0,scale=stdev[j])
			samplePdf = stats.norm.pdf(sampleDimension,loc=0,scale=stdev[j])
			CI = abs(sampleCdf - 0.5)*2.0
			if (CI < tol):
				plt.scatter(sampleDimension, samplePdf, color=acolor,  marker='o', zorder=2, s=50)
			else:
				plt.scatter(sampleDimension, samplePdf, color=acolor, marker='x', zorder=2, s=50, linewidth=3)
	
	# plot multiple samples
	else:
		n,d = tdata.shape
		
		color = get_color(di) 							# generate different colors for plotting each dimension
		for j in range(di):								# plot a gaussian for each orthoganol basis
			acolor = next(color)
			for i in range(n):
				# check if each dimension passes the test, plot an o or x for each dimension
				sampleDimension = tdata[i][j]
				sampleCdf = stats.norm.cdf(sampleDimension,loc=0,scale=stdev[j])
				samplePdf = stats.norm.pdf(sampleDimension,loc=0,scale=stdev[j])
				CI = abs(sampleCdf - 0.5)*2.0
				if (CI < tol):
					plt.scatter(sampleDimension, samplePdf, color=acolor,  marker='o', zorder=2, s=50)
				else:
					plt.scatter(sampleDimension, samplePdf, color=acolor, marker='x', zorder=2, s=50, linewidth=3)
def xSpace(stdev):
	l = 4.0*max(stdev) 								# plot with 4 standard deviations of the largest gaussian
	x = np.linspace(-l,l,1000)
	return x
def get_color(color):
    for hue in range(color):
        hue = 1. * hue / color
        col = [int(x) for x in colorsys.hsv_to_rgb(hue, 1.0, 230)]
        yield "#{0:02x}{1:02x}{2:02x}".format(*col)


def transformData(data, m, T):
	data = np.array(data)
	data = np.diff(data)
	cdata = data-m						# centralize the data
	tdata = np.inner(cdata, T)			# transform data into orthoganal basis with 0 mean
	return tdata
def confidence(tsample,T,stdev):
	cdf = np.array([stats.norm.cdf(d,loc=0,scale=s) for d,s in zip(tsample,stdev)])
	CI = abs(cdf - 0.5)*2.0 # confidence interval
	conf = 1.0 - CI 		# confidence -- 10% confidence is a 90% confidence interval by this lingo
	return max(CI)
def authenicated(tsample,T,stdev):
	if confidence(tsample,T,stdev) < tol:
		return 1
	else:
		return 0


def save(name, password, timings, m, T, stdev):
	a = {}
	a['password'] = password
	a['timings'] = timings							# save the training data
	a['m'] = m.tolist()								# save the mean to center new samples
	a['T'] = T.tolist()								# save the transformation to transform new samples
	a['stdev'] = stdev.tolist()						# save the standard deviations of the dimensions to test new samples
	pickle.dump( a, open( name, "wb" ) )			# save the above
def load(name):
	a = pickle.load( open(name, "rb" ) )		# Load the saved data
	password = a['password']
	timings = a['timings']							# save the training data
	m = np.array(a['m'])							# save the mean to center new samples
	T = np.array(a['T'])							# save the transformation to transform new samples
	stdev = np.array(a['stdev'])					# save the standard deviations of the dimensions to test new samples
	return password, timings, m, T, stdev


def test(password, m, T, stdev):
	println("Testing:")
	println("Type the password noting with the appropriate rhythm: " + password)
	println("    Press <enter> after typing the password")
	println("    Press <delete> to reset your typing.")
	println("    Press <escape> when youre finished.")
	nextLine()

	t = []
	p = ''

	plt.ion()
	while True:
		key = window.getch() 	# wait for a key press

		if key == 27: # escape to finish
			clearln()
			break

		elif key == 127: # delete
			clearln()
			p = ''
			t = []

		elif key == 10: # return
			clearln()
			if password == p:

				timing = [ti-t[0] for ti in t] # remove the 0 time
				tsample = transformData(timing, m, T)
				conf = confidence(tsample,T,stdev)

				plt.clf()
				plt.grid()
				if conf < tol:
					plt.title('PASS: CI='+str(conf))
				else:
					plt.title('FAIL: CI='+str(conf))
				plotGuassians(T,stdev)
				plotData(T,stdev,tsample)
				plt.draw()

			t = []
			p = ''

		else:
			t.append(time.time())
			p += chr(key)

	plt.ioff()
	plt.close()
	clearUp(6)


# handle control-c to reset window
def ctrlC(signal, frame):
    curses.endwin()
    sys.exit(0)
signal.signal(signal.SIGINT, ctrlC)
def getFileName():
	println("Save filename:")
	name = window.getstr()
	name = "passwords/" + name + ".p"
	return name

def main():
	name = getFileName()
	initCurses()
	println("Password Rhythm Authenticator")
	nextLine()
	
	if not os.path.exists('passwords/'):						# create the directory to save the figure and the data
	    os.makedirs('passwords/')

	# check if the file exists
	if os.path.isfile(name):
		password, timings, m, T, stdev = load(name)

		if yesno("Would you like to view the training data?"):
			plotTraining(timings, m, T, stdev)

		# if yesno("Would you like to retraining the data?"): # for debugging
		# 	m, T, stdev = train(password, timings, name)
		# 	save(name, password, timings, m, T, stdev)
		# 	plotTraining(timings, m, T, stdev)

		test(password, m, T, stdev)

	# train a model
	else:
		password, timings = record()
		m, T, stdev = train(password, timings, name)
		save(name, password, timings, m, T, stdev)
		plotTraining(timings, m, T, stdev)

		test(password, m, T, stdev)
		
	curses.endwin()
	return 0

# Execute main if called from the command line.
if __name__ == "__main__":
	main()