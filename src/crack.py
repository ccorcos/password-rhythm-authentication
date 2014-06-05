#!/usr/bin/env python
# coding=utf-8

# Chet Corcos on 2/28/14.

# from __future__ import print_function

import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import colorsys
import itertools
import math
import multiprocessing
import time



def load(name):
	a = pickle.load( open(name, "rb" ) )		# Load the saved data
	password = a['password']
	timings = a['timings']							# save the training data
	m = np.array(a['m'])							# save the mean to center new samples
	T = np.array(a['T'])							# save the transformation to transform new samples
	stdev = np.array(a['stdev'])					# save the standard deviations of the dimensions to test new samples
	return password, timings, m, T, stdev


tol = 0.95
name = "passwords/goalie12.p"
pwd, timings, m, T, stdev = load(name)

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
	cdf = np.array([stats.norm.cdf(d,loc=0,scale=s) for d,s in zip(tsample,stdev)]) # transformed sample
	CI = abs(cdf - 0.5)*2.0 # confidence interval
	conf = 1.0 - CI 		# confidence -- 10% confidence is a 90% confidence interval by this lingo
	return max(CI)
def authenicated(tsample,T,stdev):
	if confidence(tsample,T,stdev) < tol:
		return 1
	else:
		return 0

def f(comb):
	c = np.array(list(comb))
	sample = [0]+[sum(c[:i+1]) for i in range(len(c))]
	tdata = transformData(sample, m, T)
	if authenicated(tdata,T,stdev):
		return sample


x = np.linspace(.02,0.2,10)
d = len(m)

combs = list(itertools.product(x, repeat=d))

t = time.clock()
p = multiprocessing.Pool(8)
worked = p.map(f,combs)
print "this process took: " + str(time.clock() - t) + " seconds"

w = filter(lambda x: x != None, worked)
worked = w

pickle.dump( worked, open( "crackData.p", "wb" ) )			# save the above


n = len(worked)
worked = np.array(worked)
print str(n) + "out of " + str(len(x)**d)
print str(n/float(len(x)**d))

# 10 / 10e6 = 1e-06

plt.grid()
plt.title('Valid rhythms for "' + pwd + '", '+str(n/len(x)**d)	+'%% worked')
color = get_color(n)
for i in range(len(worked)):
	acolor = next(color)
	plt.scatter(worked[i],[i for z in worked[i]], color=acolor)
plt.savefig('cracks.png')
plt.show()



## Just this part for plotting later

# print "loading"
# worked = pickle.load( open("crackData.p", "rb" ) )		# Load the saved data
# print "loaded"

# x = np.linspace(.2,2,10)
# d = len(m)

# n = len(worked)
# worked = np.array(worked)
# print str(n) + " out of " + str(len(x)**d)
# print str(n/float(len(x)**d))


# plt.grid()
# plt.title('Valid rhythms for "' + pwd + '", '+str(float(n)/float(len(x)**d)*100.0)	+r'% worked')
# color = get_color(20)
# for i in range(0,len(worked), int(round(float(n)/20.0))):
# 	print n-i
# 	acolor = next(color)
# 	plt.scatter(worked[i],[i for z in worked[i]], color=acolor)
# plt.show()

# plt.savefig('cracks.png', dpi = 600)
# plt.savefig('cracks.svg')


