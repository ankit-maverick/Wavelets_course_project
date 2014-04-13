# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import scipy.optimize as optimize
import scipy.fftpack as fftpack
pi = np.pi

# <codecell>

data = np.loadtxt('../data/data.csv', delimiter=',')

# <codecell>

data.shape

# <codecell>

t = np.linspace(0, 336, 337).astype(int)
rcParams['figure.figsize'] = 18, 10
p1, = plot(t, data[:, 0], 'r')
p2, = plot(t, data[:, 1], 'b')
p3, = plot(t, data[:, 2], 'g')
p4, = plot(t, data[:, 3], 'm')
p5, = plot(t, data[:, 4], 'y')
legend([p5, p4, p3, p2, p1], ["Contract 4", "Contract 3", "Contract 2", "Contract 1", "Oil Price"])
plt.xlabel('No. of Months from Jan 1986')
plt.ylabel('Price in USD per Barrel')
plt.title('Crude Oil Price and Futures')
plt.show()

# <codecell>

db7 = pywt.Wavelet('db7')

# <codecell>

def forecast(timeSeries, waveletStr, predictMonths, plotGraphs=True):
	tsLength = len(timeSeries)
	wavelet = pywt.Wavelet(waveletStr)
	
	cA5, cD5, cD4, cD3, cD2, cD1 = pywt.wavedec(timeSeries, wavelet, level=5)
	
	predict = [predictMonths, predictMonths/2, predictMonths/4, predictMonths/8, predictMonths/16, predictMonths/32]
	
	# <codecell>
	if plotGraphs:
		print cA5.shape
		print cD5.shape
		print cD4.shape
		print cD3.shape
		print cD2.shape
		print cD1.shape
	
		# <codecell>
	
		rcParams['figure.figsize'] = 7, 4
		plt.plot(cA5)
		plt.show()
		
		# <codecell>
		
		plt.plot(cD5)
		plt.show()
		
		# <codecell>
		
		plt.plot(cD4)
		plt.show()
		
		# <codecell>
		
		plt.plot(cD3)
		plt.show()
		
		# <codecell>
		
		plt.plot(cD2)
		plt.show()
		
		# <codecell>
		
		plt.plot(cD1)
		plt.show()
	
	# <headingcell level=2>
	
	# Forecasting
	
	# <rawcell>
	
	# Extend the samples of cA5 using an interpolated univariate spline. This, by fitting this spline, we extended the 
	# data to values [0:349] rather than [0:336]. Wavelet reconstruct the signal using the interpolated approx
	
	# <codecell>
	# Sample extension using RBF method. Wavelet reconstruct the signal using the interpolated approx

	t0 = np.linspace(0,tsLength-1,cA5.shape[0]).astype(int)
	if (predict[4] > 0):
		ius_cA5 = InterpolatedUnivariateSpline(t0[:-predict[4]],cA5[:-predict[4]])
		rbf_cA5 = Rbf(t0[:-predict[4]],cA5[:-predict[4]])
	else:
		ius_cA5 = InterpolatedUnivariateSpline(t0,cA5)
		rbf_cA5 = Rbf(t0,cA5)
	ti = np.linspace(0,tsLength-1,cA5.shape[0]).astype(int)
	cA5i1 = ius_cA5(ti)
	datai1 = pywt.waverec([cA5i1, cD5, cD4, cD3, cD2, cD1], waveletStr)

	cA5i2 = rbf_cA5(ti)
	datai2 = pywt.waverec([cA5i2, cD5, cD4, cD3, cD2, cD1], waveletStr)
	
	# <rawcell>
	
	# Comparison of the forecast based only on extrapolation of approximate wavelet decomposition part (IUS and RBF) with the true data
	
	# <codecell>
	if plotGraphs:
		rcParams['figure.figsize'] = 18, 10
		p1, = plot(timeSeries[tsLength-predictMonths-20:], 'r')
		p2, = plot(datai1[tsLength-predictMonths-20:], 'b')
		p3, = plot(datai2[tsLength-predictMonths-20:], 'g')
		p4, = plot(data[tsLength-predictMonths-20:,1], 'y')
	
		legend([p3, p2, p1], ["RBF method", "Univariate Spline", "Oil Price"])
	
		plt.show()
	
	# <rawcell>
	
	# We are going to model the detail part of the wavelet decomposition using a trigonometric fitting/sine fitting. So, first we get the Fourier transform of each of these waveforms
	
	# <codecell>
	
	if plotGraphs:
		rcParams['figure.figsize'] = 7, 4
		
		Cd1 = np.fft.fft(cD1)
		plot(abs(Cd1))
		plt.show()
		
		# <codecell>
		
		Cd2 = np.fft.fft(cD2)
		plot(abs(Cd2))
		plt.show()
		
		# <codecell>
		
		Cd3 = np.fft.fft(cD3)
		plot(abs(Cd3))
		plt.show()
		
		# <codecell>
		
		Cd4 = np.fft.fft(cD4)
		plot(abs(Cd4))
		plt.show()
		
		# <codecell>
		
		Cd5 = np.fft.fft(cD5)
		plot(abs(Cd5))
		plt.show()
	
	# <codecell>
	
	def mysine(x, a1, a2, a3):
	    return a1 * np.sin(a2 * x + a3)
	
	def sinefit(yReal, xReal):
	    yhat = fftpack.rfft(yReal)
	    idx = (yhat**2).argmax()
	    freqs = fftpack.rfftfreq(len(xReal), d = (xReal[1]-xReal[0])/(2*pi))
	    frequency = freqs[idx]
	
	    amplitude = abs(yReal.max())
	    guess = [amplitude, frequency, 0.]
	
	    (amplitude, frequency, phase), pcov = optimize.curve_fit(mysine, xReal, yReal, guess)
	    period = 2*pi/frequency
	    return [amplitude, frequency, phase]
	
	testSig = mysine(np.linspace(0,10,5000), 2, 200, 56)
	
	params = sinefit(testSig, np.linspace(0,10,5000))
	if plotGraphs:
		print params
	
	# <codecell>
	if predict[4] > 0:
		paramsD5 = sinefit(cD5[:-predict[4]], np.linspace(0,tsLength,cD5.shape[0])[:-predict[4]])
		cD5i = np.append(cD5[:-predict[4]],mysine(np.linspace(cD5.shape[0]-predict[4],cD5.shape[0],predict[4]),paramsD5[0], paramsD5[1], paramsD5[2]))
	else:
		paramsD5 = sinefit(cD5, np.linspace(0,tsLength,cD5.shape[0]))
		cD5i = np.append(cD5,mysine(np.linspace(cD5.shape[0],cD5.shape[0],0),paramsD5[0], paramsD5[1], paramsD5[2]))
	if predict[3] > 0:
		paramsD4 = sinefit(cD4[:-predict[3]], np.linspace(0,tsLength,cD4.shape[0])[:-predict[3]])
		cD4i = np.append(cD4[:-predict[3]],mysine(np.linspace(cD4.shape[0]-predict[3],cD4.shape[0],predict[3]),paramsD4[0], paramsD4[1], paramsD4[2]))
	else:
		paramsD4 = sinefit(cD4, np.linspace(0,tsLength,cD4.shape[0]))
		cD4i = np.append(cD4,mysine(np.linspace(cD4.shape[0],cD4.shape[0],0),paramsD4[0], paramsD4[1], paramsD4[2]))
	if predict[2] > 0:
		paramsD3 = sinefit(cD3[:-predict[2]], np.linspace(0,tsLength,cD3.shape[0])[:-predict[2]])
		cD3i = np.append(cD3[:-predict[2]],mysine(np.linspace(cD3.shape[0]-predict[2],cD3.shape[0],predict[2]),paramsD3[0], paramsD3[1], paramsD3[2]))
	else:
		paramsD3 = sinefit(cD3, np.linspace(0,tsLength,cD3.shape[0]))
		cD3i = np.append(cD3,mysine(np.linspace(cD3.shape[0],cD3.shape[0],0),paramsD3[0], paramsD3[1], paramsD3[2]))
	if predict[1] > 0:
		paramsD2 = sinefit(cD2[:-predict[1]], np.linspace(0,tsLength,cD2.shape[0])[:-predict[1]])
		cD2i = np.append(cD2[:-predict[1]],mysine(np.linspace(cD2.shape[0]-predict[1],cD2.shape[0],predict[1]),paramsD2[0], paramsD2[1], paramsD2[2]))
	else:
		paramsD2 = sinefit(cD2, np.linspace(0,tsLength,cD2.shape[0]))
		cD2i = np.append(cD2,mysine(np.linspace(cD2.shape[0],cD2.shape[0],0),paramsD2[0], paramsD2[1], paramsD2[2]))
	if predict[0] > 0:
		paramsD1 = sinefit(cD1[:-predict[0]], np.linspace(0,tsLength,cD1.shape[0])[:-predict[0]])
		cD1i = np.append(cD1[:-predict[0]],mysine(np.linspace(cD1.shape[0]-predict[0],cD1.shape[0],predict[0]),paramsD1[0], paramsD1[1], paramsD1[2]))
	else:
		paramsD1 = sinefit(cD1, np.linspace(0,tsLength,cD1.shape[0]))
		cD1i = np.append(cD1,mysine(np.linspace(cD1.shape[0],cD1.shape[0],0),paramsD1[0], paramsD1[1], paramsD1[2]))
		
	datai1 = pywt.waverec([cA5i1, cD5i, cD4i, cD3i, cD2i, cD1i], waveletStr)
	datai2 = pywt.waverec([cA5i2, cD5i, cD4i, cD3i, cD2i, cD1i], waveletStr)
	
	# <headingcell level=2>
	
	# Forecast using all the interpolated signals
	
	# <codecell>
	if plotGraphs:
		rcParams['figure.figsize'] = 18, 10
		p1, = plot(timeSeries[tsLength-predictMonths-20:], 'r')
		p2, = plot(datai1[tsLength-predictMonths-20:], 'b')
		p3, = plot(datai2[tsLength-predictMonths-20:], 'g')
		p4, = plot(data[tsLength-predictMonths-20:,1], 'y')
		
		legend([p4, p3, p2, p1], ["Contract 1", "RBF method", "Univariate Spline", "Oil Price"])
		
		plt.show()
	return datai2
	
forecast(data[:,0],'db7',32)
	
	
