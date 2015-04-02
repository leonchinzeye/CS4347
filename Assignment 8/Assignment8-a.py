import math
import numpy
import scipy.signal
import pylab as plot
from scipy.io.wavfile import write as wavwrite

sampling_rate = 44100.0
freq = 1000.0
midi_note = 60
amplitude = 0.5
duration = 1.0

def main():
	maxSineWaves = int(math.floor(sampling_rate / 2 / freq))
	sawToothArray = numpy.zeros((maxSineWaves, duration * sampling_rate))

	for i in range(1, maxSineWaves + 1):
		temp = calcSineWave(freq, i, duration)
		sawToothArray[i - 1] = temp

	resultSawtooth = makeSawToothArray(sawToothArray)

	t_array = numpy.arange(duration * sampling_rate)
	perfectSaw = amplitude * scipy.signal.sawtooth(freq / sampling_rate * 2.0 * numpy.pi * t_array)

	wavwrite("part1.wav", sampling_rate, resultSawtooth)
	wavwrite("part2.wav", sampling_rate, perfectSaw)

	# drawTimeDomain(perfectSaw, resultSawtooth, "Part 1a.png", "sawtooth-wave reconstruction with 22 sine waves")
	# drawDBMag(db_spectrum(perfectSaw[0:8192]), db_spectrum(resultSawtooth[0:8192]), "Part 1b.png", "dB-magnitude FFT")

def makeSawToothArray(data):
	totalData = numpy.sum(data, axis = 0)
	return -2 * 0.5 / numpy.pi * totalData

def calcFundamentalFreq(midi):
	if(midi == 0):
		return 0.0;
	else:
		return 440 * pow(2, (midi - 69) / 12.0)

def calcSineWave(freq, k, duration):
	sineArray = numpy.arange(int(duration * sampling_rate))
	sineArray = pow(k, -1) * numpy.sin(k * 2.0 * numpy.pi * (freq / sampling_rate) * sineArray)
	return sineArray

def db_spectrum(data):
	window = numpy.blackman(len(data))
	fft = numpy.fft.fft(data * window)
	fft = fft[:len(fft) / 2 + 1]
	magfft = abs(fft) / (numpy.sum(window) / 2.0)
	epsilon = 1e-10
	db = 20 * numpy.log10(magfft + epsilon)
	return db

def drawTimeDomain(data1, data2, fileName, title):
	plot.figure(figsize = (18, 8))
	plot.title(title)
	plot.xlabel("Samples")
	plot.ylabel("Amplitude")
	plot.plot(data1, label = "perfect sawtooth")
	plot.plot(data2, label = "reconstructed")
	plot.legend(loc = 1)
	plot.xlim(0, 240)
	plot.ylim(-0.8, 0.8)
	plot.savefig(fileName)

def drawDBMag(data1, data2, fileName, title):
	plot.figure(figsize = (18, 8))
	plot.title(title)
	plot.xlabel("FFT bin")
	plot.ylabel("dB")
	plot.plot(data1, label = "perfect sawtooth")
	plot.plot(data2, label = "reconstructed")
	plot.legend(loc = 1)
	plot.xlim(0, 4096)
	plot.savefig(fileName)

main()