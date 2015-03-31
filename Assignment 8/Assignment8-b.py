import math
import numpy as np
import scipy.signal
import pylab as plot
from scipy.io.wavfile import write as wavwrite

fs = 44100.0

freq1 = 100.0
freq2 = 1234.56
size1 = 16384
size2 = 2048

duration = 1.0

def main():
	freq1_vanilla_size1 = useLutGetWaves(size1, freq1)
	freq2_vanilla_size1 = useLutGetWaves(size1, freq2)

	freq1_vanilla_size2 = useLutGetWaves(size2, freq1)
	freq2_vanilla_size2 = useLutGetWaves(size2, freq2)

	perfect_sine_wave_freq1 = calcSineWave(freq1)
	perfect_sine_wave_freq2 = calcSineWave(freq2)

	plotWave(freq1_vanilla_size1, "100.0Hz - LUT size: 16384", "100.0Hz - LUT size: 16384.png")
	plotWave(freq1_vanilla_size2, "100.0Hz - LUT size: 2048", "100.0Hz - LUT size: 2048.png")

	plotWave(freq2_vanilla_size1, "1234.56Hz - LUT size: 16384", "1234.56Hz - LUT size: 16384.png")
	plotWave(freq2_vanilla_size2, "1234.56Hz - LUT size: 2048", "1234.56Hz - LUT size: 2048.png")

def plotWave(data, title, fileName):
	plot.figure(figsize = (18, 8))
	plot.title(title)
	plot.xlabel("Time(Samples)")
	plot.ylabel("Amplitude")
	plot.plot(data, label = "no interpolation")
	plot.xlim(0, 240)
	plot.legend(loc = 1)
	plot.savefig(fileName)

def useLutGetWaves(sample_size, freq):
	lut = np.zeros(sample_size)
	lut = createLookUpTable(sample_size)

	delta_phi = freq / fs * sample_size

	return genBufferOutputVanilla(sample_size, delta_phi, lut)

def genBufferOutputVanilla(sample_size, delta_phi, lut):
	phase = 0.0
	buff = np.zeros(sample_size)

	for i in range(sample_size):
		buff[i] = lut[int(phase)]
		phase += delta_phi

		if phase >= sample_size:
			phase %= sample_size

	return buff

def genBufferOutputLinear(sample_size, phase_increment, lut):
	result = numpy.zeros(0)
	
	for i in range(sample_size):
		x0 = np.floor(i * phase_increment)
		x1 = (x0 + 1) % len(lut)
		y0 = lut(x0)
		y1 = lut(x1)

		result = np.concatenate((result, y0 + (y1 - y0) * (phase_increment - x0) / (x1 - x0)))

	return result

def createLookUpTable(sample_size):
	array = np.arange(sample_size)
	lut = np.sin(2.0 * np.pi * array / sample_size)

	return lut

def calcSineWave(freq, duration = duration, amplitude = 1.0, phase = 0.0):
	sineArray = np.arange(int(duration * sampling_rate))
	sineArray = amplitude * np.sin(phase + 2.0 * numpy.pi * (freq / fs) * sineArray)
	return sineArray

main()