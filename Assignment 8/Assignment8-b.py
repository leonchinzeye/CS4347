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

	freq1_linear_size1 = useLutGetLinear(size1, freq1)
	freq2_linear_size1 = useLutGetLinear(size1, freq2)

	freq1_linear_size2 = useLutGetLinear(size2, freq1)
	freq2_linear_size2 = useLutGetLinear(size2, freq2)

	perfect_sine_wave_freq1 = calcSineWave(freq1)
	perfect_sine_wave_freq2 = calcSineWave(freq2)

	err1 = np.max(np.abs(freq1_vanilla_size1 - perfect_sine_wave_freq1[:size1]))
	err2 = np.max(np.abs(freq1_vanilla_size2 - perfect_sine_wave_freq1[:size2]))
	err3 = np.max(np.abs(freq1_linear_size1 - perfect_sine_wave_freq1[:size1]))
	err4 = np.max(np.abs(freq1_linear_size2 - perfect_sine_wave_freq1[:size2]))

	err5 = np.max(np.abs(freq2_vanilla_size1 - perfect_sine_wave_freq2[:size1]))
	err6 = np.max(np.abs(freq2_vanilla_size2 - perfect_sine_wave_freq2[:size2]))
	err7 = np.max(np.abs(freq2_linear_size1 - perfect_sine_wave_freq2[:size1]))
	err8 = np.max(np.abs(freq2_linear_size2 - perfect_sine_wave_freq2[:size2]))

	err1 = 32767 * err1
	err2 = 32767 * err2
	err3 = 32767 * err3
	err4 = 32767 * err4
	err5 = 32767 * err5
	err6 = 32767 * err6
	err7 = 32767 * err7
	err8 = 32767 * err8

	writeFile(err1, err2, err3, err4, err5, err6, err7, err8)
	# plotWave(freq1_vanilla_size1, freq1_linear_size1, perfect_sine_wave_freq1, "100.0Hz - LUT size = 16384", "100.0Hz - LUT size = 16384.png")
	# plotWave(freq1_vanilla_size2, freq1_linear_size2, perfect_sine_wave_freq1, "100.0Hz - LUT size = 2048", "100.0Hz - LUT size = 2048.png")

	# plotWave(freq2_vanilla_size1, freq2_linear_size1, perfect_sine_wave_freq2, "1234.56Hz - LUT size = 16384", "1234.56Hz - LUT size = 16384.png")
	# plotWave(freq2_vanilla_size2, freq2_linear_size2, perfect_sine_wave_freq2, "1234.56Hz - LUT size = 2048", "1234.56Hz - LUT size = 2048.png")

def writeFile(err1, err2, err3, err4, err5, err6, err7, err8):
	newFile = open("max_audio_file_error.txt", "w")
	newFile.write("Frequency\tInterpolation\t16384-sample\t\t\t2048-sample\n")
	newFile.write("100Hz\t\tNo\t\t\t\t" + str(err1) + "\t\t\t" + str(err2) + "\n")
	newFile.write("\t\t\tLinear\t\t\t" + str(err3) + "\t\t" + str(err4) + "\n")
	newFile.write("1234.56Hz\tNo\t\t\t\t" + str(err5) + "\t\t\t" + str(err6) + "\n")
	newFile.write("\t\t\tLinear\t\t\t" + str(err7) + "\t\t" + str(err8) + "\n")

def plotWave(data1, data2, data3, title, fileName):
	plot.figure(figsize = (18, 8))
	plot.title(title)
	plot.xlabel("Time(Samples)")
	plot.ylabel("Amplitude")
	plot.plot(data1, label = "no linear")
	plot.plot(data2, label = "with linear")
	plot.plot(data3, label = "perfect")
	plot.xlim(0, 240)
	plot.legend(loc = 1)
	plot.savefig(fileName)

def useLutGetWaves(sample_size, freq):
	lut = np.zeros(sample_size)
	lut = createLookUpTable(sample_size)

	delta_phi = freq / fs * sample_size

	return genBufferOutputVanilla(sample_size, delta_phi, lut)

def useLutGetLinear(sample_size, freq):
	lut = np.zeros(sample_size)
	lut = createLookUpTable(sample_size)

	delta_phi = freq / fs * sample_size

	return genBufferOutputLinear(sample_size, delta_phi, lut)

def genBufferOutputVanilla(sample_size, delta_phi, lut):
	buff = np.zeros(sample_size)

	for i in range(sample_size):
		buff[i] = lut[round(i * delta_phi) % sample_size]

	return buff

def genBufferOutputLinear(sample_size, delta_phi, lut):
	result = np.zeros(0)
	
	for i in range(sample_size):
		x0 = np.floor(i * delta_phi % sample_size)
		x1 = (x0 + 1)
		y0 = lut[x0 % sample_size]
		y1 = lut[x1 % sample_size]

		result = np.append(result, y0 + (y1 - y0) * ((i * delta_phi % sample_size) - x0) / (x1 - x0))

	return result

def createLookUpTable(sample_size):
	array = np.arange(sample_size)
	lut = np.sin(2.0 * np.pi * array / sample_size)

	return lut

def calcSineWave(freq, duration = duration, amplitude = 1.0, phase = 0.0):
	sineArray = np.arange(int(duration * fs))
	sineArray = amplitude * np.sin(phase + 2.0 * np.pi * (freq / fs) * sineArray)
	return sineArray

main()