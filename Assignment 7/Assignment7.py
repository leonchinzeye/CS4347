import math
import numpy
import scipy.fftpack
import pylab as plot
from scipy.io.wavfile import write as wavwrite


midis = [60, 62, 64, 65, 67, 69, 71, 72, 72, 0, 67, 0, 64, 0, 60]
scale = [60, 62, 64, 65, 67, 69, 71, 72, 72, 71, 69, 67, 65, 64, 62, 60, 0, 0]
fireworks = [60, 63, 63, 60, 60, 60, 0, 0, 56, 58, 58, 56, 56, 56, 0, 0, 56, 58, 58, 56, 56, 56,
		 	 0, 56, 56, 56, 58, 60, 60, 60, 0, 0, 
		 	 60, 63, 63, 60, 60, 60, 0, 0, 56, 58, 58, 56, 56, 56, 0, 0, 56, 58, 58, 56, 56, 56,
		 	 0, 56, 56, 56, 58, 60, 60, 60]

duration = 0.25
samp_rate = 16000

def scaleMajorMaker():
	scaleMidi = []
	scaleMidi = scaleMidi + scale
	prev = scale
	for i in range(0, 7):
		prev = scaleIncreaser(prev)
		scaleMidi = scaleMidi + prev

	return scaleMidi

def scaleIncreaser(midiArray):
	midi = []
	for i in range(len(midiArray)):
		if(midiArray[i] == 0):
			midi.append(0)
		else:
			midi.append(midiArray[i] + 2)
	return midi

def main():
	musicFinal = numpy.zeros(0)
	musicFinal2 = numpy.zeros(0)
	adsr = make_adsr(duration)
	katyperry = numpy.zeros(0)
	scaleMidi = scaleMajorMaker()
	
	for perry in scaleMidi:
		temp = createMusic(perry, duration)
		katyperry = numpy.concatenate((katyperry, adsr * temp))
	katyperry = numpy.array(katyperry)
	(32767 * katyperry).astype(numpy.int16)
	wavwrite("music_output/fireworks.wav", samp_rate, katyperry)

	# for note in midis:
	# 	temp = createMusic(note, duration)
	# 	musicFinal = numpy.concatenate((musicFinal, temp))
	# 	musicFinal2 = numpy.concatenate((musicFinal2, adsr * temp))
	# musicFinal = numpy.array(musicFinal)
	# musicFinal2 = numpy.array(musicFinal2)

	# (32767 * musicFinal).astype(numpy.int16)
	# (32767 * musicFinal2).astype(numpy.int16)


	# make_spectrogram(musicFinal, "spectrogram-notes.png")
	# make_spectrogram(musicFinal2, "spectrogram-notes-adsr.png")
	
	# wavwrite("music_output/notes.wav", samp_rate, musicFinal)
	# wavwrite("music_output/notes-adsr.wav", samp_rate, musicFinal2)
	

def calcFundamentalFreq(m):
	if (m == 0):
		return 0.0;
	else:
		return 440 * pow(2, (m - 69) / 12.0)

def calcSineWave(freq, phase, amplitude, duration):
	sineArray = numpy.arange(int(duration * samp_rate))
	return amplitude * numpy.sin(sineArray * (freq / samp_rate) * 2 * numpy.pi + phase)

def db_spectrum(time_domain_data, window):
	fft = scipy.fftpack.fft(window * time_domain_data)
	fft = fft[: len(fft) / 2 + 1]
	magfft = abs(fft) / (numpy.sum(window) / 2.0)
	epsilon = pow(10, -10)
	db = 20 * numpy.log10(magfft + epsilon)
	return db

def createMusic(midiNote, duration):
	freq = calcFundamentalFreq(midiNote)
	wave = numpy.zeros(int(duration * samp_rate))
	for i in range(1, 5):
		wave = wave + calcSineWave(freq * i, 0.0, 0.25, duration);
	return wave

def make_adsr(duration):
	timeToAttack = duration * samp_rate * 0.1
	attackValues = numpy.linspace(0, 1, timeToAttack)
	# attackValues = numpy.delete(attackValues, timeToAttack)
	timeForDecay = duration * samp_rate * 0.15
	decayValues = numpy.linspace(1, 0.5, timeForDecay)
	# decayValues = numpy.delete(decayValues, timeForDecay)
	timeForSustain = duration * samp_rate * 0.3
	sustainValues = numpy.empty(timeForSustain)
	sustainValues.fill(0.5)
	timeForRelease = duration * samp_rate * 0.45
	releaseValues = numpy.linspace(0.5, 0, timeForRelease)
	return numpy.concatenate((attackValues, decayValues, sustainValues, releaseValues))

def make_spectrogram(data, fileName):
	numBuffers = int(len(data) / 256 - 1)
	buffers = numpy.zeros((numBuffers, 257))

	for i in range(numBuffers):
		start = i * 256
		end = i * 256 + 512
		sliced = data[start:end]
		buffers[i] = db_spectrum(sliced, numpy.blackman(512))

	buffers = buffers.transpose()
	plot.figure(figsize = (18, 8))
	plot.title("MIDI Spectrogram")
	plot.xlabel("Time(hops)")
	plot.ylabel("Frequency Bin")
	plot.imshow(buffers, origin = "lower")
	plot.savefig(fileName)

	return

main()