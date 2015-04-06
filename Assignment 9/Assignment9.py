import numpy as np
import math
import pylab as py
import scipy.fftpack
from scipy.io import wavfile

audio_file = "clear_d1.wav"
divide_number = 32768.0
window_size = 128
fs = 22050.0

def main():
	analyse_audio()

def reconstruct_audio(freq_amp, numBuffers):
	sine_wav = np.zeros(0)

	for i in range(numBuffers):
		wave = calcSineWave(freq_amp[i, 0], freq_amp[i, 1])
		sine_wav = np.concatenate((sine_wav, wave))

	re_wav_norm = sine_wav / sine_wav.max() * 32767
	re_wav_norm = re_wav_norm.astype(np.int16)
	wavfile.write("reconstructed.wav", fs, re_wav_norm)

def calcSineWave(freq, amplitude):
	sine_wav = amplitude * np.sin(2 * np.pi * freq / fs * np.arange(window_size))
	return sine_wav

def analyse_audio():
	rate, data = wavfile.read(audio_file)

	numBuffers = int(len(data) / window_size)
	bufferArr = np.zeros((numBuffers, window_size))
	data = data / divide_number

	for i in range(numBuffers):
		start = i * window_size
		end = start + window_size
		buffer_data = data[start:end]
		bufferArr[i, :] = buffer_data

	mag_spec = db_spectrum(bufferArr, np.hamming(window_size))
	freq_amp = getFreqAmp(mag_spec, numBuffers)

	np.savetxt("freq_amp.csv", freq_amp, fmt = "%.6g", delimiter = ",")

	reconstruct_audio(freq_amp, numBuffers)

	rate, data = wavfile.read("reconstructed.wav")
	data = data / divide_number
	bufferArr2 = np.zeros((numBuffers, window_size))

	for i in range(numBuffers):
		start = i * window_size
		end = start + window_size
		buffer_data = data[start:end]
		bufferArr2[i, :] = buffer_data

	mag_spec2 = db_spectrum(bufferArr2, np.hamming(window_size))

	plot_spectrogram(mag_spec, mag_spec2)

def getFreqAmp(bufferArr, numBuffers):
	maxVal = np.amax(bufferArr, axis = 1)
	maxIndex = np.argmax(bufferArr, axis = 1)
	correspondingFreq = maxIndex / float(window_size) * fs

	freq_amp = np.zeros((numBuffers, 2))
	freq_amp[:,0] = correspondingFreq
	freq_amp[:,1] = maxVal

	return freq_amp

def db_spectrum(data, window):
	fft = scipy.fftpack.fft(window * data)
	fft = fft[:,:len(fft[0]) / 2 + 1]
	magfft = abs(fft) 
	return magfft

def plot_spectrogram(orig, recon):
	py.subplot(2, 1, 1)
	py.title("Spectrogram")
	py.imshow(orig.T / orig.max(), origin = "lower", aspect = "auto")
	py.xlabel("frames (original wav)")
	py.ylabel("freq bin")

	py.subplot(2, 1, 2)
	py.imshow(recon.T / recon.max(), origin = "lower", aspect = "auto")
	py.xlabel("frames (reconstructed wave)")
	py.ylabel("freq bin")
	py.savefig("spectrogram.png")

main()