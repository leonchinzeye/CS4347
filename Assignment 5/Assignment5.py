__author__ = 'leon'

import pylab
import numpy
from scipy.io import wavfile
from scipy.signal import hamming
from scipy import fftpack


# declaration of variables that will store the names of the files as well as the labels
GROUND_TRUTH_DATA = "music_speech.mf.txt"
DIVIDE_NUMBER = 32768.0
DECIMAL_PLACES = "%.6f"
wavNameList = []
labelList = []
bufferLength = 1024
numBuffers = 661500 / bufferLength * 2
numWindows = 26
resultList = numpy.zeros((128, 10))

# main driver function
def main():
    readFile(GROUND_TRUTH_DATA)
    performWAVCalculations()
    # writeToFile()


"""
This function reads in the necessary wav files and obtains the sample data. Upon obtaining the sample
data, the data will then be passed to the appropriate functions to perform calculations to derive the
required answers
"""
def performWAVCalculations():
    bufferMatrix = numpy.zeros((numBuffers, bufferLength))
    hammingMatrix = numpy.zeros((numBuffers, bufferLength))

    for i in range(0, 1):
        rate, data = wavfile.read("music_speech/" + wavNameList[i])
        data = data / DIVIDE_NUMBER
        data = addPreEmphasis(data)

        mel_interval = melFunction(rate)


        for j in range(numBuffers):
            start = j * (bufferLength / 2)
            end = start + bufferLength
            buffer_data = data[start:end]
            bufferMatrix[j, :] = buffer_data
            hammingMatrix[j, :] = buffer_data * hamming(bufferLength)

        dftMatrix = fftpack.fft(hammingMatrix, axis = 1)
        dftMatrix = numpy.abs(dftMatrix[:, 0:bufferLength / 2 + 1])

def addPreEmphasis(data):
    temp = numpy.delete(data, len(data) - 1)
    temp = numpy.insert(temp, [0.0], 0)
    return data - 0.95 * temp

def calcMelInterval(rate):
    nyquistLimit = rate / 2.0
    return melFunction(nyquistLimit) / (numWindows + 1)         # 26 windows, need to consider the last point             

def melFunction(frequency):
    return 1127 * math.log(1 + frequency / 700)

def melFunctionInverse(mel):
    return 700 * (math.exp(mel / 1127) - 1)    

def calcMagSpec(data, hammingWindow):
    fft = numpy.fft.fft(data)
    fft = fft[:len(fft) / 2 + 1]
    magfft = abs(fft) / (numpy.sum(hammingWindow) / 2.0)
    return magfft

def readFile(fileName):
    file = open(fileName, "r")

    for line in file:
        readLine = line.split("\t")
        wavNameList.append(readLine[0])
        labelList.append(readLine[1].strip())

    file.close()

def writeToFile():
    newFile = open("results.arff", "w")

    newFile.write("@RELATION music_speech\n")
    newFile.write("@ATTRIBUTE SC_MEAN NUMERIC\n@ATTRIBUTE SRO_MEAN NUMERIC\n@ATTRIBUTE SFM_MEAN NUMERIC\n@ATTRIBUTE PARFFT_MEAN NUMERIC\n@ATTRIBUTE FLUX_MEAN NUMERIC\n")
    newFile.write("@ATTRIBUTE SC_STD NUMERIC\n@ATTRIBUTE SRO_STD NUMERIC\n@ATTRIBUTE SFM_STD NUMERIC\n@ATTRIBUTE PARFFT_STD NUMERIC\n@ATTRIBUTE FLUX_STD NUMERIC\n")
    newFile.write("@ATTRIBUTE class {music,speech}\n")
    newFile.write("\n")
    newFile.write("@DATA\n")

    for i in range(0, len(wavNameList)):
        arrayToBeWritten = resultList[i,:]

        for j in range(len(arrayToBeWritten)):
            newFile.write(str.format(DECIMAL_PLACES % arrayToBeWritten[j]) + ",")
        newFile.write(labelList[i] + "\n")

    newFile.close()

main()