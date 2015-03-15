__author__ = 'leon'

import pylab as plt
import math
import numpy
import scipy.fftpack
from scipy.io import wavfile
from scipy.signal import hamming


# declaration of variables that will store the names of the files as well as the labels
GROUND_TRUTH_DATA = "music_speech.mf.txt"
DIVIDE_NUMBER = 32768.0
DECIMAL_PLACES = "%.6f"
wavNameList = []
labelList = []
bufferLength = 1024
numBuffers = 661500 / bufferLength * 2
numWindows = 26
resultList = numpy.zeros((128, 52))

# main driver function
def main():
    readFile(GROUND_TRUTH_DATA)
    performWAVCalculations()
    writeToFile()

"""
This function reads in the necessary wav files and obtains the sample data. Upon obtaining the sample
data, the data will then be passed to the appropriate functions to perform calculations to derive the
required answers
"""
def performWAVCalculations():
    bufferMatrix = numpy.zeros((numBuffers, bufferLength))
    hammingMatrix = numpy.zeros((numBuffers, bufferLength))

    for i in range(0, len(wavNameList)):
        rate, data = wavfile.read("music_speech/" + wavNameList[i])
        data = data / DIVIDE_NUMBER

        for j in range(numBuffers):
            start = j * (bufferLength / 2)
            end = start + bufferLength
            buffer_data = data[start:end]
            bufferMatrix[j, :] = buffer_data
        
        bufferMatrix = addPreEmphasis(bufferMatrix)
        hammingMatrix = applyHamming(bufferMatrix)
        fftMatrix = applyMag(hammingMatrix)

        mel_interval = calcMelInterval(rate)
        normalisationVal = rate / 1024.0
        melData = numpy.zeros((numWindows, 513))

        for k in range(numWindows):
            left = melInverse(mel_interval * k) / normalisationVal
            top = melInverse(mel_interval * (k + 1)) / normalisationVal
            right = melInverse(mel_interval * (k + 2)) / normalisationVal
            melData[k] = createMelPoints(left, top, right)

        mfccResultMatrix = performMfcc(fftMatrix, melData)
        performFinalCalc(mfccResultMatrix, i)

    plotGraph(melData, rate / 2.0)

def performFinalCalc(result, val):
    temp = numpy.zeros(52)
    temp[0:26] = numpy.mean(result, axis = 0)
    temp[26:52] = numpy.std(result, axis = 0)
    resultList[val, :] = temp

def performMfcc(fft, melData):
    # applies the MFCC on to the buffers
    return scipy.fftpack.dct(numpy.log10(numpy.dot(fft, melData.T)))

def applyMag(data):
    # applies the mag spectrum calculation on the buffers
    result = scipy.fftpack.fft(data, axis = 1)
    result = numpy.abs(result[:, 0:bufferLength / 2 + 1])
    return result

def applyHamming(data):
    hammMult = []
    for i in range(numBuffers):
        hammMult.append(hamming(bufferLength))
    hammMult = numpy.array(hammMult)

    return data * hammMult

def createMelPoints(left, top, right):
    left = math.floor(left)
    top = round(top)
    right = math.ceil(right)
    leftPoints = numpy.linspace(0, 1, num = (top - left + 1))
    leftPoints = numpy.delete(leftPoints, top - left)
    rightPoints = numpy.linspace(1, 0, num = (right - top + 1))
    leftZeroes = numpy.zeros(left)
    melPoints = numpy.concatenate((leftZeroes, leftPoints, rightPoints))

    rightZeroes = numpy.zeros(512 - right)
    return numpy.concatenate((melPoints, rightZeroes))

def addPreEmphasis(data):
    subMatrix = numpy.zeros((1290, 1024))
    temp = data[:, :-1]
    subMatrix[:, 1:1024] = temp
    emphasisMatrix = data - (0.95 * subMatrix)

    return emphasisMatrix

def calcMelInterval(rate):
    melInterval = 1127 * math.log(1 + (rate / 2.0) / 700) / (numWindows + 1)    # 26 windows, need to consider the last point 
    return melInterval                                                                      

def melInverse(mel):
    return 700 * (math.exp(mel / 1127) - 1)    

def plotGraph(melWindow, rate):
    freq = numpy.linspace(0, rate, num = 513)
    
    plt.figure()
    for i in range(numWindows):
        plt.plot(freq, melWindow[i]) 
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("26 Triangular MFCC filters, 22050 Hz signal, window size 1024")
    plt.savefig("Figure 1")

    plt.figure()
    for i in range(numWindows):
        plt.plot(freq, melWindow[i], '.-')    
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("26 Triangular MFCC filters, 22050 Hz signal, window size 1024")
    plt.xlim(0, 300)
    plt.savefig("Figure 2")

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
    
    for i in range(0, 52):
        sentence = "@ATTRIBUTE MFCC_" + str(i) + " NUMERIC\n"
        newFile.write(sentence)
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