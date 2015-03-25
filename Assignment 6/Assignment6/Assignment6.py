__author__ = 'leon'

import csv
import math
import numpy
import pylab as plt
import scipy.fftpack
from scipy.io import wavfile
from scipy.signal import hamming


# declaration of variables that will store the names of the files as well as the labels
WAV_FILE_NAME = "Moskau.wav"
DIVIDE_NUMBER = 32768.0
DECIMAL_PLACES = "%.6f"
bufferLength = 1024

# main driver function
def main():
    result = performWAVCalculations()
    writeToFile(result)

def performWAVCalculations():
    rate, data = wavfile.read(WAV_FILE_NAME)
    data = data / DIVIDE_NUMBER 
    numBuffers =  len(data) / (bufferLength / 2) - 1

    bufferMatrix = numpy.zeros((numBuffers, bufferLength))
    hammingMatrix = numpy.zeros((numBuffers, bufferLength))

    for i in range(numBuffers):
        start = i * (bufferLength / 2)
        end = start + bufferLength
        buffer_data = data[start:end]
        bufferMatrix[i, :] = buffer_data

    hammingMatrix = applyHamming(bufferMatrix, numBuffers)
    fftMatrix = applyMag(hammingMatrix)
    accentResult = calcAccentSignal(fftMatrix, numBuffers)
    autoCorrelation = calcAutoCorr(accentResult)

    lowerBPMIndex = int(math.floor(findIndex(180)))
    upperBPMIndex = int(math.ceil(findIndex(60)))

    tempoIndex = findTempoIndex(autoCorrelation, lowerBPMIndex, upperBPMIndex)
    beatLocations = beatAnalysis(accentResult, tempoIndex)


    result = numpy.array(beatLocations) * 0.0116

    plotGraph(accentResult * 0.0116, result) 

    return result

def calcAccentSignal(data, numBuffers):
    sN = numpy.abs(data[1:numBuffers])
    sN_1 = numpy.abs(data[0:numBuffers - 1])
    resultingMatrix = sN - sN_1
    resultingMatrix = numpy.where(resultingMatrix < 0, 0, resultingMatrix)
    result = numpy.sum(resultingMatrix, axis = 1)
    return result

def calcAutoCorr(accentSignal):
    result = numpy.correlate(accentSignal, accentSignal, mode = 'full')
    return result[result.size / 2:]

def findIndex(bpm):
    return 60 / (bpm * 0.0116)

def findTempoIndex(data, lower, upper):
    temp = data[lower:upper + 1]
    index_max = temp.argmax()
    return index_max + lower

def beatAnalysis(origAccent, tempoIndex):
    firstBeatIndex = findFirstBeatIndex(origAccent, tempoIndex)

    arrayOfBeatLocations = []
    arrayOfBeatLocations.append(firstBeatIndex)

    beatIndexCounter = firstBeatIndex + tempoIndex
    while (beatIndexCounter < len(origAccent)):
        lower = beatIndexCounter - 10
        upper = beatIndexCounter + 10
        beatIndex = findTempoIndex(origAccent, lower, upper)
        beatIndexCounter = beatIndex
        arrayOfBeatLocations.append(beatIndex)
        beatIndexCounter += tempoIndex

    return arrayOfBeatLocations

def findFirstBeatIndex(origAccent, tempoIndex):
    temp = origAccent[0:tempoIndex]
    return temp.argmax()

def applyMag(data):
    # applies the mag spectrum calculation on the buffers
    result = scipy.fftpack.fft(data, axis = 1)
    result = numpy.abs(result[:, 0:bufferLength / 2 + 1])
    return result

def applyHamming(data, numBuffers):
    hammMult = []
    for i in range(numBuffers):
        hammMult.append(hamming(bufferLength))
    hammMult = numpy.array(hammMult)

    return data * hammMult

def writeToFile(results):
    results.tofile("beat_time.csv", sep = ",")

main()