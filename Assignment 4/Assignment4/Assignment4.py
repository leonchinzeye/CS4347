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
resultList = numpy.zeros((128, 10))

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
            hammingMatrix[j, :] = buffer_data * hamming(bufferLength)

        dftMatrix = fftpack.fft(hammingMatrix, axis = 1)
        dftMatrix = numpy.abs(dftMatrix[:, 0:bufferLength / 2 + 1])

        featuresMatrix = numpy.zeros((numBuffers, 5))
        featuresMatrix[:,0] = calculateSC(dftMatrix)
        featuresMatrix[:,1] = numpy.apply_along_axis(calculateSRO, 1, dftMatrix)
        featuresMatrix[:,2] = calculateSFM(dftMatrix)
        featuresMatrix[:,3] = calculatePARFFT(dftMatrix)
        featuresMatrix[:,4] = calculateSF(dftMatrix)

        resultMatrix = numpy.zeros(10)
        resultMatrix[0:5] = numpy.mean(featuresMatrix, axis = 0)
        resultMatrix[5:10] = numpy.std(featuresMatrix, axis = 0)

        resultList[i, :] = resultMatrix


"""
Function to calculate spectral centroid
"""
def calculateSC(data):
    arrayK = numpy.indices((numBuffers, bufferLength / 2 + 1))
    answer = numpy.sum(data * arrayK[1], axis = 1) / numpy.sum(data, axis = 1)

    return answer


"""
Function to calculate the spectral roll-off
"""
def calculateSRO(data):
    comparedNumber = numpy.sum(data) * 0.85
    calcSum = 0

    for i in range(0, len(data)):
        calcSum += data[i]

        if calcSum >= comparedNumber:
            return i


"""
Function to calculate the spectral flatness measure
"""
def calculateSFM(data):
    answer = numpy.exp(numpy.mean(numpy.log(data), axis = 1)) / numpy.mean(data, axis = 1)

    return answer


"""
Function to calculate the PARFFT
"""
def calculatePARFFT(data):
    answer = numpy.amax(data, axis = 1) / numpy.sqrt(numpy.mean(numpy.square(data), axis = 1))

    return answer


"""
Function to calculate the spectral flux
"""
def calculateSF(data):
    arrayN_1 = numpy.vstack([numpy.zeros(data.shape[1]), data[:-1]])
    answer = numpy.sum((data - arrayN_1).clip(0), axis = 1)

    return answer

    
"""
Function to read file
"""
def readFile(fileName):
    file = open(fileName, "r")

    for line in file:
        readLine = line.split("\t")
        wavNameList.append(readLine[0])
        labelList.append(readLine[1].strip())

    file.close()


"""
Function to write to a file of CSV type
"""
def writeToFile():
    newFile = open("results.arff", "w")

    # newFile.write("@RELATION music_speech\n")
    # newFile.write("@ATTRIBUTE SC_MEAN\n@ATTRIBUTE SRO_MEAN\n@ATTRIBUTE SFM_MEAN\n@ATTRIBUTE PARFFT_MEAN\n@ATTRIBUTE FLUX_MEAN\n")
    # newFile.write("@ATTRIBUTE SC_STD\n@ATTRIBUTE SRO_STD\n@ATTRIBUTE SFM_STD\n@ATTRIBUTE PARFFT_STD\n@ATTRIBUTE FLUX_STD\n")
    # newFile.write("@ATTRIBUTE class {music,speech}\n")
    # newFile.write("\n")
    # newFile.write("@DATA\n")

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