__author__ = 'leon'

import pylab
import numpy
from scipy.io import wavfile


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

    for i in range(0, len(wavNameList)):
        rate, data = wavfile.read("music_speech/" + wavNameList[i])
        data = data / DIVIDE_NUMBER

        for j in range(numBuffers):
            start = j * (bufferLength / 2)
            end = start + bufferLength
            buffer_data = data[start:end]
            bufferMatrix[j, :] = buffer_data

        featuresMatrix = numpy.zeros((numBuffers, 5))
        featuresMatrix[:,0] = calculateRMS(bufferMatrix)
        featuresMatrix[:,1] = calculatePAR(bufferMatrix, featuresMatrix[:,0])
        featuresMatrix[:,2] = calculateZCR(bufferMatrix)
        featuresMatrix[:,3] = calculateMAD(bufferMatrix)
        featuresMatrix[:,4] = calculateMEANAD(bufferMatrix)

        # Creating a (1, 10) array here which will have the mean and std of the features
        resultMatrix = numpy.zeros(10)
        resultMatrix[0:5] = numpy.mean(featuresMatrix, axis = 0)
        resultMatrix[5:10] = numpy.std(featuresMatrix, axis = 0)

        # Storing the (1, 10) array into the (128, 10) array
        resultList[i, :] = resultMatrix


"""
Function to calculate RMS given a numpy array containing the sample data
"""
def calculateRMS(data):

    answer = numpy.power(data, 2)
    answer = numpy.sum(answer, axis = 1)
    answer = answer / bufferLength
    answer = numpy.power(answer, 0.5)

    return answer


"""
Function to calculate PAR given a numpy array containing the sample data
"""
def calculatePAR(data, rmsArray):

    answer = numpy.absolute(data)
    answer = numpy.amax(answer, axis = 1)
    rmsArray = numpy.power(rmsArray, -1)
    answer = answer * rmsArray

    return answer


"""
Function to calculate ZCR given a numpy array containing the sample data
"""
def calculateZCR(data):

    x1array = data[:numBuffers, :bufferLength - 1]
    x2array = data[:numBuffers, 1:bufferLength]
    answer = x1array * x2array
    answer = numpy.where(answer < 0, 1, 0)
    answer = numpy.sum(answer, axis = 1)
    answer = answer / (bufferLength - 1)

    return answer


"""
Function to calculate MAD given a numpy array containing the sample data
"""
def calculateMAD(data):
    medianMatrix = numpy.median(data, axis = 1)
    medianMatrix = numpy.reshape(medianMatrix, (len(medianMatrix), 1))
    answer = data - medianMatrix
    answer = numpy.absolute(answer)
    answer = numpy.median(answer, axis = 1)

    return answer


def calculateMEANAD(data):
    meanMatrix = numpy.mean(data, axis = 1)
    meanMatrix = numpy.reshape(meanMatrix, (len(meanMatrix), 1))
    answer = data - meanMatrix
    answer = numpy.absolute(answer)
    answer = numpy.mean(answer, axis = 1)

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
    newFile = open("answers.arff", "w")

    newFile.write("@RELATION music_speech\n")
    newFile.write("@ATTRIBUTE RMS_MEAN NUMERIC\n@ATTRIBUTE PAR_MEAN NUMERIC\n@ATTRIBUTE ZCR_MEAN NUMERIC\n@ATTRIBUTE MAD_MEAN NUMERIC\n@ATTRIBUTE MEAN_AD_MEAN NUMERIC\n")
    newFile.write("@ATTRIBUTE RMS_STD NUMERIC\n@ATTRIBUTE PAR_STD NUMERIC\n@ATTRIBUTE ZCR_STD NUMERIC\n@ATTRIBUTE MAD_STD NUMERIC\n@ATTRIBUTE MEAN_AD_STD NUMERIC\n")
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