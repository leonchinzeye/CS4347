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
resultRMS = []
resultPAR = []
resultZCR = []
resultMAD = []


# main driver function
def main():
    readFile(GROUND_TRUTH_DATA)
    performWAVCalculations()
    graphDriver()
    writeToFile()


"""
This function reads in the necessary wav files and obtains the sample data. Upon obtaining the sample
data, the data will then be passed to the appropriate functions to perform calculations to derive the
required answers
"""
def performWAVCalculations():
    for i in range(0, len(wavNameList)):
        rate, data = wavfile.read("music_speech/" + wavNameList[i])
        data = data / DIVIDE_NUMBER

        resultRMS.append(calculateRMS(data))
        resultPAR.append(calculatePAR(data, i))
        resultZCR.append(calculateZCR(data))
        resultMAD.append(calculateMAD(data))


"""
Function to calculate RMS given a numpy array containing the sample data
"""
def calculateRMS(data):
    data = numpy.power(data, 2)
    total = numpy.sum(data)
    total = total / len(data)
    total = numpy.power(total, 0.5)
    return total


"""
Function to calculate PAR given a numpy array containing the sample data
"""
def calculatePAR(data, dataIndex):
    data = numpy.absolute(data)
    biggestValue = numpy.max(data)
    answer = biggestValue / resultRMS[dataIndex]

    return answer


"""
Function to calculate ZCR given a numpy array containing the sample data
"""
def calculateZCR(data):
    signOfNumbers = numpy.sign(data)
    diffArray = numpy.abs(numpy.diff(signOfNumbers))
    answer = len(numpy.where(diffArray == 2)[0])
    answer = answer / (len(data) - 1.0)

    return answer


"""
Function to calculate MAD given a numpy array containing the sample data
"""
def calculateMAD(data):
    median = numpy.median(data)
    data = data - median
    data = numpy.abs(data)
    median = numpy.median(data)

    return median


def draw_Graph(features_music, features_speech, fileName, xindex, yindex, xlabel, ylabel):
    pylab.figure(figsize=(18, 8))
    pylab.plot(features_music[:, xindex], features_music[:, yindex], "go", label="music")
    pylab.plot(features_speech[:, xindex], features_speech[:, yindex], "ro", label="speech")
    pylab.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)
    pylab.title(xlabel + " against " + ylabel)
    pylab.savefig(fileName)


def graphDriver():
    counter = 0
    features_music = numpy.zeros((len(wavNameList) / 2, 4))
    features_speech = numpy.zeros((len(wavNameList) / 2, 4))

    for i in range(len(wavNameList)):
        features = [resultRMS[i], resultPAR[i], resultZCR[i], resultMAD[i]]

        if labelList[i] == "music":
            features_music[i] = features
        else:
            features_speech[counter] = features
            counter = counter + 1

    draw_Graph(features_music, features_speech, "ZCR_vs_PAR.png", 2, 1, "ZCR", "PAR")
    draw_Graph(features_music, features_speech, "MAD_vs_RMS.png", 3, 0, "MAD", "RMS")

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

    newFile.write("@RELATION music_speech\n@ATTRIBUTE RMS NUMERIC\n@ATTRIBUTE PAR NUMERIC\n@ATTRIBUTE ZCR NUMERIC\n@ATTRIBUTE MAD NUMERIC\n@ATTRIBUTE class {music,speech}\n")
    newFile.write("\n")
    newFile.write("@DATA\n")

    for i in range(0, len(wavNameList)):
        newFile.write(str.format(DECIMAL_PLACES % resultRMS[i]) + "," + str.format(DECIMAL_PLACES % resultPAR[i]) + "," + str.format(DECIMAL_PLACES % resultZCR[i]) + "," + str.format(DECIMAL_PLACES % resultMAD[i]) + "," + labelList[i] + "\n")

    newFile.close()

main()