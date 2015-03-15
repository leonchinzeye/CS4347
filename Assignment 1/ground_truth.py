__author__ = 'leon'

import csv
import numpy
from scipy.io import wavfile

# declaration of variables that will store the names of the files as well as the labels
GROUND_TRUTH_DATA = "music_speech.mf.txt"
DIVIDE_NUMBER = 32768.0
DECIMAL_PLACES = "%.6f"
wavNameList = []
resultRMS = []
resultPAR = []
resultZCR = []
resultMAD = []


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

    # for i in range(0, len(signOfNumbers) - 1):
    #     if (signOfNumbers[i] == 1 and signOfNumbers[i + 1] == -1) or (signOfNumbers[i] == -1 and signOfNumbers[i + 1] == 1):
    #         answer += 1
    #
    # answer = answer / (len(data) - 1.0)

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

"""
Function to read file
"""
def readFile(fileName):
    file = open(fileName, "r")

    for line in file:
        readLine = line.split("\t")
        wavNameList.append(readLine[0])

    file.close()


"""
Function to write to a file of CSV type
"""
def writeToFile():
    file = open("answers.csv", "wb")
    csvWriter = csv.writer(file)

    for i in range(0, len(wavNameList)):
        csvWriter.writerow([wavNameList[i], DECIMAL_PLACES % resultRMS[i], DECIMAL_PLACES % resultPAR[i], DECIMAL_PLACES % resultZCR[i], DECIMAL_PLACES % resultMAD[i]])

    file.close()

main()