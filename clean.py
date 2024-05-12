import soundfile
import librosa
import noisereduce
import numpy as np
import lib

DROP_AUDIO_THRESHOLD = -10
DATA_SUM_THRESHOLD  = -3
NOISE_REDUCETION_ROUND = 4

def reduce_noise(fileName):
    for i in range (0, NOISE_REDUCETION_ROUND):
        data, sampleRate = soundfile.read(fileName)
        denoisedData = noisereduce.reduce_noise(y=data, sr=sampleRate)
        soundfile.write(file=fileName, data=denoisedData, samplerate=sampleRate)

def noiseDetector(fileName):
    data, _ = librosa.load(fileName)
    sum = 0
    for d in data:
        sum += d
    return sum

def audio_cleaner(dataFileNames):
    eliminatedFiles = []
    for fileName in dataFileNames:
        dataSum = noiseDetector(lib.DIR + fileName)
        if dataSum < DROP_AUDIO_THRESHOLD:
            eliminatedFiles.append(fileName)

        elif dataSum < DATA_SUM_THRESHOLD:
            reduce_noise(lib.DIR + fileName)

    return eliminatedFiles

def eliminateFiles(eliminationList, dataFileNames):
    for el in eliminationList:
        dataFileNames.remove(el)

    return dataFileNames
