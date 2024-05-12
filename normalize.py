from pydub import AudioSegment
import lib

VOLUME_THRESHOLD = -20

def upgrade_volume(fileName, currentVolume):
    audio = AudioSegment.from_file(fileName)
    modifiedAudio = audio + (VOLUME_THRESHOLD - currentVolume)
    modifiedAudio.export(fileName, format="wav")

def get_volume(fileName):
    audio = AudioSegment.from_file(fileName)
    volume = audio.dBFS
    return volume

def flatten_volumes(dataFileNames):
    for fileName in dataFileNames:
        fileName = lib.DIR + fileName
        volume = get_volume(fileName)
        if volume < VOLUME_THRESHOLD:
            upgrade_volume(fileName, volume)
