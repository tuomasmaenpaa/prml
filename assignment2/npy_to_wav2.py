import numpy as np
import scipy.io.wavfile
import tensorflow as tf

PATH_DS1 = "/home/maenpaat/prml/assignment2/ff1010bird/wav/"
PATH_DS2 = "/home/maenpaat/prml/assignment2/warblrb10k_public/wav/"
PATH_DS3 = "/home/maenpaat/prml/assignment2/audio/"

CSV_DS1 = "/home/maenpaat/prml/assignment2/ff1010bird/ff1010bird_metadata_2018.csv"
CSV_DS2 = "/home/maenpaat/prml/assignment2/warblrb10k_public/warblrb10k_public_metadata_2018.csv"

for i in range(4512):
    print(i, "/4512", sep="")
    audio = np.load(PATH_DS3 + str(i) + ".npy")
    audio = (audio * (2**15 / np.max(np.abs(audio))) - 2**14).astype(np.int16)
    samplerate = 48000
    scipy.io.wavfile.write(PATH_DS3 + str(i) + ".wav", samplerate, audio)
