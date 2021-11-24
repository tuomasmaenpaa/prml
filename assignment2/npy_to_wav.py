import numpy as np
import scipy.io.wavfile
import tensorflow as tf

PATH_DS1 = "/Users/maenpaat/Documents/pr_n_ml/assignment2/ff1010bird/wav/"
PATH_DS2 = "/Users/maenpaat/Documents/pr_n_ml/assignment2/warblrb10k_public/wav/"
PATH_DS3 = "/Users/maenpaat/Documents/pr_n_ml/assignment2/audio/"

CSV_DS1 = "C:/PRML/ff1010bird_metadata_2018.csv"
CSV_DS2 = "C:/PRML/warblrb10k_public_metadata_2018.csv"

for i in range(4513):
    print(i, "/4513", sep="")
    audio = np.load(PATH_DS3 + str(i) + ".npy")
    audio = (audio * (2**16 / np.max(audio))).astype(np.int16)
    samplerate = 48000
    scipy.io.wavfile.write(PATH_DS3 + str(i) + ".wav", samplerate, audio)
