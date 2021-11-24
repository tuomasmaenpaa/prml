#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:34:30 2020

@author: maenpaat
"""

import numpy as np
import csv
import random
from scipy.io import wavfile
import scipy.signal 
from scipy.signal import butter
from scipy.signal import sosfilt
import time
from librosa.feature import melspectrogram

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.model_selection import  cross_val_score



PATH_DS1 = "/home/maenpaat/prml/assignment2/ff1010bird/wav/"
PATH_DS2 = "/home/maenpaat/prml/assignment2/warblrb10k_public/wav/"
PATH_DS3 = "/home/maenpaat/prml/assignment2/audio/"

CSV_DS1 = "/home/maenpaat/prml/assignment2/ff1010bird/ff1010bird_metadata_2018.csv"
CSV_DS2 = "/home/maenpaat/prml/assignment2/warblrb10k_public/warblrb10k_public_metadata_2018.csv"

AUDIO_SIZE = 441000
SG_WIDTH = 513
SG_HEIGHT = 860

def read_csv(csv_file, path):
    data = []
    reader = csv.reader(open(csv_file, "r"))
    next(reader)
    for row in reader:
        data.append((path + row[0] + ".wav", int(row[2])))    
    return data


def load_wav(path, label):
    
    fs, audio = wavfile.read(path)
    #audio, fs = tf.audio.decode_wav(tf.io.read_file(path))
    
    # Scale to [-1, 1]
    audio = audio / np.max(audio)
    #audio = audio / tf.math.abs(tf.reduce_max(audio))
    
    # Add padding if there are not enough samples and remove samples that exceed AUDIO_SIZE
    #audio = tf.pad(audio, [[0, tf.math.abs(AUDIO_SIZE - tf.shape(audio)[0])], [0, 0]])[:AUDIO_SIZE]
    
    audio = np.pad(audio, AUDIO_SIZE, mode='constant', constant_values=(0,0))
    sos = butter(4,10, btype='highpass', fs=fs, output='sos')

    # Highpass filter audio
    audio = sosfilt(sos, audio)
    
    spectrogram = melspectrogram(audio, fs, n_fft=1024, hop_length=512)
    spectrogram = np.log(spectrogram + np.ones(spectrogram.shape))
    #label = tf.keras.utils.to_categorical(label)
    
    """
    spectrogram = tf.raw_ops.AudioSpectrogram(input=audio, window_size=1024, stride=512)
    spectrogram = tf.math.log(tf.math.exp(tf.constant(1.0)) + spectrogram)
    spectrogram = spectrogram / tf.reduce_max(spectrogram)
   #spectrogram = tf.transpose(spectrogram)
    label = tf.one_hot(label, 2)
    spectrogram = tf.reshape(spectrogram,[-1])
    #tf.expand_dims(spectrogram[0], -1)
    """
    return spectrogram, label





# Get full paths and labels
data_ds1 = read_csv(CSV_DS1, PATH_DS1)
data_ds2 = read_csv(CSV_DS2, PATH_DS2)

# Shuffle both training sets
random.shuffle(data_ds1)
random.shuffle(data_ds2)
"""
paths_ds1  = tf.constant(np.asarray([i[0] for i in data_ds1]))
paths_ds2  = tf.constant(np.asarray([i[0] for i in data_ds2]))
labels_ds1 = tf.constant(np.asarray([i[1] for i in data_ds1]))
labels_ds2 = tf.constant(np.asarray([i[1] for i in data_ds2]))
"""
# Create training and validation sets
#tr_dataset = tf.data.Dataset.from_tensor_slices((tf.concat([paths_ds1[:6150], paths_ds2[:6400]], 0), tf.concat([labels_ds1[:6150], labels_ds2[:6400]], 0))).shuffle(12550, reshuffle_each_iteration=True).map(load_wav).batch(5)
#va_dataset = tf.data.Dataset.from_tensor_slices((tf.concat([paths_ds1[6150:], paths_ds2[6400:]], 0), tf.concat([labels_ds1[6150:], labels_ds2[6400:]], 0))).map(load_wav).batch(5)

#tr_dataset = tf.data.Dataset.from_tensor_slices((tf.concat([paths_ds1[:500], paths_ds2[:500]], 0), tf.concat([labels_ds1[:500], labels_ds2[:500]], 0))).shuffle(1000, reshuffle_each_iteration=True).map(load_wav).batch(5)
#va_dataset = tf.data.Dataset.from_tensor_slices((tf.concat([paths_ds1[500:550], paths_ds2[500:550]], 0), tf.concat([labels_ds1[500:550], labels_ds2[500:550]], 0))).map(load_wav).batch(5)

# Create test set
paths_ds3 = []
for i in range(4512):
    paths_ds3.append(PATH_DS3 + str(i) + ".wav")

#paths_ds3 = tf.constant(np.asarray(paths_ds3))
#labels_ds3 = tf.zeros([4512], dtype=tf.dtypes.int32)

#te_dataset = tf.data.Dataset.from_tensor_slices((paths_ds3[:100], labels_ds3[:100])).map(load_wav).batch(5)


# Create datasets 
X_train = []
y_train = []
X_test = []
y_test = []

X_pred = []


print("Loading training data")
i = 0
for path, label in data_ds1:
    
    if(i == 200):
        break
    
    x, y = load_wav(path, label)
    
    X_train.append(x)
    y_train.append(y)

i=0
for path, label in data_ds2:

    if(i == 200):
        break

    
    x, y = load_wav(path, label)
    
    X_train.append(x)
    y_train.append(y)
    
   # X_test.append(x)
   # y_test.append(y)
    
print("Loading competition data")
for path in paths_ds3:
    
    x, l = load_wav(path, 0)
    X_pred.append(x)
    
 
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_pred = np.array(X_pred)    
 
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)
    



classifiers = ["""(KNeighborsClassifier(n_neighbors=9), "3-Nearest Neighbor"),
               (LinearDiscriminantAnalysis(), "Linear Discriminant Analysis"),
               (LogisticRegression(max_iter=10000), "Logistic Regression"),
               (SVC(kernel='linear'), "SVC -- linear kernel"),
               (SVC(kernel='rbf'), "SVC -- rbf kernel"),"""
               (RandomForestClassifier(n_estimators = 2000), "Random Forest")]


print("Datasets loaded, training classifier")
top_score = 0.0
# Train and test classifiers, print results
for clf, name in classifiers:
    
    # Train and test classifiers, time their performance
    start_training_time = time.time()
    clf.fit(X_train, y_train)
    elapsed_training_time = time.time() - start_training_time
    
    start_test_time = time.time()
    score = clf.score(X_test, y_test)
    elapsed_test_time = time.time()- start_test_time
    
    
    print("Results for " + name + ":", sep="")
    print("----------------------------------")
    print("Accuracy score: " + '{:.3f}'.format(score))
    print("Training time: "+ '{:.4f}'.format(elapsed_training_time) + "s")
    print("Prediction time (per sample): " + '{:.4f}'.format(1000*elapsed_test_time/len(y_train)) + "ms")
    print()
    
    if(score > top_score):
        top_score = score
        best_model = clf
        model_name = name


# Cross validation testing
print("The best model was " + model_name)
print("Cross validation scoring for " + model_name)
print("----------------------------------")

cv_scores = cross_val_score(best_model,X_test, y_test, cv=5)
mean = np.mean(cv_scores)
std = np.std(cv_scores)

print("Mean score for " + model_name + " with cross-validation: " + '{:.2f}'.format(mean))
print("Standard deviation for the cross-validation scores: " + '{:.4f}'.format(std))


predictions = best_model.predict_proba(X_pred)

print("Writing .csv file")
with open("submission_BAD.csv", "w") as fp: 
    fp.write("ID,Predicted\n") 
    for i in range(100): 
        fp.write(f"{i},{predictions[i][1]}\n")
print("\nDone")

