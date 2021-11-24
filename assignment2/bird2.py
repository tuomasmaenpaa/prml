import tensorflow as tf
import numpy as np
import csv
import random

"""
PATH_DS1 = "/home/maenpaat/prml/assignment2/ff1010bird/wav/"
PATH_DS2 = "/home/maenpaat/prml/assignment2/warblrb10k_public/wav/"
PATH_DS3 = "/home/maenpaat/prml/assignment2/audio/"

CSV_DS1 = "/home/maenpaat/prml/assignment2/ff1010bird/ff1010bird_metadata_2018.csv"
CSV_DS2 = "/home/maenpaat/prml/assignment2/warblrb10k_public/warblrb10k_public_metadata_2018.csv"

"""
PATH_DS1 = "/Users/maenpaat/Documents/pr_n_ml/assignment2/ff1010bird/wav/"
PATH_DS2 = "/Users/maenpaat/Documents/pr_n_ml/assignment2/warblrb10k_public/wav/"
PATH_DS3 = "/Users/maenpaat/Documents/pr_n_ml/assignment2/audio/"

CSV_DS1 = "/Users/maenpaat/Documents/pr_n_ml/assignment2/ff1010bird/ff1010bird_metadata_2018.csv"
CSV_DS2 = "/Users/maenpaat/Documents/pr_n_ml/assignment2/warblrb10k_public/warblrb10k_public_metadata_2018.csv"


AUDIO_SIZE = 441000

AUDIO_SIZE = 480000
SG_WIDTH = 513
SG_HEIGHT = 936

def read_csv(csv_file, path):
    data = []
    reader = csv.reader(open(csv_file, "r"))
    next(reader)
    for row in reader:
        data.append((path + row[0] + ".wav", int(row[2])))    
    return data


def load_wav(path, label):
    audio, fs = tf.audio.decode_wav(tf.io.read_file(path))
    
    # Center to zero
    audio = audio - (tf.reduce_max(audio) + tf.reduce_min(audio)) / 2

    # Scale to [-0.5, 0.5]
    audio = audio * (0.5 / tf.reduce_max(audio))
    
    # Add padding if there are not enough samples and remove samples that exceed AUDIO_SIZE
    audio = tf.pad(audio, [[0, tf.math.abs(AUDIO_SIZE - tf.shape(audio)[0])], [0, 0]])[:AUDIO_SIZE]
    
    spectrogram = tf.raw_ops.AudioSpectrogram(input=audio, window_size=1024, stride=512)
    label = tf.one_hot(label, 2)
    return tf.expand_dims(spectrogram[0], -1), label



# Get full paths and labels
data_ds1 = read_csv(CSV_DS1, PATH_DS1)
data_ds2 = read_csv(CSV_DS2, PATH_DS2)

# Shuffle both training sets
random.shuffle(data_ds1)
random.shuffle(data_ds2)

paths_ds1  = tf.constant(np.asarray([i[0] for i in data_ds1]))
paths_ds2  = tf.constant(np.asarray([i[0] for i in data_ds2]))
labels_ds1 = tf.constant(np.asarray([i[1] for i in data_ds1]))
labels_ds2 = tf.constant(np.asarray([i[1] for i in data_ds2]))

# Create training and validation sets
tr_dataset = tf.data.Dataset.from_tensor_slices((tf.concat([paths_ds1[:6150], paths_ds2[:6400]], 0), tf.concat([labels_ds1[:6150], labels_ds2[:6400]], 0))).shuffle(12550, reshuffle_each_iteration=True).map(load_wav).batch(5)
va_dataset = tf.data.Dataset.from_tensor_slices((tf.concat([paths_ds1[6150:], paths_ds2[6400:]], 0), tf.concat([labels_ds1[6150:], labels_ds2[6400:]], 0))).map(load_wav).batch(5)

# Create test set
paths_ds3 = []
for i in range(4512):
    paths_ds3.append(PATH_DS3 + str(i) + ".wav")

paths_ds3 = tf.constant(np.asarray(paths_ds3))
labels_ds3 = tf.zeros([4512], dtype=tf.dtypes.int32)

te_dataset = tf.data.Dataset.from_tensor_slices((paths_ds3, labels_ds3)).map(load_wav).batch(5)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(SG_HEIGHT,SG_WIDTH,1)),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
    tf.keras.layers.MaxPool2D((4, 4)),
    tf.keras.layers.Conv2D(32 , (5, 5), activation='relu'),
    tf.keras.layers.MaxPool2D((4, 4)),
    tf.keras.layers.Conv2D(32 , (5, 5), activation='relu'),
    tf.keras.layers.GlobalMaxPool2D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
model.summary()
# Set training options
cb1 = tf.keras.callbacks.ModelCheckpoint('bird.h5', monitor='val_loss', verbose=1, save_best_only=True)
cb2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
opt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# Train the model
model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
model.fit(tr_dataset, epochs=1, validation_data=va_dataset, callbacks=[cb1, cb2], verbose=1)

# Predictions for test set
model = tf.keras.models.load_model('bird.h5')
print("\nCalculating predicted labels for test set")
predictions = model.predict(te_dataset)

print("Writing .csv file")
with open("submission_1e.csv", "w") as fp: 
    fp.write("ID,Predicted\n") 
    for i in range(4512): 
        fp.write(f"{i},{predictions[i][1]}\n")
print("\nDone")