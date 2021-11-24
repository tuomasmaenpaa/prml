import tensorflow as tf
import csv

AUDIO_SIZE = 441000


def read_csv(csv_file, path):
    data = []
    reader = csv.reader(open(csv_file, "r"))
    next(reader)
    for row in reader:
        data.append((path + row[0] + ".wav", int(row[2])))    
    return data


def load_wav(path, label):
    audio, fs = tf.audio.decode_wav(tf.io.read_file(path))
    
    # Scale to [-1, 1]
    audio = audio / tf.math.abs(tf.reduce_max(audio))
    
    
    # Add padding if there are not enough samples and remove samples that exceed AUDIO_SIZE
    audio = tf.pad(audio, [[0, tf.math.abs(AUDIO_SIZE - tf.shape(audio)[0])], [0, 0]])[:AUDIO_SIZE]
    
    spectrogram = tf.raw_ops.AudioSpectrogram(input=audio, window_size=1024, stride=512)
    spectrogram = tf.math.log(tf.math.exp(tf.constant(1.0)) + spectrogram)
    spectrogram = spectrogram / tf.reduce_max(spectrogram)
    #spectrogram = tf.transpose(spectrogram)
    label = tf.one_hot(label, 2)
    return tf.expand_dims(spectrogram[0], -1), label