import tensorflow as tf
import numpy as np
import cv2
import random

# If using Windows, it's recommended to disable real-time virus protection for much faster loading:
# Windows Security -> Virus & threat protection -> Real-time protection

N_LABELS = 10
N_TRAINING_IMAGES = 60000
N_TEST_IMAGES = 10000
SIZE = 28
PATH_TRAIN = "/Users/maenpaat/Documents/pr_n_ml/assignment1/tau-ethiopic-digit-recognition/train/train/"
PATH_TEST  = "/Users/maenpaat/Documents/pr_n_ml/assignment1/tau-ethiopic-digit-recognition/test/test/"


def preprocess_image(path):
    # Load the image and binarize it
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]

    # Find the character and move it to the center of the image
    br = cv2.boundingRect(img_bin)
    M = np.float32([[1, 0, (SIZE-2*br[0]-br[2])/2], [0, 1, (SIZE-2*br[1]-br[3])/2]])
    img_bin = cv2.warpAffine(img_bin, M, img.shape)
    img = cv2.warpAffine(img, M, img.shape)

    # Find outer contours, minimum area rectangle and its angle
    contours = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, 2)
    if len(contours[0]) == 1:
        contours = contours[0][0]
    else:
        contours = contours[0][1]

    rect = cv2.minAreaRect(contours)
    angle = rect[2] if rect[2] > -45 else 90 + rect[2]

    # Correct the rotation with respect to center point and zoom in if needed
    M = cv2.getRotationMatrix2D((SIZE/2, SIZE/2), angle, (SIZE-2)/max(br[2:4]))
    img = cv2.warpAffine(img, M, img.shape)
    
    return np.reshape(img.astype(np.float32)/255, (SIZE,SIZE,1))


# Construct a list of pairs (full path, label) for all images in train folder    
image_list = []
for label in range(1, N_LABELS + 1):
    for image_id in range(6000):
        image_list.append((PATH_TRAIN + str(label) + f"/{image_id+1:05}.jpg", label))

# Randomize the order of images
random.shuffle(image_list)
paths  = [i[0] for i in image_list]
labels = [i[1] for i in image_list]

# Load training set
training_images = np.zeros((N_TRAINING_IMAGES, SIZE, SIZE, 1), dtype=np.float32)
training_labels = np.zeros((N_TRAINING_IMAGES, N_LABELS), dtype=np.float32)

print("Loading training set")
for i in range(N_TRAINING_IMAGES):
    print(i+1, "/", N_TRAINING_IMAGES, sep="", end="\r", flush=True)
    training_images[i,:,:,:] = preprocess_image(paths[i])
    training_labels[i,:] = tf.one_hot(labels[i]-1, N_LABELS).numpy()

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (5, 5), activation='relu', input_shape=(SIZE,SIZE,1)),
    tf.keras.layers.Conv2D(24, (5, 5), activation='relu'),
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
    tf.keras.layers.Conv2D(40, (5, 5), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(N_LABELS)
])
model.summary()

# Set training options
cb1 = tf.keras.callbacks.ModelCheckpoint('ethiopic.h5', monitor='val_loss', verbose=1, save_best_only=True)
cb2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# Train the model
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.fit(training_images, training_labels, batch_size=5, epochs=50, validation_split=1/6, callbacks=[cb1, cb2], verbose=1)

# Load the best checkpoint
model = tf.keras.models.load_model('ethiopic.h5')

# Load test images and calculate predicted labels
test_images = np.zeros((N_TEST_IMAGES, SIZE, SIZE, 1), dtype=np.float32)

print("\nLoading test set")
for i in range(N_TEST_IMAGES):
    print(i+1, "/", N_TEST_IMAGES, sep="", end="\r", flush=True)
    test_images[i,:,:,:] = preprocess_image(PATH_TEST + f"{i:05}.jpg")

print("\nCalculating predicted labels for test set")
predictions = model.predict(test_images)
predictions = np.argmax(predictions, axis=1)

print("Writing .csv file")
with open("submission.csv", "w") as fp: 
    fp.write("Id,Category\n") 
    for i in range(N_TEST_IMAGES): 
        fp.write(f"{i:05},{predictions[i]+1}\n")
print("\nDone")
