import os, csv, cv2
import numpy as np
import tensorflow as tf
import sklearn

from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D


def prepare_data():
    """
    Reading Center, Left, Right Image pahts from driving_log.csv
    """
    
    lines = []
    with open('./mydataIMG/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    measurments = []
    
    CenterImgPaths = []
    LeftImgPaths = []
    RightImgPaths = []
    
    for line in lines:
        source_path = line[0]
        CenterImgPaths.append(line[0].split('/')[-1])
        LeftImgPaths.append(line[1].split('/')[-1])
        RightImgPaths.append(line[2].split('/')[-1])
        measurments.append(float(line[3]))
        
    return (CenterImgPaths, LeftImgPaths, RightImgPaths, measurments)

def all_sensor_data(center, left, right, measruments, corr):
    """
    Combine Center, Left, Right images and adding correction factor to measurments
    """
    imagePaths = []
    imagePaths.extend(center)
    imagePaths.extend(left)
    imagePaths.extend(right)
    corr_measurments = []

    corr_measurments.extend(measurments)
    for m in measurments:
        corr_measurments.extend([float(m) + corr])
    for m in measurments:
        corr_measurments.extend([float(m) - corr])
    
    return (imagePaths, corr_measurments)

def generator(train_set, batch_size=128):
    """
     Generate the required images and measruments for training
     train_set - combination of images and respective measurments
    """
    no_of_samples = len(train_set)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(train_set)
        for idx in range(0, no_of_samples, batch_size):
            batch_samples = samples[idx: idx + batch_size]

            images = []
            theta = []
            
            for imgPath, measurment in batch_samples:
                img = cv2.imread(imgPath)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(rgb)
                theta.append(measurment)
                # Flipping
                images.append(cv2.flip(rgb, 1))
                theta.append(-measurment)
            inputs = np.array(images)
            outputs = np.array(theta)
            yield sklearn.utils.shuffle(inputs, outputs)
            

def normalization():
    """
    Model with the pre-processing layers.
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model

def network():

    model = normalization()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, subsample=(1,1), activation='relu'))
    model.add(Convolution2D(64,3,3, subsample=(1,1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model
    
# Reading images locations.
CenterImgPaths, LeftImgPaths, RightImgPaths, measurments = prepare_data()

imagePaths, measurements = all_sensor_data(CenterImgPaths, LeftImgPaths, RightImgPaths, measurments, 0.065)
print('Total Images: {}'.format( len(imagePaths)))

# train test split and creating generators.
samples = list(zip(imagePaths, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Total Train samples: {}'.format(len(train_samples)))
print('Total Validation samples: {}'.format(len(validation_samples)))

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Model creation
model = network()

print("total train samples", len(train_samples))
print("total val samples", len(validation_samples))

# Compiling and training the model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
                 int(np.ceil(len(train_samples)/32)), nb_epoch=5, validation_data=validation_generator, \
                 nb_val_samples=int(np.ceil(len(validation_samples)/32)), verbose=1)

model.save('model.h5')
    
