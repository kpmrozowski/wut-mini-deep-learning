import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LSTM, GRU
from time import time
from models import ResNet, LSTMNet
from collections import Counter
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix   
import wave
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (20,7)

def load_wave(filename):
   # Read file to get buffer
   ifile = wave.open(filename)
   samples = ifile.getnframes()
   audio = ifile.readframes(samples)

   # Convert buffer to float32 using NumPy
   audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
   audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

   # Normalise float32 array so that values are between -1.0 and +1.0
   max_int16 = 2**15
   audio_normalised = audio_as_np_float32 / max_int16
   return audio_normalised

print('\nimport training and validation data not included in repo, set your own folder location:')
# import training and validation data not included in repo, set your own folder location
X_train = np.load('data/X_train.npy')
Y_train = np.load('data/Y_train.npy')
X_val = np.load('data/X_val.npy')
Y_val = np.load('data/Y_val.npy')

classes = ['yes', 'no', 
           'up', 'down', 
           'left', 'right', 
           'on', 'off', 
           'stop', 'go', 
           'silence', 'unknown']

# all_classes = [x for x in classes[:11]]
# for ind, cl in enumerate(os.listdir('data/train/audio/')):
#     if cl not in classes:
#         all_classes.append(cl)
all_classes = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'bird', 'bed', 'house', 'seven', 'six', 'nine', 'dog', 'two', 'eight', 'four', 'tree', 'zero', 'marvin', 'happy', 'sheila', 'wow', 'three', 'one', 'cat', 'five']
print(all_classes)

def get_class_weights(y):
   counter = Counter(y)
   majority = max(counter.values())
   return  {cls: float(majority/count) for cls, count in counter.items()}  

class_weights = get_class_weights(Y_train)
print('\nclass_weights =', class_weights)

# some constants we need for all models
input_size = X_train.shape[1:]
batch_size = 196

## First the ResNet
# declare filters for each block of blocks and set output size.
filters_list = [8,16,32]
output_size = 12

#adjust these strings for organizeing the saved files
date = '20220410'
arch = 'resnet8_16_32'

# Build the model
sr = ResNet(filters_list, input_size, output_size)
# sr = LSTMNet(input_size[:-1], output_size, LSTM)
# sr = LSTMNet(input_size[:-1], output_size, GRU)
sr.build()
sr.m.compile(loss='categorical_crossentropy', 
             optimizer='adadelta', 
             metrics=['accuracy'])

# to save a png of the model you need pydot and graphviz installed
if not os.path.exists('./models'):
   os.makedirs('./models')
plot_model(sr.m, 
           to_file = './models/{}_{}.png'.format(arch,date), 
           show_shapes = True)

#callbacks, remember to make folders to store files 
checkpointer = ModelCheckpoint(filepath='./models/{}_{}_best.h5'.format(arch, date),
                               verbose=0,
                               save_best_only=True)
   
earlystopping = EarlyStopping()

if not os.path.exists('./logs'):
   os.makedirs('./logs')
tensorboard = TensorBoard(log_dir = './logs/{}_{}'.format(date, time()), 
                          histogram_freq = 0, 
                          write_graph = True, 
                          write_images = True)

# train the model
# the history object stores training data for later access, like plotting training curves
history = sr.m.fit(X_train, 
                   to_categorical(Y_train), 
                   batch_size = batch_size, 
                   epochs = 5000, 
                   verbose = 1, shuffle = True, 
                   class_weight = class_weights,
                   validation_data = (X_val, to_categorical(Y_val)), 
                   callbacks = [checkpointer, tensorboard]) # add more callbacks if you want

sr.m.save_weights("./models/{}_{}_last.h5".format(arch, date)) 

print('\nplot the training graphs, and save them:')
# plot the training graphs, and save them
#%% visualize training
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('graphs/{}_{}_acc.png'.format(arch, date),bbox_inches='tight')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('graphs/{}_{}_loss.png'.format(arch, date), bbox_inches='tight')

print('\npredict the validation classification and score:')
# predict the validation classification and score
val_pred = sr.m.predict(X_val, batch_size = batch_size, verbose = 1)
print(classification_report(Y_val, np.argmax(val_pred, axis = 1), target_names = classes, digits = 3))
print('\nALL DONE!')

