import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import backend as K
from time import time
from models import ResNet
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

print('import training and validation data not included in repo, set your own folder location:')
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

all_classes = [x for x in classes[:11]]
for ind, cl in enumerate(os.listdir('data/train/audio/')):
    if cl not in classes:
        all_classes.append(cl)
print(all_classes)

def get_class_weights(y):
   counter = Counter(y)
   majority = max(counter.values())
   return  {cls: float(majority/count) for cls, count in counter.items()}  

class_weights = get_class_weights(Y_train)
print('class_weights =', class_weights)

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
sr.build()
sr.m.compile(loss='categorical_crossentropy', 
             optimizer='adadelta', 
             metrics=['accuracy'])

# to save a png of the model you need pydot and graphviz installed
if not os.path.exists('/models'):
   os.makedirs('/models')
plot_model(sr.m, 
           to_file = './models/{}_{}.png'.format(arch,date), 
           show_shapes = True)

#callbacks, remember to make folders to store files 
checkpointer = ModelCheckpoint(filepath='./models/{}_{}_best.h5'.format(arch, date),
                               verbose=0,
                               save_best_only=True)
   
earlystopping = EarlyStopping()

if not os.path.exists('/logs'):
   os.makedirs('/logs')
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
                   callbacks = [checkpointer]) # add more callbacks if you want

sr.m.save_weights("./models/{}_{}_last.h5".format(arch, date)) 

print('plot the training graphs, and save them:')
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

print('predict the validation classification and score:')
# predict the validation classification and score
val_pred = sr.m.predict(X_val, batch_size = batch_size, verbose = 1)
print(classification_report(Y_val, np.argmax(val_pred, axis = 1), target_names = classes, digits = 3))


## Now the CTC model
from models import CTC, ctc_lambda_func #used in the CTC build method
from ctc_utils import char_map, index_map, text_to_int, get_intseq, get_ctc_params

# dummy loss
def ctc(y_true, y_pred):
    return y_pred

# The Conv1D layer at the top of the CTC model takes a 3-dimensional input, not 4 as the ResNet
sr_ctc = CTC((122,85), 28)
sr_ctc.build()

sr_ctc.m.compile(loss = ctc, optimizer = 'adam', metrics = ['accuracy'])
sr_ctc.tm.compile(loss = ctc, optimizer = 'adam')

Y_train_all = np.load('data/Y_train_all.npy')
Y_val_all = np.load('data/Y_val_all.npy')

# get the ctc parameters needed for the three extra ctc model inputs
labels, input_length, label_length = get_ctc_params(Y = Y_train_all, classes_list = all_classes)
labels_val, input_length_val, label_length_val = get_ctc_params(Y = Y_val_all, classes_list = all_classes)

checkpointer = ModelCheckpoint(filepath="./models/ctc_{}_best.h5".format(date),
                               verbose=0,
                               save_best_only=True)

# fit the model
history = sr_ctc.m.fit([np.squeeze(X_train), 
                            labels, 
                            input_length, 
                            label_length], 
                       np.zeros([len(Y_train_all)]), 
                       batch_size = 128, 
                       epochs = 5000, 
                       validation_data = ([np.squeeze(X_val), 
                                           labels_val, 
                                           input_length_val, 
                                           label_length_val],
                                          np.zeros([len(Y_val_all)])), 
                       callbacks = [checkpointer], 
                       verbose = 1, shuffle = True)

sr_ctc.m.save_weights('./models/ctc_{}.h5'.format(date))
sr_ctc.tm.load_weights('./models/ctc_{}_best.h5'.format(date))

print('ctc-plot:')
# plot
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('graphs/ctc_{}_acc.png'.format(date),bbox_inches='tight')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('graphs/ctc_{}_loss.png'.format(date), bbox_inches='tight')

print('X_val.shape =', X_val.shape)

def str_out(dataset = X_val):
    k_ctc_out = K.ctc_decode(sr_ctc.tm.predict(np.squeeze(dataset), 
                                                verbose = 1), 
                             np.array([28 for _ in dataset]))
    decoded_out = K.eval(k_ctc_out[0][0])
    str_decoded_out = []
    for i, _ in enumerate(decoded_out):
        to_join = [index_map[c] for c in decoded_out[i] if not c == -1]
        str_decoded_out.append("".join(to_join))
        
    return str_decoded_out

y_pred_val = str_out()

# MAGIC! Turn the target array Y_val_all into strings with 'all_classes[Y_val_all[i]]'
print('MAGIC! Turn the target array Y_val_all into strings with all_classes[Y_val_all[i]]:')
print('PREDICTED: \t REAL:')
for i in range(10):
    print(y_pred_val[i], '\t\t',all_classes[Y_val_all[i]])

print(classification_report([all_classes[Y_val_all[i]] for i, _ in enumerate(Y_val_all)], y_pred_val, labels = all_classes))

plt.figure(figsize = (8,8))
plt.imshow(confusion_matrix([all_classes[Y_val_all[i]] for i, _ in enumerate(Y_val_all)], 
                            y_pred_val, labels = all_classes))
plt.xticks(np.arange(0, len(all_classes)), all_classes, rotation = 'vertical', size = 12)
plt.yticks(np.arange(0, len(all_classes)), all_classes, size = 12)
plt.savefig('graphs/ctc_{}_confusion_matrix.png'.format(date), bbox_inches='tight')
print('ALL DONE!')

