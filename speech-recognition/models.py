#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 17:43:51 2018

@author: chris
"""

from tensorflow.keras.layers import *
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model


#%%
class ResNet():
    """
    Usage: 
        sr = ResNet([4,8,16], input_size=(50,50,1), output_size=12)
        sr.build()
        followed by sr.m.compile(loss='categorical_crossentropy', 
                                 optimizer='adadelta', metrics=["accuracy"])
        save plotted model with: 
            keras.utils.plot_model(sr.m, to_file = '<location>.png', 
                                   show_shapes=True)
    """
    def __init__(self,
                 filters_list=[], 
                 input_size=None, 
                 output_size=None,
                 initializer='glorot_uniform'):
        self.filters_list = filters_list
        self.input_size = input_size
        self.output_size = output_size
        self.initializer = initializer
        self.m = None        
    
    def _block(self, filters, inp):
        """ one residual block in a ResNet
        
        Args:
            filters (int): number of convolutional filters
            inp (tf.tensor): output from previous layer
            
        Returns:
            tf.tensor: output of residual block
        """
        layer_1 = BatchNormalization()(inp)
        act_1 = Activation('relu')(layer_1)
        conv_1 = Conv2D(filters, (3,3), 
                        padding = 'same', 
                        kernel_initializer = self.initializer)(act_1)
        layer_2 = BatchNormalization()(conv_1)
        act_2 = Activation('relu')(layer_2)
        conv_2 = Conv2D(filters, (3,3), 
                        padding = 'same', 
                        kernel_initializer = self.initializer)(act_2)
        return(conv_2)

    def build(self):
        """
        Returns:
            keras.engine.training.Model
        """
        i = Input(shape = self.input_size, name = 'input')
        x = Conv2D(self.filters_list[0], (3,3), 
                   padding = 'same', 
                   kernel_initializer = self.initializer)(i)
        x = MaxPooling2D(padding = 'same')(x)        
        x = Add()([self._block(self.filters_list[0], x),x])
        x = Add()([self._block(self.filters_list[0], x),x])
        x = Add()([self._block(self.filters_list[0], x),x])
        if len(self.filters_list) > 1:
            for filt in self.filters_list[1:]:
                x = Conv2D(filt, (3,3),
                           strides = (2,2),
                           padding = 'same',
                           activation = 'relu',
                           kernel_initializer = self.initializer)(x)
                x = Add()([self._block(filt, x),x])
                x = Add()([self._block(filt, x),x])
                x = Add()([self._block(filt, x),x])
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.output_size, activation = 'softmax')(x)
        
        self.m = Model(i,x)
        return self.m
#%%
class ResNetLstm():
    """
    Usage: 
        sr = ResNetLstm([4,8,16], input_size=(50,50,1), output_size=12)
        sr.build()
        followed by sr.m.compile(loss='categorical_crossentropy', 
                                 optimizer='adadelta', metrics=["accuracy"])
        save plotted model with: 
            keras.utils.plot_model(sr.m, to_file = '<location>.png', 
                                   show_shapes=True)
    """
    def __init__(self,
                 filters_list=[], 
                 input_size=None, 
                 output_size=None,
                 initializer='glorot_uniform'):
        self.filters_list = filters_list
        self.input_size = input_size
        self.output_size = output_size
        self.initializer = initializer
        self.m = None        
    
    def _block(self, filters, inp):
        """ one residual block in a ResNetLstm
        
        Args:
            filters (int): number of convolutional filters
            inp (tf.tensor): output from previous layer
            
        Returns:
            tf.tensor: output of residual block
        """
        layer_1 = BatchNormalization()(inp)
        act_1 = Activation('relu')(layer_1)
        conv_1 = Conv2D(filters, (3,3), 
                        padding = 'same', 
                        kernel_initializer = self.initializer)(act_1)
        layer_2 = BatchNormalization()(conv_1)
        act_2 = Activation('relu')(layer_2)
        conv_2 = Conv2D(filters, (3,3), 
                        padding = 'same', 
                        kernel_initializer = self.initializer)(act_2)
        return(conv_2)

    def build(self):
        """
        Returns:
            keras.engine.training.Model
        """
        i = Input(shape = self.input_size, name = 'input')
        x = Conv2D(self.filters_list[0], (3,3), 
                   padding = 'same', 
                   kernel_initializer = self.initializer)(i)
        x = MaxPooling2D(padding = 'same')(x)        
        x = Add()([self._block(self.filters_list[0], x),x])
        x = Add()([self._block(self.filters_list[0], x),x])
        x = Add()([self._block(self.filters_list[0], x),x])
        if len(self.filters_list) > 1:
            for filt in self.filters_list[1:]:
                x = Conv2D(filt, (3,3),
                           strides = (2,2),
                           padding = 'same',
                           activation = 'relu',
                           kernel_initializer = self.initializer)(x)
                x = Add()([self._block(filt, x),x])
                x = Add()([self._block(filt, x),x])
                x = Add()([self._block(filt, x),x])
        x = GlobalAveragePooling2D()(x)
        
        print('shapes: ', x.shape, int(x.shape[-1]) * int(x.shape[-2]))
        x = Reshape((16,9*128))(x)
        x = Bidirectional(CuDNNLSTM(128,return_sequences=True))(x)
        x = RepeatVector(Dense(128))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Bidirectional(CuDNNLSTM(128,return_sequences=True))(x)
        x = Reshape((16,9*128))(x)
        
        x = Dense(self.output_size, activation = 'softmax')(x)
        
        self.m = Model(i,x)
        return self.m
           
    
#%%
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args    
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)    
    
        
#%%
class LSTMNet():
    def __init__(self,
                 input_size,
                 output_size,
                 layer,
                 initializer='glorot_uniform'):
        self.input_size = input_size
        self.output_size = output_size
        self.layer = layer
        self.initializer = initializer
        self.m = None

    def build(self):
        # Shape somewhat inspired by https://www.researchgate.net/publication/322927149_Bengali_speech_recognition_A_double_layered_LSTM-RNN_approach
        # but I made it bigger for no reason
        i = Input(shape = self.input_size, name = 'input')
        x = self.layer(256, kernel_initializer = self.initializer, return_sequences = True)(i)
        x = self.layer(256, kernel_initializer = self.initializer, return_sequences = True)(i)
        x = self.layer(256, kernel_initializer = self.initializer)(x)
        x = Dense(self.output_size, activation = 'softmax')(x)

        self.m = Model(i,x)
        return self.m
