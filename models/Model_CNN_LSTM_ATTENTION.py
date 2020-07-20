
"""
Anthor:liu jia ming
Date:2020/7
Theme:AQI Prediction
"""
#-*- coding: utf-8 -*-
import sys
sys.path.append("../")
from keras.layers import Input, Dense, LSTM, merge ,Conv1D,Dropout,Bidirectional,Multiply
from keras.models import Model
from utils.attention_utils import get_activations
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras import backend as K
import  pandas as pd
import  numpy as np
single_attention_vector = False

def attention_3d_block(inputs):
	time_steps = K.int_shape(inputs)[1]
	input_dim = K.int_shape(inputs)[2]
	a = Permute((2, 1))(inputs)
	a = Dense(time_steps, activation='softmax')(a)
	if single_attention_vector:
	    a = Lambda(lambda x: K.mean(x, axis=1))(a)
	    a = RepeatVector(input_dim)(a)

	a_probs = Permute((2, 1))(a)
	output_attention_mul = Multiply()([inputs, a_probs]) 
	return output_attention_mul

def attention_3d_block2(inputs, single_attention_vector=False):
	time_steps = K.int_shape(inputs)[1]
	input_dim = K.int_shape(inputs)[2]
	a = Permute((2, 1))(inputs)
	a = Dense(time_steps, activation='softmax')(a)
	if single_attention_vector:
	    a = Lambda(lambda x: K.mean(x, axis=1))(a)
	    a = RepeatVector(input_dim)(a)
	a_probs = Permute((2, 1))(a)
	output_attention_mul = Multiply()([inputs, a_probs])
	return output_attention_mul

def attention_model(DAY_STEPS,INPUT_DIMS,lstm_units):
	inputs = Input(shape=(DAY_STEPS, INPUT_DIMS))

	# x = Conv1D(filters = 64, kernel_size = 1, activation = 'relu')(inputs)  #, padding = 'same'
	# x = Dropout(0.3)(x)
	# lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(inputs)
	lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
	lstm_out = Dropout(0.3)(lstm_out)
	lstm_out = LSTM(lstm_units, return_sequences=True)(lstm_out)
	lstm_out = Dropout(0.3)(lstm_out)
	# lstm_out = LSTM(lstm_units, return_sequences=True)(lstm_out)
	# lstm_out = Dropout(0.3)(lstm_out)

	# attention_mul = attention_3d_block(lstm_out)
	# attention_mul = Flatten()(attention_mul)
	lstm_out = Flatten()(lstm_out)
	output = Dense(1, activation='sigmoid')(lstm_out)
	model = Model(inputs=[inputs], outputs=output)
	return model

