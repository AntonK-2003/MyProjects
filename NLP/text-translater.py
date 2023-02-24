import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages are logged, 3 - INFO, WARNING, and ERROR messages are not printed

import string 
import re
import numpy as np
import pandas as pd
from keras.models import Sequential 
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint 
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model 
from keras import optimizers 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', 200)

# Read raw text file
def read_text(file):
    with open(file, mode='rt', encoding='utf-8') as file:
        txt = file.read()
        sents = txt.strip().split('\n')
        return [i.split('\t') for i in sents]

data = read_text("deutch.txt")
rus_eng = np.array(data)

rus_eng = rus_eng[:30000,:]
print("Dictionary size:", rus_eng.shape)

# Remove punctuation 
rus_eng[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in rus_eng[:,0]] 
rus_eng[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in rus_eng[:,1]] 

# Convert text to lowercase 
for i in range(len(rus_eng)): 
    rus_eng[i,0] = ru_eng[i,0].lower() 
    ru_eng[i,1] = ru_eng[i,1].lower()
    
# Prepare English tokenizer
eng_tr = Tokenizer()
eng_tr.fit_on_texts(rus_eng[:, 0])
eng_vsize = len(eng_tr.word_index) + 1 
eng_length = 8 

# Prepare Deutch tokenizer 
rus_tr = Tokenizer()
rus_tr.fit_on_texts(rus_eng[:, 1])
rus_vsize = len(rus_tr.word_index) + 1 
rus_length = 8 

# Encode and pad sequences 
def encode_sequences(tr, length, lines):          
    # integer encode sequences
    seq = tr.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq
     
# Split data into train and test set 
train, test = train_test_split(rus_eng, test_size=0.2, random_state=12)

# Prepare training data 
trainX = encode_sequences(eng_tr, eng_length, train[:, 0])
trainY = encode_sequences(rus_tr, deu_length, train[:, 1])

# Prepare validation data 
testX = encode_sequences(eng_tr, eng_length, test[:, 0])
testY = encode_sequences(rus_tr, deu_length, test[:, 1])

# Build NMT model 
def make_model(in_vocab, out_vocab, in_timesteps, out_timesteps, n):
    model = Sequential()
    model.add(Embedding(in_vocab, n, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(n))
    model.add(Dropout(0.3))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(n, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Dense(out_vocab, activation='softmax'))
    model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='sparse_categorical_crossentropy')
    return model

print("Russian vocabulary size:", rus_vsize, deu_length)
print("English vocabulary size:", eng_vsize, eng_length)

# Model compilation (with 512 hidden units)
model = make_model(eng_vsize, rus_vsize, eng_length, rus_length, 512)

# Train model
num_epochs = 250
history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1), epochs=num_epochs, batch_size=512, validation_split=0.2, callbacks=None, verbose=1)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['train','validation'])
# plt.show()
model.save('en-de-model.h5')

# Load model
model = load_model('en-de-model.h5')

def get_word(n, tokenizer):
    if n == 0:
        return ""
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return ""


phrs_enc = encode_sequences(eng_tokenizer, eng_length, ["the weather is nice today", "my name is tom", "how old are you", "where is the nearest shop"])
print("phrs_enc:", phrs_enc.shape)

preds = model.predict_classes(phrs_enc)
print("Preds:", preds.shape)
print(preds[0])
print(get_word(preds[0][0], rus_tr), get_word(preds[0][1], rus_tr), get_word(preds[0][2], rus_tr), get_word(preds[0][3], rus_tr))
print(preds[1])
print(get_word(preds[1][0], rus_tr), get_word(preds[1][1], rus_tr), get_word(preds[1][2], rus_tr), get_word(preds[1][3], rus_tr))
print(preds[2])
print(get_word(preds[2][0], rus_tr), get_word(preds[2][1], rus_tr), get_word(preds[2][2], rus_tr), get_word(preds[2][3], rus_tr))
print(preds[3])
print(get_word(preds[3][0], rus_tr), get_word(preds[3][1], rus_tr), get_word(preds[3][2], rus_tr), get_word(preds[3][3], rus_tr))
print()
