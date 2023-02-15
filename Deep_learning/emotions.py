model = Sequential()
model.add(Conv1D(256, 5,padding='same', input_shape=(216,1))) #1
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same')) #2
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same')) #3
model.add(Activation('relu'))
#model.add(Conv1D(128, 5,padding='same')) #4
#model.add(Activation('relu'))
#model.add(Conv1D(128, 5,padding='same')) #5
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Conv1D(128, 5,padding='same')) #6
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(10)) #7
model.add(Activation('softmax'))
opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

# Указываем нужное кол-во классов
target_class = 5
# Модель 
model = Sequential()
model.add(Conv1D(256, 8, padding='same',input_shape=(X_train.shape[1],1))) #1
model.add(Activation('relu'))
model.add(Conv1D(256, 8, padding='same')) #2
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 8, padding='same')) #3
model.add(Activation('relu')) 
model.add(Conv1D(128, 8, padding='same')) #4
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same')) #5
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same')) #6
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(64, 8, padding='same')) #7
model.add(Activation('relu'))
model.add(Conv1D(64, 8, padding='same')) #8
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(target_class)) #9
model.add(Activation('softmax'))
opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)

lr_reduce = ReduceLROnPlateau(monitor=’val_loss’, factor=0.9, patience=20, min_lr=0.000001)
mcp_save = ModelCheckpoint(‘model/baseline_2class_np.h5’, save_best_only=True, monitor=’val_loss’, mode=’min’)
cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=700, validation_data=(x_testcnn, y_test), callbacks=[mcp_save, lr_reduce])

def dyn_change(data):
   """
   Случайное изменение значений
   """
   dyn_change = np.random.uniform(low=1.5,high=3)
   return (data * dyn_change)

def pitch(data, sample_rate):
   """
   Настройка высоты звука
   """
   bins_per_octave = 12
   pitch_pm = 2
   pitch_change =  pitch_pm * 2*(np.random.uniform())  
   data = librosa.effects.pitch_shift(data.astype('float64'),
                                     sample_rate, n_steps=pitch_change,
                                     bins_per_octave=bins_per_octave)

def shift(data):
   """
   Случайное смещение
   """
   s_range = int(np.random.uniform(low=-5, high = 5)*500)
   return np.roll(data, s_range)


def noise(data):
   """
   Добавление белого шума
   """
   # можете взять любой дистрибутив отсюда: https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
   noise_amp = 0.005*np.random.uniform()*np.amax(data)
   data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
   return data

