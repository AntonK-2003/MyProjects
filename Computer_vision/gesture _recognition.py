from keras.models import load_model
import cv2
import numpy as np
from random import choice
import cv2
import os
import sys
import cv2
import numpy as np
from keras_squeezenet import SqueezeNet
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential
import tensorflow as tf
import os

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "nothing"
}
def mapper(val):
    return REV_CLASS_MAP[val]
def calculate_winner(user_move, Pi_move):
    if user_move == Pi_move:
        return "Tie"
    elif user_move == "rock" and Pi_move == "scissors":
        return "You" 
    elif user_move == "rock" and Pi_move == "paper":
        return "Pi"
    elif user_move == "scissors" and Pi_move == "rock":
        return "Pi"
    elif user_move == "scissors" and Pi_move == "paper":
        return "You"
    elif user_move == "paper" and Pi_move == "rock":
        return "You"
    elif user_move == "paper" and Pi_move == "scissors":
        return "Pi"
model = load_model("game-model.h5")
cap = cv2.VideoCapture(0)
prev_move = None
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.rectangle(frame, (10, 70), (300, 340), (0, 255, 0), 2)
    cv2.rectangle(frame, (330, 70), (630, 370), (255, 0, 0), 2)
    # извлекаем область изображения внутри прямоугольника пользователя
    roi = frame[70:300, 10:340]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))
    # определяем сделанный ход
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)
    # определяем победителя
    if prev_move != user_move_name:
        if user_move_name != "nothing":
            computer_move_name = choice(['rock', 'paper', 'scissors'])
            winner = calculate_winner(user_move_name, computer_move_name)
        else:
            computer_move_name = "nothing"
            winner = "Waiting..."
    prev_move = user_move_name
    # отображаем информацию на экране
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your Move: " + user_move_name,
                (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Pi's Move: " + computer_move_name,
                (330, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner,
                (100, 450), font, 2, (0, 255, 0), 4, cv2.LINE_AA)
    if computer_move_name != "nothing":
        icon = cv2.imread(
            "test_img/{}.png".format(computer_move_name))
        icon = cv2.resize(icon, (300, 300))
        frame[70:370, 330:630] = icon
    cv2.imshow("Rock Paper Scissors", frame)
    k = cv2.waitKey(10)
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
 

cam = cv2.VideoCapture(0)
start = False
counter = 0
num_samples = int(sys.argv[1])
IMG_SAVE_PATH = 'images'
try:
    os.mkdir(IMG_SAVE_PATH)
except FileExistsError:
    pass
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    if counter == num_samples:
       break
    cv2.rectangle(frame, (10, 30), (310, 330), (0, 255, 0), 2)
    k = cv2.waitKey(1)
    if k == ord('r'):
            name = 'rock'
            IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, name)
            os.mkdir(IMG_CLASS_PATH)
    if k == ord('p'):
            name = 'paper'
            IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, name)
            os.mkdir(IMG_CLASS_PATH)
    if k == ord('s'):
            name = 'scissors'
            IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, name)
            os.mkdir(IMG_CLASS_PATH)
    if k == ord('n'):
            name = 'nothing'
            IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, name)
            os.mkdir(IMG_CLASS_PATH)
    if start:
        roi = frame[25:335, 8:315]
        save_path = os.path.join(IMG_CLASS_PATH, '{}.jpg'.format(counter + 1))
        print(save_path)
        cv2.imwrite(save_path, roi)
        counter += 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,"Collecting {}".format(counter),
            (10, 20), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting images", frame)
    if k == ord('a'):
        start = not start
    if k == ord('q'):
            break
print("\n{} image(s) saved to {}".format(counter, IMG_CLASS_PATH))
cam.release()
cv2.destroyAllWindows()
 
 

IMG_SAVE_PATH = 'images'
CLASS_MAP = {
    "rock": 0,
    "paper": 1,
    "scissors": 2,
    "nothing": 3
}
NUM_CLASSES = len(CLASS_MAP)
def mapper(val):
    return CLASS_MAP[val]
def get_model():
    model = Sequential([
        SqueezeNet(input_shape=(227, 227, 3), include_top=False),
        Dropout(0.5),
        Convolution2D(NUM_CLASSES, (1, 1), padding='valid'),
        Activation('relu'),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ])
    return model
# считываем изображения из каталога
dataset = []
for directory in os.listdir(IMG_SAVE_PATH):
    path = os.path.join(IMG_SAVE_PATH, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        # убеждаемся что на нашем пути нет скрытых файлов
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (227, 227))
        dataset.append([img, directory])
data, labels = zip(*dataset)
labels = list(map(mapper, labels))
# one hot encode the labels
labels = np_utils.to_categorical(labels)
# определяем (получаем) модель
model = get_model()
model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# начинаем обучение модели
model.fit(np.array(data), np.array(labels), epochs=15)
# сохраняем модель для последующего использования
model.save("game-model.h5")
