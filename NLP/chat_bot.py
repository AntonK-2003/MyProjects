import csv
import pymorphy2
import re


import psycopg2
conn = psycopg2.connect(dbname='energy', user='mao', password='daring', host='localhost')
cursor = conn.cursor()

morph = pymorphy2.MorphAnalyzer(lang='ru')


answer_id=[] 
answer = dict()

cursor.execute('SELECT id, answer FROM app.chats_answer;')
records = cursor.fetchall()
for row in records:
 answer[row[0]]=row[1]

questions=[] 

cursor.execute('SELECT question, answer_id FROM app.chats_question;')
records = cursor.fetchall()
transform=0

for row in records:
 if row[0]>"":
  if row[1]>0:
   phrases=row[0]
   words=phrases.split(' ')
   phrase=""
   for word in words:
    word = morph.parse(word)[0].normal_form  
    phrase = phrase + word + " "
   if (len(phrase)>0):
    questions.append(phrase.strip())
    answer_id.append(row[1])
    transform=transform+1



cursor.close()
conn.close()

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

vectorizer_q = TfidfVectorizer()
vectorizer_q.fit(questions)
matrix_big_q = vectorizer_q.transform(questions)
print ("Размер матрицы: ")
print (matrix_big_q.shape)

if transform>200:
 transform=200
print(transform)
svd_q = TruncatedSVD(n_components=transform)
svd_q.fit(matrix_big_q)
matrix_small_q = svd_q.transform(matrix_big_q)
print ("Коэффициент уменьшения матрицы: ")
print ( svd_q.explained_variance_ratio_.sum())


# тело программы k=5, temperature=10.0 можно подбирать
import numpy as np

from sklearn.neighbors import BallTree
from sklearn.base import BaseEstimator

def softmax(x):
  #создание вероятностного распределения
  proba = np.exp(-x)
  return proba / sum(proba)

class NeighborSampler(BaseEstimator):
  def __init__(self, k=5, temperature=10.0):
    self.k=k
    self.temperature = temperature
  def fit(self, X, y):
    self.tree_ = BallTree(X)
    self.y_ = np.array(y)
  def predict(self, X, random_state=None):
    distances, indices = self.tree_.query(X, return_distance=True, k=self.k)
    result = []
    for distance, index in zip(distances, indices):
      result.append(np.random.choice(index, p=softmax(distance * self.temperature)))
    return self.y_[result]

from sklearn.pipeline import make_pipeline

ns_q = NeighborSampler()
ns_q.fit(matrix_small_q, answer_id) 
pipe_q = make_pipeline(vectorizer_q, svd_q, ns_q)


import re
import telebot
telebot.apihelper.ENABLE_MIDDLEWARE = True
bot = telebot.TeleBot("299999999:sdfgnreognrtgortgmrtgmrtgm")

@bot.message_handler(commands=['start'])
def start_message(message):
	bot.send_message(message.from_user.id, " Здравствуйте. Я Ваш виртуальный помощник")

@bot.message_handler(func=lambda message: True)
def get_text_messages(message):
	request=message.text
	words= re.split('\W',request)
	phrase=""
	for word in words:
		word = morph.parse(word)[0].normal_form  
		phrase = phrase + word + " "
	reply_id    = int(pipe_q.predict([phrase.strip()]))
	bot.send_message(message.from_user.id, answer[reply_id])
	print("Запрос:", request, " \n\tНормализованный: ", phrase, " \n\t\tОтвет :", answer[reply_id])

bot.infinity_polling(none_stop=True, interval=1)


print("Ваш запрос: (для выхода - exit)")
request=""
while request not in ['exit']:
 request=input()
 words= re.split('\W',request)
 phrase=""
 for word in words:
  word = morph.parse(word)[0].normal_form  
  phrase = phrase + word + " "
 reply_id    = int(pipe_q.predict([phrase.strip()]))
 print (answer[reply_id])
