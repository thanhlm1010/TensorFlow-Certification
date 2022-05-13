import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.text import Tokenizer

sentences = ['Today is a sunny day',
             'Today is a rainy day',
             'Is it sunny today']

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

