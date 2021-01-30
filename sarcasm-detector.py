import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json

# loading the data into a dict from json
data = [json.loads(line) for line in open('./Sarcasm_Headlines_Dataset.json', 'r')]

headline = []
is_sarcastic = []
articles = []
for item in data:
    headline.append(item['headline'])
    is_sarcastic.append(item['is_sarcastic'])
    articles.append(item['article_link'])

# NLP
# tokenizer = Tokenizer(oov_token='<OOV>')
# tokenizer.fit_on_texts(headline)
# word_index = tokenizer.word_index
# print('Word Index length :', len(word_index))
# print(word_index)

# sequences = tokenizer.texts_to_sequences(headline)
# padded = pad_sequences(sequences)
# print(padded[0])
# print(padded.shape)

# Some constants definitions
vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_size = 20000

# dividing the headline data into training and testing
training_sentences = headline[:training_size]
testing_sentences = headline[training_size:]
training_labels = is_sarcastic[:training_size]
testing_labels = is_sarcastic[training_size:]

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequence = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequence, maxlen= max_length)

testing_sequence = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequence, maxlen=max_length)

# convert to numpy array to work with tensorflow 2.0
training_padded = np.array(training_padded)
testing_padded = np.array(testing_padded)
training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)

# model creation
model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ]
)

print(model.summary())

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

history = model.fit(training_padded, training_labels, epochs=30, validation_data=(testing_padded, testing_labels), verbose=2)

print(training_padded[1])
print(training_sentences[1])
print(is_sarcastic[1])

## lets test the model
sentences = [
    "granny starting to fear spiders in the garden might be real",
    "game of thrones season finale showing this sunday night"
]
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type,truncating=trunc_type)
print(model.predict(padded))




