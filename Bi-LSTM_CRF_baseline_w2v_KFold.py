#%% Import modules & Define functions
import re
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
from contextlib import redirect_stdout
from sklearn.model_selection import train_test_split, KFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, SGD, RMSprop, Adamax, Nadam
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Embedding, LSTM, Bidirectional, TimeDistributed
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from keras_contrib.layers import CRF
from gensim.models import KeyedVectors

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')

            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def sequence_to_tag(sequences):
    result = []
    for sequence in sequences:
        temp = []
        for pred in sequence:
            pred_index = np.argmax(pred)
            temp.append(index_to_ner[pred_index].replace('PAD', 'O'))
        result.append(temp)
    return result


def plot(hist, epochnum, batchnum, savepath, fullname):
    train_loss = hist['loss']
    val_loss = hist['val_loss']
    acc = hist['crf_viterbi_accuracy']
    val_acc = hist['val_crf_viterbi_accuracy']

    plt.figure()
    epochs = np.arange(1, len(train_loss) + 1, 1)
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.title('{}, Epochs={}, Batch={}'.format(fullname, epochnum, batchnum))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.show() # <- When we run this before 'savefig', blank image files are generated.
    plt.savefig('{}/{} - Loss.png'.format(savepath, fullname), format='png')

    plt.figure()
    plt.plot(epochs, acc, 'b', label='Training score')
    plt.plot(epochs, val_acc, 'r', label='Validation score')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.title('{}, Epochs={}, Batch={}'.format(fullname, epochnum, batchnum))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    # plt.show()
    plt.savefig('{}/{} - Accuracy.png'.format(savepath, fullname), format='png')


losses = {'crf': 'crf loss'}
losses_list = list(losses.keys())  # -> index : l, usage -> losses[losses_list[l]]
optimizers = {'adam': Adam(lr=1e-3),
              'rmsprop': RMSprop(lr=1e-3),
              'sgd': SGD(lr=0.1e-3, momentum=0.90, decay=1e-6),
              'adamax': Adamax(lr=1e-3),
              'nadam': Nadam(lr=1e-3)}
optimizers_list = list(optimizers.keys())


#%% Settings
epochnum = 10
batchnum = 128
lossname = 'crf'
optsname = 'rmsprop'  # 'adam', 'nadam', 'rmsprop'
modelname = 'Bi-LSTM-CRF-w2v baseline 5-Fold'
dataname = 'D1'
epochname = '10'
loss = losses[lossname]
optimizer = optimizers[optsname]
w2vname = 'w2v_128_5_3_4_0'
num_folds = 5

#%% Load Data & Get Sentence+Tag
root_dir = os.getcwd()
data_dir = os.path.join(root_dir, 'data')
save_dir = os.path.join(root_dir, 'results', modelname)
weight_dir = os.path.join(root_dir, 'weights', modelname)
annotation_dir = data_dir + '\\data.txt'
w2v_dir = os.path.join(root_dir, 'word2vec')
w2v_file = w2v_dir + '\\' + w2vname + '.bin'

makedirs(save_dir)
makedirs(weight_dir)

f = open(annotation_dir, 'r', encoding='UTF8')
tagged_sentences = []
sentence = []

for line in f:
    # for checking start, end line of report & Day information of reports
    if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == '\n' or line.find(',') > 0:
        if len(sentence) > 0:
            # Append last line to tagged_sentences
            tagged_sentences.append(sentence)
            sentence = []
        continue
    splits = line.split(' ')
    splits[-1] = re.sub(r'\n', '', splits[-1])
    word = splits[0].lower()
    sentence.append([word, splits[-1]])

#%% Check tagged_sentences & print some part of them
print(len(tagged_sentences))
print(tagged_sentences[0])

#%% Parse sentences & tags from tagged_sentences
sentences, ner_tags = [], []
for tagged_sentence in tagged_sentences:
    sentence, tag_info = zip(*tagged_sentence)
    sentences.append(list(sentence))
    ner_tags.append(list(tag_info))

#%% Check parsed results
print(sentences[0])
print(ner_tags[0])

#%% Check max & average length of sentences
max_length = max(len(sentence) for sentence in sentences)
avg_length = (sum(map(len, sentences)) / len(sentences))
print('max length: %d' % max_length)    # --> 68
print('average length: %f' % avg_length) # --> 5.527604
# Draw histogram of distribution for number of samples
plt.figure()
plt.hist([len(sentence) for sentence in sentences], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.savefig(save_dir+'/Sentence distribution.png')
plt.show()

#%% Make OOV for infrequent words (over 2000 words)
# max_words = 1700
# src_tokenizer = Tokenizer(num_words=max_words, oov_token='OOV')

src_tokenizer = Tokenizer()
src_tokenizer.fit_on_texts(sentences)
max_words = len(src_tokenizer.word_index) + 1

tar_tokenizer = Tokenizer()
tar_tokenizer.fit_on_texts(ner_tags)

#%% Print sizes
word_size = len(src_tokenizer.word_index) + 1
vocab_size = max_words
tag_size = len(tar_tokenizer.word_index) + 1
print('original word size: {}'.format(word_size))
print('processed word size: {}'.format(vocab_size))
print('tag size: {}'.format(tag_size))

#%% Integer encoding
X = src_tokenizer.texts_to_sequences(sentences)
y = tar_tokenizer.texts_to_sequences(ner_tags)

#%% Check encoded sentences
print(sentences[1000])
print(X[1000])
print(ner_tags[1000])
print(y[1000])

#%% Decode data
index_to_word = src_tokenizer.index_word
index_to_ner = tar_tokenizer.index_word
# Check 0 index's tag
index_to_ner[0] = 'PAD'
print(index_to_ner)
# Compare original texts and OOV processed texts
# for i in range(len(X_train)):
#     decoded = []
#     for index in X_train[i]:
#         decoded.append(index_to_word[index])
#     print('original text: {}'.format(sentences[i]))
#     print('OOV processed: {}'.format(decoded))

#%% Pad sequences
X = pad_sequences(X, padding='post', maxlen=max_length)
y = pad_sequences(y, padding='post', maxlen=max_length)

#%% Split train/test set
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.2, random_state=777)

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1

for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    fullname = modelname+'_'+lossname+'_'+optsname+'_'+dataname+'_'+w2vname+'_'+str(fold_no)  # data compare

    #%% One-hot encoding for y
    y_train = to_categorical(y_train, num_classes=tag_size)
    y_test = to_categorical(y_test, num_classes=tag_size)

    #%% Check padded sequence and data shape
    print('The size of training samples: {}'.format(X_train.shape))
    print('The size of training samples\' label: {}'.format(y_train.shape))
    print('The size of test samples: {}'.format(X_test.shape))
    print('The size of test samples\' label: {}'.format(y_test.shape))

    #%% Import Word2Vec embedding
    word2vec = KeyedVectors.load_word2vec_format(w2v_file)

    num_words, embedding_dim = word2vec.vectors.shape
    print(num_words)
    print(embedding_dim)

    embedding_matrix = np.zeros((max_words, embedding_dim))

    for word, idx in src_tokenizer.word_index.items():
        embedding_vector = word2vec[word] if word in word2vec else None
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector

    #%% Model construction
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length, mask_zero=True, weights=[embedding_matrix], trainable=False))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(256, activation='relu')))
    crf = CRF(tag_size)
    model.add(crf)

    #%% Save model summary
    with open(os.path.join(save_dir, modelname+'_summary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()
    model.summary()

    #%% Compile & Run model
    checkpoint = ModelCheckpoint(os.path.join(weight_dir, fullname+'.hdf5'), monitor='val_crf_viterbi_accuracy',
                                 verbose=1, save_best_only=True,
                                 save_weights_only=True, mode='max', period=1)

    #%% Compile
    model.compile(loss=crf.loss_function, optimizer=optimizer, metrics=[crf.accuracy])

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    #%% Train
    history = model.fit(X_train, y_train, batch_size=batchnum, epochs=epochnum,
                        validation_data=(X_test, y_test), verbose=2, callbacks=[checkpoint])
    # history = model.fit(X_train, y_train, batch_size=batchnum, epochs=epochnum,
    #                     validation_data=(X_test, y_test), verbose=2)

    #%% Get accuracy
    scores = model.evaluate(X_test, y_test)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Save history plots
    plot(history.history, epochnum, batchnum, save_dir, fullname)
    model.load_weights(os.path.join(weight_dir, fullname+'.hdf5'))

    #%% Check all results
    y_predicted = model.predict(X_test)
    pred_tags = sequence_to_tag(y_predicted)
    test_tags = sequence_to_tag(y_test)
    print(classification_report(test_tags, pred_tags, digits=4))

    #%% Record results
    result = classification_report(test_tags, pred_tags, digits=4)
    csvfile = open(os.path.join(save_dir, fullname+' - Result.csv'), 'w', newline="")
    wtr = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    result = result.split('\n')
    for idx, line in enumerate(result):
        low = line.split()
        if idx == 0:
            low.insert(0, 'tags')
        elif idx > len(result)-4:
            low[0:2] = [''.join(low[0:2])]
        wtr.writerow(low)
    csvfile.close()


    #%% Save the incorrect results
    csvfile = open(os.path.join(save_dir, fullname+' - Incorrect.csv'), 'w', newline="", encoding='utf-8')
    wtr = csv.writer(csvfile, delimiter=',', lineterminator='\n')

    for x, test, pred in zip(X_test, test_tags, pred_tags):
        if not np.array_equal(test, pred):
            i, j, k = [], [], []
            for w, t, p in zip(x, test, pred):
                if w != 0:
                    i.append(index_to_word[w])
                    j.append(t)
                    k.append(p)
            wtr.writerow(i)
            wtr.writerow(j)
            wtr.writerow(k)
            wtr.writerow('\n')
    csvfile.close()

    fold_no = fold_no + 1


# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
