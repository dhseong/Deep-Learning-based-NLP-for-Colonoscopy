#%% Import modules & Define functions
import re
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch, transformers
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adamax, Nadam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, TimeDistributed, GRU
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from transformers import BertTokenizer, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

device = torch.device('cuda:0')
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

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
    acc = hist['accuracy']
    val_acc = hist['val_accuracy']

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

losses = {'cce'      : 'categorical_crossentropy',
          'KL'       : 'kullback_leibler_divergence',
          'poisson'  : 'poisson'}
losses_list = list(losses.keys()) # -> index : l, usage -> losses[losses_list[l]]
optimizers = {'adam'    : Adam(lr=1e-3),
              'rmsprop' : RMSprop(lr=1e-3),
              'adamax'  : Adamax(lr=1e-3),
              'nadam'   : Nadam(lr=1e-3)}
optimizers_list = list(optimizers.keys())


#%% Settings
epochnum = 10
batchnum = 128
lossname = 'cce' # 'cce','KL','poisson','cos_prox'
optsname = 'adam'
modelname = 'BioBERT-CRF'
dataname = 'D1'
epochname = '10'
loss      = losses[lossname]
optimizer = optimizers[optsname]
max_grad_norm = 1.0
fullname  = '_'.join([modelname, lossname, optsname])

#%% Load Data & Get Sentence+Tag
root_dir   = os.getcwd()
data_dir   = os.path.join(root_dir, 'data')
save_dir   = os.path.join(root_dir, 'results', modelname)
weight_dir = os.path.join(root_dir, 'weights', modelname)
annotation_dir = data_dir + '\\data.txt'

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
X = pad_sequences(X, dtype='int', value=0.0, padding='post', maxlen=max_length)
y = pad_sequences(y, dtype='longlong', value=0, padding='post', maxlen=max_length)

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

    fullname = modelname+'_'+lossname+'_'+optsname+'_'+dataname+'_'+str(fold_no)  # data compare

    #%% One-hot encoding for y
    y_train = to_categorical(y_train, num_classes=tag_size)
    y_test = to_categorical(y_test, num_classes=tag_size)

    #%% Check padded sequence and data shape
	print('The size of training samples: {}'.format(X_train.shape))
	print('The size of training samples\' label: {}'.format(y_train.shape))
	print('The size of test samples: {}'.format(X_test.shape))
	print('The size of test samples\' label: {}'.format(y_test.shape))

	#%% Word2Vec
	import gensim
	word2vec = gensim.models.KeyedVectors.load_word2vec_format('../w2v.bin')

	num_words, embedding_dim = word2vec.vectors.shape
	print(num_words)
	print(embedding_dim)

	embedding_matrix = np.zeros((max_words, embedding_dim))

	np.shape(embedding_matrix)

	def get_vector(word):
	    if word in word2vec:
	        return word2vec[word]
	    else:
	        return None

	for word, i in src_tokenizer.word_index.items():
	    temp = get_vector(word)
	    if temp is not None:
	        embedding_matrix[i] = temp

	#%% Create the attention mask to ignore the padded elements in the sequences during training, development and testing
	attention_masks_train = [[bool(i != 0.0) for i in ii] for ii in x_train]
	attention_masks_test = [[bool(i != 0.0) for i in ii] for ii in x_test]

	#%% Set tensors
	train_inputs = torch.tensor(X_train)
	test_inputs = torch.tensor(X_test)
	train_tags = torch.tensor(y_train)
	test_tags = torch.tensor(y_test)
	train_masks = torch.tensor(attention_masks_train)
	test_masks = torch.tensor(attention_masks_test)

	#%% We define the dataloaders.
	train_data = TensorDataset(train_inputs, train_masks, train_tags)
	train_sampler = RandomSampler(train_data)
	train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batchnum)

	# Load dev and test data sequentially with SequentialSampler.
	dev_data = TensorDataset(test_inputs, test_masks, test_tags)
	dev_sampler = SequentialSampler(dev_data)
	dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batchnum)

	test_data = TensorDataset(test_inputs, test_masks, test_tags)
	test_sampler = SequentialSampler(test_data)
	test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batchnum)

	#%% The BertForTokenClassification class is used for token-level predictions.
	from transformers import AdamW, BertModel, BertConfig, BertForTokenClassification
	from torch import nn, optim
	from TorchCRF import CRF

	config = {"hidden_size": 768,
	  "maxlen" : max_length,
	  "epochs": epochnum,
	  "batch_size": batchnum,
	  "dropout": 0.1,
	  "learning_rate": 5e-5,
	  "warmup_proportion": 0.1,
	  "gradient_accumulation_steps": 1,
	  "summary_step": 250,
	  "adam_epsilon": 1e-8,
	  "warmup_steps": 0,
	  "max_grad_norm": 1,
	  "logging_steps": 50,
	  "evaluate_during_training": True,
	  "save_steps": 250,
	  "output_dir": weight_dir}

	bert_config = {'attention_probs_dropout_prob': 0.1,
	                 'hidden_act': 'gelu',
	                 'hidden_dropout_prob': 0.1,
	                 'hidden_size': 768,
	                 'initializer_range': 0.02,
	                 'intermediate_size': 3072,
	                 'max_position_embeddings': 512,
	                 'num_attention_heads': 12,
	                 'num_hidden_layers': 12,
	                 'type_vocab_size': 2,
	                 'vocab_size': vocab_size}

	# Set true before loading models
	class BERT_CRF(nn.Module):
	    def __init__(self, config, num_classes) -> None:
	        super(BERT_CRF, self).__init__()
	        self.bert = BertModel(config=BertConfig.from_dict(bert_config)).from_pretrained("dmis-lab/biobert-base-cased-v1.1", num_labels=len(index_to_ner),
	    output_attentions = False, output_hidden_states = False)
	        self.dropout = nn.Dropout(config['dropout'])
	        self.position_wise_ff = nn.Linear(config['hidden_size'], num_classes)
	        self.crf = CRF(num_classes, batch_first=True)

	    def forward(self, input_ids, token_type_ids=None, attention_mask = None, labels=None, evaluation=False):
	        # outputs: (last_encoder_layer, pooled_output, attention_weight)
	        outputs = self.bert(input_ids=input_ids,
	                            token_type_ids=token_type_ids,
	                            attention_mask=attention_mask)
	        last_encoder_layer = outputs[0]
	        last_encoder_layer = self.dropout(last_encoder_layer)
	        emissions = self.position_wise_ff(last_encoder_layer)

	        if evaluation is False:
	            log_likelihood = self.crf(emissions, labels, attention_mask)
	            sequence_of_tags = self.crf._viterbi_decode(emissions, attention_mask)
	            return log_likelihood, sequence_of_tags
	        else:
	            log_likelihood = self.crf(emissions, labels, attention_mask)
	            sequence_of_tags = self.crf.decode(emissions)
	            return log_likelihood, sequence_of_tags

	model = BERT_CRF(config = config, num_classes=tag_size)
	model.cuda()

	#%% Optimizer settings
	FULL_FINETUNING = True
	if FULL_FINETUNING:
	    param_optimizer = list(model.named_parameters())
	    no_decay = ['bias', 'gamma', 'beta']
	    optimizer_grouped_parameters = [
	        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
	         'weight_decay_rate': 0.01},
	        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
	         'weight_decay_rate': 0.0}
	    ]
	else:
	    param_optimizer = list(model.classifier.named_parameters())
	    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

	# Adam optimizer
	optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8)

	#%% Scheduler
	from transformers import get_linear_schedule_with_warmup

	# Total number of training steps is number of batches * number of epochs.
	total_steps = len(train_dataloader) * epochnum

	# Create the learning rate scheduler.
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

	# Import seqeval
	import seqeval
	from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report

	#%% Train and evaluate model
	loss_values, development_loss_values = [], []

	for _ in range(epochnum):
	    # Training
	    # Set the model into training mode
	    model.train()
	    # Reset the total loss for each epoch
	    total_loss = 0

	    for step, batch in enumerate(train_dataloader):
	        # Transfer batch to gpu
	        batch = tuple(t.to(device) for t in batch)
	        b_input_ids, b_input_mask, b_labels = batch
	        # Remove previous gradients before each backward pass
	        model.zero_grad()
	        # This returns the loss (not the model output) since we have input the labels.
	        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
	        # Get the loss
	        loss = -1*outputs[0]
	        # Backward pass to compute the gradients
	        loss.backward()
	        # Train loss
	        total_loss += loss.item()
	        # Clip the norm of the gradient
	        # This is to help prevent the "exploding gradients" problem.
	        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
	        # Update parameters
	        optimizer.step()
	        # Update the learning rate.
	        scheduler.step()
	    # Calculate the average loss over the training data
	    avg_train_loss = total_loss / len(train_dataloader)
	    print("Average train loss: {}".format(avg_train_loss))

	    # Store each loss value for plotting the learning curve afterwards
	    loss_values.append(avg_train_loss)
	    # After each training epoch, measure performance on development set
	    # Set the model into evaluation mode
	    model.eval()
	    # Reset the development loss for this epoch
	    eval_loss, eval_accuracy = 0, 0
	    nb_eval_steps, nb_eval_examples = 0, 0
	    predictions, true_labels = [], []
	    for batch in dev_dataloader:
	        batch = tuple(t.to(device) for t in batch)
	        b_input_ids, b_input_mask, b_labels = batch

	        # The model must not compute or save gradients, in order to save memory and speed up this step
	        with torch.no_grad():
	            # Forward pass, compute predictions
	            # This will return the logits (logarithm of the odds), not the loss (we do not provide labels)
	            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels, evaluation=True)
	        # Transfer logits and labels to CPU
	        logits = np.array(outputs[1])
	        label_ids = b_labels.to('cpu').numpy()

	        # Compute the accuracy for this batch of development sentences
	        eval_loss -= 1*outputs[0].mean().item()
	        predictions.extend(logits)
	        true_labels.extend(label_ids)

	    eval_loss = eval_loss / len(dev_dataloader)
	    development_loss_values.append(eval_loss)
	    print("Development loss: {}".format(eval_loss))
	    pred_tags = [index_to_ner[p_i] for p, l in zip(predictions, true_labels)
	                 for p_i, l_i in zip(p, l) if index_to_ner[l_i] != "PAD"]
	    dev_tags = [index_to_ner[l_i] for l in true_labels
	                for l_i in l if index_to_ner[l_i] != "PAD"]
	    print("Development Accuracy: {}".format(accuracy_score(pred_tags, dev_tags)))
	    print("Development classification report:\n{}".format(classification_report(pred_tags, dev_tags, digits=4)))
	    print()

	torch.save(model, os.path.join(weight_dir,fullname+'.hdf5'))
	# model2 = torch.load(os.path.join(weight_dir,fullname+'.hdf5'))
	# model2.eval()

	#%% Plot the training loss
	import matplotlib.pyplot as plt
	import seaborn as sns
	# Use plot styling from seaborn.
	sns.set(style='darkgrid')
	# Increase the plot size and font size.
	sns.set(font_scale=3)
	plt.figure(figsize=(30, 12))
	# Plot the learning curve.
	plt.plot(loss_values, 'b-o', label="training loss")
	plt.plot(development_loss_values, 'r-o', label="validation loss")
	# Label the plot.
	plt.title("Learning curve")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()
	plt.savefig(os.path.join(save_dir,fullname+'_training_loss.png'))
	plt.clf()

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
