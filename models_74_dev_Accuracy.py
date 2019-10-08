# models.py

from sentiment_data import *
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np
import random

class FFNN(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, word_vectors):
        super(FFNN, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.input_layer = nn.Linear(embedding_size,hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sig = nn.Sigmoid()

        self.embedding = word_vectors
        self.dropout = nn.Dropout(0.3)
        
        nn.init.xavier_uniform(self.input_layer.weight)
        nn.init.xavier_uniform(self.hidden_layer.weight)

    # Forward computation. Backward computation is done implicitly (nn.Module already has an implementation of
    # it that you shouldn't need to override)
    def forward(self, x):
        batch_size = x.size(0)
        embeddings = torch.ones([batch_size,300], dtype=torch.float64)
        for sent_idx in range(0,batch_size): 
            embed_sum = 0
            for word_idx in range(60): #60
                embed_sum += self.embedding.get_embedding_from_index(int(x[sent_idx][word_idx]))
            embeddings[sent_idx] = torch.from_numpy(embed_sum/60)
        out = self.input_layer(embeddings.float())
        # out = self.hidden_layer(out)    
        # dropout and fully connected layer
        out = self.dropout(out)
        out = self.fc(out)
        
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out

# Form the input to the neural network. In general this may be a complex function that synthesizes multiple pieces
# of data, does some computation, handles batching, etc.
def form_input(x):
    # print(torch.from_numpy(x).float())
    return torch.from_numpy(x).float()

def pad_to_length(np_arr, length):
    """
    Forces np_arr to length by either truncation (if longer) or zero-padding (if shorter)
    :param np_arr:
    :param length: Length to pad to
    :return: a new numpy array with the data from np_arr padded to be of length length. If length is less than the
    length of the base array, truncates instead.
    """
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result


def print_evaluation(gold_labels, predicted_labels):
    """
    Evaluates the predicted labels with respect to the gold labels
    :param gold_labels:
    :param predicted_labels:
    :return:
    """
    correct = 0
    num_pred = 0
    num_gold = 0
    for gold, guess in zip(gold_sentences, guess_sentences):
        correct += len(set(guess.chunks) & set(gold.chunks))
        num_pred += len(guess.chunks)
        num_gold += len(gold.chunks)
    if num_pred == 0:
        prec = 0
    else:
        prec = correct/float(num_pred)
    if num_gold == 0:
        rec = 0
    else:
        rec = correct/float(num_gold)
    if prec == 0 and rec == 0:
        f1 = 0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    print("Labeled F1: " + "{0:.2f}".format(f1 * 100) +\
          ", precision: %i/%i" % (correct, num_pred) + " = " + "{0:.2f}".format(prec * 100) + \
          ", recall: %i/%i" % (correct, num_gold) + " = " + "{0:.2f}".format(rec * 100))

# , using dev_exs for development and returning
# predictions on the *blind* test_exs (all test_exs have label 0 as a dummy placeholder value).
def train_evaluate_ffnn(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], test_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> List[SentimentExample]:
    """
    Train a feedforward neural network on the given training examples, using dev_exs for development, and returns
    predictions on the *blind* test examples passed in. Returned predictions should be SentimentExample objects with
    predicted labels and the same sentences as input (but these won't be read by the external code). The code is set
    up to go all the way to test predictions so you have more freedom to handle example processing as you see fit.
    :param train_exs:
    :param dev_exs:
    :param test_exs:
    :param word_vectors:
    :return:
    """
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])


    #dev data
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    dev_labels_arr = np.array([ex.label for ex in dev_exs])

    #test data
    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])
    test_labels_arr = np.array([ex.label for ex in test_exs])

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_mat), torch.from_numpy(train_labels_arr))
    valid_data = TensorDataset(torch.from_numpy(dev_mat), torch.from_numpy(dev_labels_arr))
    test_data = TensorDataset(torch.from_numpy(test_mat), torch.from_numpy(test_labels_arr))

    # dataloaders
    batch_size = 10

    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    #TRAINING
    num_epochs = 2
    batch_size = 3

    output_size = 1
    embedding_size = 300 
    hidden_size = 100
    counter = 0

    ffnn = FFNN(output_size, embedding_size, hidden_size, word_vectors)
    initial_learning_rate = 0.1
    criterion = nn.BCELoss()
    # criterion = nn.NLLLoss()
    optimizer = optim.Adam(ffnn.parameters(),lr=0.1)
    ffnn.train()
    
    for epoch in range(num_epochs):
        # h = ffnn.init_hidden(batch_size)
        for inputs, labels in train_loader:
            counter += 1
            total_loss = 0.0
            # h = tuple([each.data for each in h])
            ffnn.zero_grad()
            probs= ffnn.forward(inputs)

            # loss = torch.neg(torch.log(probs)).dot(y_onehot)
            loss = criterion(probs.squeeze(), labels.float())
            total_loss += loss
            loss.backward()
            optimizer.step()
        
            # loss stats
            if counter % 100 == 0:
                # Get validation loss
                # val_h = ffnn.init_hidden(batch_size)
                val_losses = []
                ffnn.eval()
                num_correct = 0
                for inputs, labels in valid_loader:

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    # val_h = tuple([each.data for each in val_h])

                    output= ffnn.forward(inputs)
                    val_loss = criterion(output.squeeze(), labels.float())
                    val_losses.append(val_loss.item())
                
                    pred = torch.round(output.squeeze())
                    correct_tensor = pred.eq(labels.float().view_as(pred))
                    correct = np.squeeze(correct_tensor.numpy())
                    num_correct += np.sum(correct)
                train_acc = num_correct/len(valid_loader.dataset)
                print("Epoch: {}/{}...".format(epoch+1, num_epochs),
                          # "Step: {}...".format(counter),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(np.mean(val_losses)))
                print("Train accuracy: {:.3f}".format(train_acc))

    #TEST ON DEVELOPMENT SET
    # val_h = ffnn.init_hidden(batch_size)
    val_losses = []
    predictions = []
    ffnn.eval()
    num_correct = 0
    for inputs, labels in valid_loader:
        # val_h = tuple([each.data for each in val_h])
        output = ffnn.forward(inputs)
        # val_loss = criterion(output.squeeze(), labels.float())
        # val_losses.append(val_loss.item())
        predictions.append(torch.round(output.squeeze()))
        pred = torch.round(output.squeeze())
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)
    dev_acc = num_correct/len(valid_loader.dataset)
    print("correct predictions:", num_correct," dataset size:", len(valid_loader.dataset))
    print("Dev accuracy: {:.3f}".format(dev_acc))


# Analogous to train_ffnn, but trains your fancier model.
def train_evaluate_fancy(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], test_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> List[SentimentExample]:
    raise Exception("Not implemented")