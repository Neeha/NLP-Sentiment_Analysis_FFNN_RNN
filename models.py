# models.py

from sentiment_data import *
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np
import random
import itertools

class FFNN(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, word_vectors):
        super(FFNN, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
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
        sequence_len = x.size(1)
        embeddings = torch.ones([batch_size,self.embedding_size], dtype=torch.float64)
        for sent_idx in range(0,batch_size): 
            embed_sum = 0
            for word_idx in range(sequence_len): #60
                embed_sum += self.embedding.get_embedding_from_index(int(x[sent_idx][word_idx]))
            embeddings[sent_idx] = torch.from_numpy(embed_sum/sequence_len)
        out = self.input_layer(embeddings.float())
        
        #Two hidden layers
        out = self.hidden_layer(out) 
        out = self.hidden_layer(out) 

        #Dropout layer
        out = self.dropout(out)

        #Output fully connected layer
        out = self.fc(out)
        # out = self.dropout(out)
        #Sigmoid layer
        sig_out = self.sig(out)
        
        # reshaping
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # getting last batch of labels
        # sig_out = sig_out.mean(dim=1)
        
        return sig_out

class RNN(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, word_vectors):
        super(RNN, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm_layer = nn.LSTM(embedding_size, hidden_size, 1, dropout=0.5, batch_first=True)#, bidirectional=True)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sig = nn.Sigmoid()

        self.embedding = word_vectors
        self.hidden = (torch.randn(1, 1, embedding_size), torch.randn(1, 1, embedding_size))
        
        nn.init.xavier_uniform_(self.hidden_layer.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        for name, param in self.lstm_layer.named_parameters():
          if 'bias' in name:
             nn.init.constant_(param, 0.0)
          elif 'weight' in name:
             nn.init.xavier_normal_(param)


    # Forward computation. Backward computation is done implicitly (nn.Module already has an implementation of
    # it that you shouldn't need to override)
    def forward(self, x):
        batch_size = x.size(0)
        sequence_len = x.size(1)
        embeddings = torch.ones([batch_size, sequence_len,self.embedding_size], dtype=torch.float64)
        for sent_idx in range(0,batch_size): 
            embed_sum = 0
            for word_idx in range(sequence_len):
                embeddings[sent_idx,word_idx] = torch.from_numpy(self.embedding.get_embedding_from_index(int(x[sent_idx][word_idx])))

        out,h = self.lstm_layer(embeddings.float())   
        # out = self.dropout(out)
        out = out.contiguous().view(-1, self.hidden_size) 
        out = self.hidden_layer(out)
        # out = self.hidden_layer(out)
        sig_out = self.sig(out)
        # reshaping
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out.mean(dim=1)
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

# , using dev_exs for development and returning
# predictions on the *blind* test_exs (all test_exs have label 0 as a dummy placeholder value).
def train_evaluate_ffnn(train_data: TensorDataset, dev_data: TensorDataset, test_data: TensorDataset, test_seq_lens, word_vectors: WordEmbeddings) -> List[SentimentExample]:
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
    # dataloaders
    batch_size = 5

    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=1)

    #TRAINING
    num_epochs = 1

    output_size = 1
    embedding_size = 300 
    hidden_size = 50
    counter = 0

    ffnn = FFNN(output_size, embedding_size, hidden_size, word_vectors)
    print(ffnn)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(ffnn.parameters(),lr=0.01)
    ffnn.train()
    
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            counter += 1
            total_loss = 0.0
            ffnn.zero_grad()
            probs= ffnn.forward(inputs)
            loss = criterion(probs.squeeze(), labels.float())
            total_loss += loss
            loss.backward()
            optimizer.step()
        
            # loss stats
            if counter % 100 == 0:
                # Get validation loss
                val_losses = []
                ffnn.eval()
                num_correct = 0
                for inputs, labels in valid_loader:
                    output= ffnn.forward(inputs)
                    val_loss = criterion(output.squeeze(), labels.float())
                    val_losses.append(val_loss.item())
                    pred = torch.round(output.squeeze())
                    correct_tensor = pred.eq(labels.float().view_as(pred))
                    correct = np.squeeze(correct_tensor.numpy())
                    num_correct += np.sum(correct)
                train_acc = num_correct/len(valid_loader.dataset)
                print("Epoch: {}/{}...".format(epoch+1, num_epochs),
                          "Step: {}...".format(counter),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(np.mean(val_losses)))
                print("Train accuracy: {:.3f}".format(train_acc))

    #ACCURACY ON DEVELOPMENT SET
    predictions = []
    ffnn.eval()
    num_correct = 0
    for inputs, labels in valid_loader:
        output = ffnn.forward(inputs)
        predictions.append(torch.round(output.squeeze()))
        pred = torch.round(output.squeeze())
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)

    dev_acc = num_correct/len(valid_loader.dataset)
    print("correct predictions:", num_correct," dataset size:", len(valid_loader.dataset))
    print("Dev accuracy: {:.3f}".format(dev_acc))


    #TEST SET
    predictions = []
    final_predictions = []
    ffnn.eval()
    num_correct = 0
    i = 0
    for inputs, labels in test_loader:
        output = ffnn.forward(inputs)
        predictions.append(torch.round(output.squeeze()))
        pred = torch.round(output.squeeze())
        inputs = inputs.numpy()
        inputs = list(itertools.chain(*inputs))
        final_predictions.append(SentimentExample(inputs[:test_seq_lens[i]],int(pred.detach().numpy())))
        i += 1
    return final_predictions


# Analogous to train_ffnn, but trains your fancier model.
def train_evaluate_fancy(train_data: TensorDataset, dev_data: TensorDataset, test_data: TensorDataset, test_seq_lens, word_vectors: WordEmbeddings) -> List[SentimentExample]:
    # dataloaders
    batch_size = 5

    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=1)
    
    
    #TRAINING
    #parameters
    num_epochs = 5
    output_size = 1
    embedding_size = 300
    hidden_size = 80
    counter = 0

    rnn = RNN(output_size, embedding_size, hidden_size, word_vectors)
    print(rnn)
    initial_learning_rate = 0.01
    criterion = nn.BCELoss()
    optimizer = optim.Adam(rnn.parameters(),lr=initial_learning_rate)
    rnn.train()
    
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            counter += 1
            total_loss = 0.0
            rnn.zero_grad()
            probs= rnn.forward(inputs)

            loss = criterion(probs.squeeze(), labels.float())
            total_loss += loss
            loss.backward()
            optimizer.step()
        
            # loss stats
            if counter % 100 == 0:
                val_losses = []
                rnn.eval()
                num_correct = 0
                for inputs, labels in valid_loader:

                    output= rnn.forward(inputs)
                    val_loss = criterion(output.squeeze(), labels.float())
                    val_losses.append(val_loss.item())
                
                    pred = torch.round(output.squeeze())
                    correct_tensor = pred.eq(labels.float().view_as(pred))
                    correct = np.squeeze(correct_tensor.numpy())
                    num_correct += np.sum(correct)
                train_acc = num_correct/len(valid_loader.dataset)
                print("Epoch: {}/{}...".format(epoch+1, num_epochs),
                          # "Step: {}...".format(counter),
                          # "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(np.mean(val_losses)))
                print("Train accuracy: {:.3f}".format(train_acc))

    torch.save(rnn, "rnnmodel.pth")
    #ACCURACY ON DEVELOPMENT SET
    val_losses = []
    predictions = []
    rnn.eval()
    num_correct = 0
    for inputs, labels in valid_loader:
        output = rnn.forward(inputs)
        predictions.append(torch.round(output.squeeze()))
        pred = torch.round(output.squeeze())
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)
    dev_acc = num_correct/len(valid_loader.dataset)
    print("correct predictions:", num_correct," dataset size:", len(valid_loader.dataset))
    print("Dev accuracy: {:.3f}".format(dev_acc))

    #TEST SET
    predictions = []
    final_predictions = []
    rnn.eval()
    num_correct = 0
    i = 0
    for inputs, labels in test_loader:
        output = rnn.forward(inputs)
        predictions.append(torch.round(output.squeeze()))
        pred = torch.round(output.squeeze())
        inputs = inputs.numpy()
        inputs = list(itertools.chain(*inputs))
        final_predictions.append(SentimentExample(inputs[:test_seq_lens[i]],int(pred.detach().numpy())))
        i += 1
    return final_predictions