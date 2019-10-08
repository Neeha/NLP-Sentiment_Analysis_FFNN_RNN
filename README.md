# NLP-Sentiment_Analysis_FFNN_RNN

The project involved implementation of neural networks for the task of Sentiment Analysis on the Rotten Tomatoes data set which consists of sentences classified as either positive or negative sentiment. A simple feed forward neural network using deep averaging and a recurrent neural network was implemented on PyTorch.

Run the project:

python3 sentiment.py --model FF

Trains the feed forward neural network model

FFNN(
  (input_layer): Linear(in_features=300, out_features=50, bias=True)
  (hidden_layer): Linear(in_features=50, out_features=50, bias=True)
  (fc): Linear(in_features=50, out_features=1, bias=True)
  (sig): Sigmoid()
  (dropout): Dropout(p=0.3, inplace=False)
)
Dev Accuracy - 75.5


python3 sentiment.py --model FANCY
Trains the recurrent neural netword model
RNN(
  (lstm_layer): LSTM(300, 80, batch_first=True, dropout=0.5)
  (hidden_layer): Linear(in_features=80, out_features=80, bias=True)
  (fc): Linear(in_features=80, out_features=1, bias=True)
  (sig): Sigmoid()
)
Dev Accuracy - 76.9
