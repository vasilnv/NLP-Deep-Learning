import torch

corpusFileName = 'corpusPoems'
modelFileName = 'modelLSTM'
trainDataFileName = 'trainData'
testDataFileName = 'testData'
char2idFileName = 'char2id'

device = torch.device("cuda:0")
#device = torch.device("cpu")

batchSize = 32
char_emb_size = 64

hid_size = 256
lstm_layers = 2
dropout = 0.25

epochs = 10
learning_rate = 0.005

defaultTemperature = 0.2
