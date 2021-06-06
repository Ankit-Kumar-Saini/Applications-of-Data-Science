import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_rate = 0.25):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # define Embedding layer
        self.embed_layer = nn.Embedding(self.vocab_size, self.embed_size)
        # define LSTM layer
        self.lstm_layer = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers,
                                  batch_first = True)
        # define Dropout layer to avoid overfitting
        self.dropout = nn.Dropout(drop_rate)
        # define the output layer (fully connected)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

        
    
    def forward(self, features, captions):
        """
        Forward pass of the network
        """   
        # get caption embeddings
        cap_embeddings = self.embed_layer(captions[:, :-1])
        # concatenate the features and caption embeddings
        concat = torch.cat((features.unsqueeze(1), cap_embeddings), 1)
        # pass the concat input through the LSTM layer
        all_hidden, last_hidden = self.lstm_layer(concat)
        # pass through the dropout layer
        all_hidden = self.dropout(all_hidden)
        # get the final output
        output = self.fc(all_hidden)
        return output
    
  

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        preds = []
        for idx in range(max_len):
            out, states = self.lstm_layer(inputs, states)
            out = self.fc(out.squeeze(1))
            _, pred = out.max(1)
            preds.append(pred.item())
            inputs = self.embed_layer(pred)
            inputs = inputs.unsqueeze(1)
            
        return preds
            