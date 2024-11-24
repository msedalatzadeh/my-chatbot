import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np

# Define dummy dataset
class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_sentence, tgt_sentence = self.data[idx]
        src_indices = [self.src_vocab[word] for word in src_sentence.split()]
        tgt_indices = [self.tgt_vocab[word] for word in tgt_sentence.split()]
        return torch.tensor(src_indices), torch.tensor(tgt_indices)

# Define positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_size)
        self.encoding.requires_grad = False
        
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        _2i = torch.arange(0, embed_size, step=2).float()
        
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / embed_size)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / embed_size)))
        
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:seq_len, :].to(x.device)

# Define Transformer-based translation model
class TranslationModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, num_heads, ff_hidden_dim, num_layers, dropout=0.1):
        super(TranslationModel, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size)
        
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ff_hidden_dim,
            dropout=dropout
        )
        
        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask):
        # Embed and add positional encoding
        src_emb = self.dropout(self.positional_encoding(self.src_embedding(src)))
        tgt_emb = self.dropout(self.positional_encoding(self.tgt_embedding(tgt)))
        
        # Pass through the transformer
        transformer_out = self.transformer(
            src_emb.transpose(0, 1), 
            tgt_emb.transpose(0, 1),
            src_mask, tgt_mask,
            src_padding_mask, tgt_padding_mask
        )
        
        # Pass through the output layer
        out = self.fc_out(transformer_out)
        return out

# Utility function to generate masks
def create_mask(src, tgt, src_pad_idx, tgt_pad_idx):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]
    
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).type(torch.bool)
    
    src_padding_mask = (src == src_pad_idx)
    tgt_padding_mask = (tgt == tgt_pad_idx)
    
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# Example data
src_vocab = {"<pad>": 0, "hello": 1, "world": 2}
tgt_vocab = {"<pad>": 0, "bonjour": 1, "monde": 2}
data = [("hello world", "bonjour monde")]

# Create dataset and dataloader
dataset = TranslationDataset(data, src_vocab, tgt_vocab)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Hyperparameters
SRC_VOCAB_SIZE = len(src_vocab)
TGT_VOCAB_SIZE = len(tgt_vocab)
EMBED_SIZE = 128
NUM_HEADS = 4
FF_HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.1
PAD_IDX = src_vocab["<pad>"]

# Initialize model, loss, and optimizer
model = TranslationModel(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, FF_HIDDEN_DIM, NUM_LAYERS, DROPOUT)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    for src, tgt in dataloader:
        src = src.to(torch.long)
        tgt = tgt.to(torch.long)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, PAD_IDX, PAD_IDX)
        
        preds = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
        preds = preds.transpose(0, 1)  # (seq_len, batch_size, vocab_size) -> (batch_size, seq_len, vocab_size)
        
        loss = loss_fn(preds.reshape(-1, preds.shape[-1]), tgt_output.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")