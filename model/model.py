# model/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiKernelBiLSTM_Embedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, conv_kernel_size, seq_len=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=vocab_size - 1)
        self.bilstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        if conv_kernel_size == "Full":
            conv_kernel_size = seq_len or 101
        self.conv = nn.Conv1d(hidden_dim * 2, hidden_dim, conv_kernel_size, padding=conv_kernel_size // 2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.bilstm(x)
        lstm_out = lstm_out.transpose(1, 2)
        conv_out = F.relu(self.conv(lstm_out)).transpose(1, 2)
        return torch.sigmoid(self.fc(conv_out)).squeeze(-1)
