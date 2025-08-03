import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiKernelBiLSTM_Embedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, conv_kernel_size, seq_len=130):
        super().__init__()

        if conv_kernel_size == 'Full':
            conv_kernel_size = seq_len

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.bilstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(hidden_dim * 2, hidden_dim, conv_kernel_size, padding=conv_kernel_size // 2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.bilstm(x)
        x = x.transpose(1, 2)
        x = F.relu(self.conv(x)).transpose(1, 2)
        return torch.sigmoid(self.fc(x)).squeeze(-1)

class MultiKernelBiLSTM_OneHot(nn.Module):
    def __init__(self, input_dim, hidden_dim, conv_kernel_size, seq_len=130):
        super().__init__()

        if conv_kernel_size == 'Full':
            conv_kernel_size = seq_len

        self.bilstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(hidden_dim * 2, hidden_dim, conv_kernel_size, padding=conv_kernel_size // 2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x, _ = self.bilstm(x)
        x = x.transpose(1, 2)
        x = F.relu(self.conv(x)).transpose(1, 2)
        return torch.sigmoid(self.fc(x)).squeeze(-1)
