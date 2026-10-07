"""Randomly initialized encoder-decoder Transformer built only with PyTorch."""
import math
import torch
from torch import nn
from .schema import QueryError


class ScratchTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, layers=3, d_ff=1024,
                 dropout=0.1, max_positions=4096):
        super().__init__()
        if d_model < 2 or d_model % 2 or nhead < 1 or d_model % nhead:
            raise QueryError("d_model must be positive, even, and divisible by nhead")
        if min(layers, d_ff, max_positions) < 1 or not 0 <= dropout < 1:
            raise QueryError("Invalid network dimensions or dropout")
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=layers,
            num_decoder_layers=layers, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True)
        self.output = nn.Linear(d_model, vocab_size)
        positions = torch.arange(max_positions).unsqueeze(1)
        scales = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_positions, d_model)
        pe[:, 0::2] = torch.sin(positions * scales)
        pe[:, 1::2] = torch.cos(positions * scales)
        self.register_buffer("positions", pe.unsqueeze(0), persistent=False)
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def embed(self, tokens):
        if tokens.shape[1] > self.positions.shape[1]:
            raise QueryError("Sequence exceeds configured positional limit")
        return self.dropout(self.embedding(tokens) * math.sqrt(self.d_model) + self.positions[:, :tokens.shape[1]])

    def encode(self, source):
        padding = source.eq(0)
        return self.transformer.encoder(self.embed(source), src_key_padding_mask=padding), padding

    def decode(self, target, memory, source_padding):
        size = target.shape[1]
        causal_mask = torch.triu(torch.ones(size, size, device=target.device, dtype=torch.bool), diagonal=1)
        hidden = self.transformer.decoder(
            self.embed(target), memory, tgt_mask=causal_mask,
            tgt_key_padding_mask=target.eq(0), memory_key_padding_mask=source_padding)
        return self.output(hidden)

    def forward(self, source, target):
        memory, padding = self.encode(source)
        return self.decode(target, memory, padding)

    @torch.inference_mode()
    def generate(self, source, max_new_tokens):
        """Greedy generation for one question; encoder memory is reused."""
        if source.shape[0] != 1:
            raise QueryError("Generation currently accepts one question at a time")
        self.eval()
        memory, padding = self.encode(source)
        target = torch.full((1, 1), 1, dtype=torch.long, device=source.device)
        for _ in range(max_new_tokens):
            logits = self.decode(target, memory, padding)[:, -1, :]
            logits[:, :2] = float("-inf")  # Never generate PAD or BOS.
            token = logits.argmax(dim=-1, keepdim=True)
            target = torch.cat((target, token), dim=1)
            if token.item() == 2:
                return target[0].tolist()
        raise QueryError("Model did not finish the plan before the output limit")
