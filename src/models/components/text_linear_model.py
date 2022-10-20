from torch import nn


class TextLinearModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, num_linear=1):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.fcs = []
        for _ in range(num_linear - 1):
            self.fcs.append(nn.Linear(embed_dim, embed_dim))
            self.fcs.append(nn.LeakyReLU())
        self.fcs.append(nn.Linear(embed_dim, num_class))
        self.fcs = nn.Sequential(*self.fcs)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fcs(embedded)
