import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights="IMAGENET1K_V1")
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        x = self.cnn(x)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)
        return x


class Attention(nn.Module):
    def __init__(self, enc_dim=512, dec_dim=256):
        super().__init__()
        self.attn = nn.Linear(enc_dim + dec_dim, dec_dim)
        self.v = nn.Linear(dec_dim, 1)

    def forward(self, hidden, enc_out):
        B, N, _ = enc_out.shape

        hidden = hidden.unsqueeze(1).repeat(1, N, 1)

        energy = torch.tanh(self.attn(torch.cat([hidden, enc_out], dim=2)))
        scores = self.v(energy).squeeze(2)

        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)

        return context, weights


class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 128)
        self.attn = Attention()

        self.rnn = nn.LSTM(128 + 512, 256, batch_first=True)
        self.fc = nn.Linear(256, vocab_size)

    def forward(self, tgt, enc_out):
        B, T = tgt.shape

        hidden = torch.zeros(1, B, 256).to(tgt.device)
        cell = torch.zeros(1, B, 256).to(tgt.device)

        outputs = []

        for t in range(T):
            emb = self.embedding(tgt[:, t])

            ctx, _ = self.attn(hidden.squeeze(0), enc_out)

            rnn_input = torch.cat([emb, ctx], dim=1).unsqueeze(1)

            out, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

            logits = self.fc(out.squeeze(1))
            outputs.append(logits)

        return torch.stack(outputs, dim=1)


class Im2LaTeX(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(vocab_size)

    def forward(self, images, tgt):
        enc = self.encoder(images)
        out = self.decoder(tgt, enc)
        return out