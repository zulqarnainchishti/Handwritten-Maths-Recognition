import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CROHMEDataset
from tokenizer import Tokenizer
from model import Im2LaTeX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = Tokenizer()

train_ds = CROHMEDataset("crohme.db", "train")
val_ds = CROHMEDataset("crohme.db", "valid")

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

model = Im2LaTeX(len(tokenizer.vocab)).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train():
    model.train()

    for epoch in range(10):

        teacher_forcing_ratio = max(0.5, 1 - epoch * 0.1)

        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images, targets, teacher_forcing_ratio)

            B, T, V = outputs.shape

            loss = criterion(
                outputs.reshape(B * T, V),
                targets[:, 1:].reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch} Step {i} Loss {loss.item()}")

    torch.save(model.state_dict(), "im2latex.pth")


if __name__ == "__main__":
    train()