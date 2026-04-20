import torch
from model import Im2LaTeX
from tokenizer import Tokenizer
from PIL import Image
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = Tokenizer()

model = Im2LaTeX(len(tokenizer.vocab)).to(device)
model.load_state_dict(torch.load("im2latex.pth", map_location=device))
model.eval()

transform = T.Compose([
    T.Resize((128, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


def greedy_decode(image_path, max_len=100):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        enc = model.encoder(image)  # (B, T, C)
        B, T, C = enc.shape

        # INITIAL LSTM STATE (CORRECT)
        h = torch.zeros(1, B, 256).to(device)
        c = torch.zeros(1, B, 256).to(device)

        inp = torch.tensor([tokenizer.sos_id], device=device)

        result = []

        for _ in range(max_len):

            emb = model.decoder.embedding(inp)

            # FIX HERE
            context, _ = model.decoder.attn(h[-1], enc)

            rnn_input = torch.cat([emb, context], dim=1).unsqueeze(1)

            out, (h, c) = model.decoder.rnn(rnn_input, (h, c))

            logits = model.decoder.fc(out.squeeze(1))

            inp = logits.argmax(1)

            if inp.item() == tokenizer.eos_id:
                break

            result.append(inp.item())

    return tokenizer.decode(result)