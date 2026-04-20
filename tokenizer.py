import string

class Tokenizer:
    def __init__(self):
        chars = (
            list(string.ascii_letters) +
            list(string.digits) +
            list("+-=*/(){}[]<>_^\\ ")
        )

        self.pad = "<PAD>"
        self.sos = "<SOS>"
        self.eos = "<EOS>"

        self.vocab = [self.pad, self.sos, self.eos] + chars

        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.itos = {i: c for i, c in enumerate(self.vocab)}

        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2

    def encode(self, text):
        ids = [self.sos_id]
        for c in text:
            ids.append(self.stoi.get(c, self.pad_id))
        ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        result = []
        for i in ids:
            if i == self.eos_id:
                break
            if i > 2:
                result.append(self.itos[i])
        return "".join(result)