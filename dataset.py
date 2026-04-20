import sqlite3
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


def clean_latex(t):
    if t is None:
        return ""
    return str(t).strip()


class CROHMEDataset(Dataset):
    def __init__(self, db_path, split):
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()

        self.data = self.cur.execute(
            "SELECT image, latex FROM data_simple WHERE split=?",
            (split,)
        ).fetchall()

        self.transform = T.Compose([
            T.Resize((128, 512)),  # keep wide layout for math
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, latex = self.data[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        latex = clean_latex(latex)

        return image, latex