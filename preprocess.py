import os
import sqlite3
import xml.etree.ElementTree as ET
from inkml2img import inkml2img


DATASET_ROOT = "crohme2019"
DB_PATH = "crohme.db"


def extract_latex(path):
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        print(f"[SKIP XML ERROR] {path}")
        return None

    root = tree.getroot()
    ns = {'inkml': 'http://www.w3.org/2003/InkML'}

    for ann in root.findall('.//inkml:annotation', ns):
        if ann.attrib.get('type') == 'truth':
            latex = ann.text.strip()
            return latex.strip('$')

    return None


conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS data (
    image TEXT,
    latex TEXT,
    split TEXT
)
""")

total_count = 0

for split in ["test", "valid", "train"]:
    folder = os.path.join(DATASET_ROOT, split)

    split_count = 0

    for root, _, files in os.walk(folder):
        for file in files:
            inkml_path = os.path.join(root, file)

            latex = extract_latex(inkml_path)
            if not latex:
                continue
            
            output_dir = os.path.join("images", split)
            os.makedirs(output_dir, exist_ok=True)

            img_name = file.replace(".inkml", ".png")
            img_path = os.path.join(output_dir, img_name)

            total_count += 1
            split_count += 1
            print(f"{split}: {split_count} | total: {total_count} | converting: {inkml_path}")

            inkml2img(inkml_path, img_path)

            cur.execute(
                "INSERT INTO data VALUES (?, ?, ?)",
                (img_path, latex, split)
            )

conn.commit()
conn.close()