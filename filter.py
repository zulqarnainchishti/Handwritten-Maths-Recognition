import sqlite3
import re

DB_PATH = "crohme.db"

# -----------------------
# ALLOWED TOKENS
# -----------------------
ALLOWED_COMMANDS = [
    r"\\frac",
    r"\\sqrt",
    r"\\sin",
    r"\\cos",
    r"\\log",
    r"\\exp"
]

ALLOWED_PATTERN = re.compile(
    r"""
    ^[
        a-zA-Z0-9
        \+\-\=\*\^\_\(\)\[\]\{\}\s
        \\. 
    ]+$
    """,
    re.VERBOSE
)


# -----------------------
# VALIDATION FUNCTIONS
# -----------------------
def allowed_characters(latex):
    return bool(ALLOWED_PATTERN.match(latex))


def allowed_commands(latex):
    commands = re.findall(r"\\[a-zA-Z]+", latex)
    for cmd in commands:
        if cmd not in ALLOWED_COMMANDS:
            return False
    return True


def length_ok(latex):
    return len(latex) <= 60


def nesting_ok(latex):
    return latex.count("{") <= 4


def basic_frac_ok(latex):
    # very rough sanity check
    if "\\frac" in latex:
        return latex.count("{") >= 2
    return True


def is_valid(latex):
    if latex is None:
        return False

    latex = latex.strip()

    if latex == "":
        return False

    if not allowed_characters(latex):
        return False

    if not allowed_commands(latex):
        return False

    if not length_ok(latex):
        return False

    if not nesting_ok(latex):
        return False

    if not basic_frac_ok(latex):
        return False

    return True


# -----------------------
# MAIN
# -----------------------
def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS data_simple")

    cur.execute("""
    CREATE TABLE data_simple (
        image TEXT,
        latex TEXT,
        split TEXT
    )
    """)

    rows = cur.execute("SELECT image, latex, split FROM data").fetchall()

    kept = []
    removed = 0

    for img, latex, split in rows:
        if is_valid(latex):
            kept.append((img, latex, split))
        else:
            removed += 1

    cur.executemany(
        "INSERT INTO data_simple (image, latex, split) VALUES (?, ?, ?)",
        kept
    )

    conn.commit()

    print("\n===== WHITELIST FILTER =====")
    print(f"Kept: {len(kept)}")
    print(f"Removed: {removed}")

    for split in ["train", "valid", "test"]:
        count = cur.execute(
            "SELECT COUNT(*) FROM data_simple WHERE split=?",
            (split,)
        ).fetchone()[0]
        print(f"{split}: {count}")

    conn.close()


if __name__ == "__main__":
    main()