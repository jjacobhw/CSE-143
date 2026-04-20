import csv
import re
from pathlib import Path


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)  # strip HTML tags
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def load_and_clean(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"text": clean_text(row["text"]), "label": int(row["label"])})
    return rows


def save_clean(data: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    for split in ("train", "dev", "test"):
        data = load_and_clean(f"raw data/{split}.csv")
        save_clean(data, f"data/cleaned/{split}.csv")
        print(f"{split}: {len(data)} examples -> data/cleaned/{split}.csv")
