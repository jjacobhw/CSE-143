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
    base_dir = Path(__file__).parent
    raw_dir = base_dir / "data" / "raw data"
    cleaned_dir = base_dir / "data" / "cleaned data"

    for split in ("train", "dev", "test"):
        data = load_and_clean(raw_dir / f"{split}.csv")
        out_path = cleaned_dir / f"{split}.csv"
        save_clean(data, out_path)
        print(f"{split}: {len(data)} examples -> {out_path}")