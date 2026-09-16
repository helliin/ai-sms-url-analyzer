from pathlib import Path
from email import policy
from email.parser import BytesParser
import pandas as pd


def read_email(path):
    try:
        msg = BytesParser(policy=policy.default).parsebytes(path.read_bytes())

        parts = []

        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    content = part.get_payload(decode=True)

                    if content:
                        charset = part.get_content_charset() or "utf-8"

                        try:
                            text = content.decode(charset, errors="replace")
                        except LookupError:
                            text = content.decode("utf-8", errors="replace")

                        parts.append(text)

                except Exception:
                    continue

        return "\n".join(parts)

    except Exception:
        return ""


rows = []

folders = [
    ("data/spamassassin/easy_ham", "ham"),
    ("data/spamassassin/spam", "spam"),
]

for folder, label in folders:
    folder_path = Path(folder)

    for path in folder_path.iterdir():
        if path.is_file():
            text = read_email(path)

            rows.append({
                "text": text,
                "label": label
            })


df = pd.DataFrame(rows)

print("=" * 60)
print("SPAMASSASSIN VERİ AKTARIMI")
print("=" * 60)

print(f"\nToplam kayıt: {len(df)}")

print("\nEtiket dağılımı:")
print(df["label"].value_counts())

print("\nBoş text:")
print(df["text"].fillna("").str.strip().eq("").sum())

df.to_csv("data/spamassassin_raw.csv", index=False)

print("\nCSV kaydedildi:")
print("data/spamassassin_raw.csv")

print("=" * 60)
print("İŞLEM TAMAMLANDI")
print("=" * 60)