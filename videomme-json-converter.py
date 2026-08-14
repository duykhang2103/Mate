import os
import pandas as pd
import json

PARQUET = "DATAS/Video-MME/videomme/test-00000-of-00001.parquet"
OUT_DIR = "DATAS/Video-MME/json"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_parquet(PARQUET)

result = {
    "short": [],
    "medium": [],
    "long": []
}

for _, row in df.iterrows():

    candidates = []

    for option in row["options"]:
        # Remove "A. ", "B. ", ...
        candidates.append(option.split(". ", 1)[1])

    answer_idx = ord(row["answer"]) - ord("A")

    sample = {
        "video": row["videoID"] + ".mp4",
        "question": row["question"],
        "candidates": candidates,
        "answer": candidates[answer_idx]
    }

    result[row["duration"]].append(sample)

for duration in result:
    with open(
        os.path.join(OUT_DIR, f"{duration}.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(result[duration], f, ensure_ascii=False, indent=2)

print("Done.")

# Command to run this script:
# python videomme-json-converter.py