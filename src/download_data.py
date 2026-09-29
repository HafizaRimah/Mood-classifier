from datasets import load_dataset

dataset = load_dataset("dair-ai/emotion")

labels = dataset["train"].features["label"].names
print("Emotion disribution:\n")

counts = dataset["train"].to_pandas()["label"].value_counts().sort_index()

for label_id, count in counts.items():
    print(f"{label_id} - {labels[label_id]} : {count}")