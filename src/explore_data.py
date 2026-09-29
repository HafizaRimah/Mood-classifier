from datasets import load_dataset

dataset = load_dataset("dair-ai/emotion")

print("Dataset splits:\n")

for split in dataset:
    print(f"{split}: {len(dataset[split])} examples")