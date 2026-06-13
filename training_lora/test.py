from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="dataset/finetune_dataset.json"
)["train"]

print(dataset[0])
print(type(dataset[0]))