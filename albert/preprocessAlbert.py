from datasets import load_dataset
import pandas as pd

# 1. Load dataset dari Hugging Face
dataset = load_dataset("abdelhakimDZ/diabetes_QA_dataset")

# 2. Ubah ke Pandas DataFrame
df = dataset['train'].to_pandas()

# 3. Simpan ke file baru (pilih salah satu format di bawah ini)
df.to_csv("diabetes_data.csv", index=False) # Akan muncul file .csv
# df.to_json("diabetes_data.json", orient="records") # Atau format .json

print("File berhasil disimpan!")