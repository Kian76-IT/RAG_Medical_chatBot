import pandas as pd
import re

def ultra_clean(text):
    if not isinstance(text, str):
        return text
    
    # 1. Memperbaiki encoding yang rusak secara umum
    try:
        text = text.encode('latin-1').decode('utf-8')
    except:
        pass

    # 2. Kamus perbaikan untuk karakter yang sering muncul di dataset medis Kaggle
    replacements = {
        'â€¢': '• ',
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â€“': '-',
        'Â': '',
        'â€˜': "'",
    }
    
    for bad, good in replacements.items():
        text = text.replace(bad, good)
        
    # 3. Menghapus karakter non-ascii yang tersisa (opsional tapi disarankan untuk NLP)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    return text.strip()

# Load data (sesuaikan nama file dengan yang ada di folder kamu)
df = pd.read_csv("dataset/chatbot_dataset.csv")

# Terapkan pada semua kolom teks (Column1 dan Column2 di gambar kamu)
# Jika namanya sudah berubah kembali jadi question/answer, sesuaikan kodenya
for col in df.columns:
    df[col] = df[col].apply(ultra_clean)

# Simpan ke CSV baru dengan encoding yang benar-benar bersih
df.to_csv("dataset/chatbot_dataset_clean.csv", index=False, encoding='utf-8')

print("Pembersihan total selesai! Silakan cek file chatbot_dataset_clean.csv")