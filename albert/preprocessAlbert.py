import pandas as pd
import csv
import re

# 1. Inisialisasi df sebagai None agar tidak NameError
df = None

try:
    # Membaca file
    df = pd.read_csv(
        "dataset/chatbot_dataset_clean.csv", 
        sep=",", 
        quotechar='"', 
        on_bad_lines='warn', 
        engine='python'
    )
    print("Dataset berhasil dimuat!")

    # 2. PROSES PEMBERSIHAN (Harus di dalam sini atau dicek if df is not None)
    if df is not None:
        # Bersihkan nama kolom dari titik koma
        df.columns = [col.replace(';', '').strip() for col in df.columns]
        
        # Bersihkan isi sel dari titik koma di akhir (termasuk jika ada spasi)
        # Kami gunakan regex untuk menghapus satu atau lebih ';' di akhir string
        df = df.applymap(lambda x: re.sub(r';+\s*$', '', str(x)) if isinstance(x, str) else x)
        
        print("Pembersihan ;;; selesai.")
        print(df.head())

except Exception as e:
    print(f"Gagal memuat dataset: {e}")

# 3. Simpan hanya jika df berhasil dibuat
if df is not None:
    df.to_csv("dataset/chatbot_dataset_ready.csv", index=False, quoting=csv.QUOTE_ALL)
    print("File final disimpan di dataset/chatbot_dataset_ready.csv")
