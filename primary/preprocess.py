import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[•â€¢]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text