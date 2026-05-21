import re

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[•â€¢]', '', text)

    text = re.sub(r"[^a-z0-9\s.,!?'\-]", '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text