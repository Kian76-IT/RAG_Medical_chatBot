import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import numpy as np

from setting import DATA_PATH
from primary.load import load_data
from models.albert import AlbertModel

print("Loading dataset...")
df, texts = load_data(DATA_PATH)

print("Loading embedding model...")
model = AlbertModel()

print("Creating embeddings...")
embeddings = model.model.encode(
    texts,
    show_progress_bar=True
)

print("Saving embeddings...")
np.save(
    "embedding/embeddings.npy",
    embeddings
)

print("Done!")
print("Shape:", embeddings.shape)