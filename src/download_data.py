"""
Script para download automático do Spotify Tracks Dataset (Maharshi Pandya).
Fonte original (Kaggle): https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
Espelho público direto (Hugging Face): https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset
"""

import os
import urllib.request

DATASET_URL = (
    "https://huggingface.co/datasets/maharshipandya/"
    "spotify-tracks-dataset/resolve/main/dataset.csv"
)
OUTPUT_PATH = os.path.join("data", "dataset.csv")


def download_dataset(url: str = DATASET_URL, output_path: str = OUTPUT_PATH) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        print(f"Dataset já existe em: {output_path} ({os.path.getsize(output_path):,} bytes)")
        return output_path

    print(f"Baixando Spotify Tracks Dataset de {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Download concluído: {output_path} ({os.path.getsize(output_path):,} bytes)")
    return output_path


if __name__ == "__main__":
    download_dataset()
