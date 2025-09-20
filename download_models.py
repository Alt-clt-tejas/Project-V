# download_models.py
import spacy
from sentence_transformers import SentenceTransformer
from textblob.download_corpora import main as download_textblob_corpora

# --- Configuration ---
# You can change these model names if you decide to use different ones later.
SPACY_MODEL = "en_core_web_sm"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

def download_all_models():
    """
    Downloads and caches all the pre-trained models required for the
    Enrichment Service. Running this script once will prevent downloads
    during the application's runtime.
    """
    print("--- Starting download of pre-trained models ---")

    # 1. Download Sentence Transformer Model
    # This model is used for creating vector embeddings.
    try:
        print(f"\n[1/3] Downloading Sentence Transformer model: '{SENTENCE_TRANSFORMER_MODEL}'...")
        # The library handles caching automatically in the user's home directory.
        # Instantiating the model triggers the download.
        SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        print(f"[SUCCESS] Model '{SENTENCE_TRANSFORMER_MODEL}' is ready.")
    except Exception as e:
        print(f"[ERROR] Failed to download Sentence Transformer model: {e}")
        print("Please check your internet connection.")

    # 2. Download spaCy Model
    # This model is used for Named Entity Recognition (NER) and tokenization.
    try:
        print(f"\n[2/3] Downloading spaCy model: '{SPACY_MODEL}'...")
        # spaCy has its own download command-line utility.
        spacy.cli.download(SPACY_MODEL)
        print(f"[SUCCESS] Model '{SPACY_MODEL}' is ready.")
    except Exception as e:
        print(f"[ERROR] Failed to download spaCy model: {e}")
        print("This might happen if you have permission issues or network problems.")

    # 3. Download TextBlob Corpora
    # These are the dictionaries and word lists TextBlob uses for sentiment analysis.
    try:
        print("\n[3/3] Downloading TextBlob corpora (punkt, averaged_perceptron_tagger)...")
        # TextBlob's download utility can be called directly from Python.
        download_textblob_corpora.run(args=['-q', 'punkt', 'averaged_perceptron_tagger'])
        print("[SUCCESS] TextBlob corpora are ready.")
    except Exception as e:
        print(f"[ERROR] Failed to download TextBlob corpora: {e}")

    print("\n--- Model download process complete ---")

if __name__ == "__main__":
    download_all_models()