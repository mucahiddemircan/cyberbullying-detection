import nltk

"""
Utility script to initialize and download required NLTK resources.
"""

def init_nltk():
    print("Downloading NLTK data...")
    resources = [
        'stopwords',
        'punkt',
        'punkt_tab',
        'wordnet',
        'omw-1.4'
    ]
    for resource in resources:
        try:
            nltk.download(resource)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {e}")

if __name__ == "__main__":
    init_nltk()
