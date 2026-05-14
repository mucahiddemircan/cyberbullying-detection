import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize NLTK resources
# These should be pre-downloaded using init_nltk.py
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text: str) -> str:
    """
    Performs initial text cleaning including noise removal and normalization.
    """
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove user mentions (@user)
    text = re.sub(r'@\w+', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove hashtags but keep the text if needed (currently removing)
    text = re.sub(r'#\w+', '', text)
    
    # Clean non-alphanumeric characters but preserve basic sentiment markers
    # We keep letters, numbers, and common punctuation that might indicate tone
    text = re.sub(r'[^a-zA-Z0-9\s?!.,]', '', text)
    
    # Remove numbers if they don't carry specific meaning in this context
    text = re.sub(r'\d+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Clean up whitespace
    text = " ".join(text.split())
    
    # Remove very short words (single characters) as they are often noise
    text = " ".join(word for word in text.split() if len(word) > 1)
    
    return text

def remove_stop_words(text: str) -> str:
    """
    Remove common English stop words.
    """
    return " ".join([word for word in str(text).split() if word not in stop_words])

def stemming(text: str) -> str:
    """
    Applies Porter Stemmer to reduce words to their root form.
    
    Stemming is generally more effective for social media text (tweets) 
    than lemmatization due to the informal nature of the language.
    """
    return " ".join([stemmer.stem(word) for word in text.split()])

def clean_text(text: str) -> str:
    """
    Full end-to-end cleaning pipeline for raw text.
    """
    text = preprocess(text)
    text = remove_stop_words(text)
    text = stemming(text)
    return text

