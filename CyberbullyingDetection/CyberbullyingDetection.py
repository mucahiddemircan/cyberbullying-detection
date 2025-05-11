#%% Kütüphaneler
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import os
import joblib

nltk.download('stopwords')

output_dir = 'model_evaluation_outputs'
os.makedirs(output_dir, exist_ok=True)

print("Veri Yükleniyor...")
dataset = pd.read_csv('cyberbullying_tweets.csv')
# Veri setini karıştır
dataset = dataset.sample(frac=1).reset_index(drop=True)

#%% 1- Veri Keşfi ve Analizi
print("\n VERİ KEŞFİ VE ANALİZİ")
# Temel bilgileri görüntüleme
print("Veri Seti Boyutu:", dataset.shape)

print("\nVeri Seti Bilgileri:")
print(dataset.info())

print("\nBetimleyici İstatistikler:")
print(dataset.describe())

print("\nSınıf Dağılımı:")
class_counts = dataset['cyberbullying_type'].value_counts()
print(class_counts)

plt.figure(figsize=(10, 6))
sns.countplot(x='cyberbullying_type', data=dataset)
plt.title('Siber Zorbalık Türlerinin Dağılımı')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.close()

dataset['text_length'] = dataset['tweet_text'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(data=dataset, x='text_length', hue='cyberbullying_type', kde=True, bins=50)
plt.title('Tweet Uzunluğu Dağılımı')
plt.xlabel('Metin Uzunluğu')
plt.ylabel('Frekans')
plt.show()
plt.close()

avg_length_by_class = dataset.groupby('cyberbullying_type')['text_length'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_length_by_class.index, y=avg_length_by_class.values)
plt.title('Sınıflara Göre Ortalama Tweet Uzunluğu')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.close()

#%% 2- Veri Ön İşleme
def preprocess(text):
    # Küçük harfe dönüştürme
    text = str(text).lower()
    # Kullanıcı adlarını temizleme (@user)
    text = re.sub(r'@\w+', '', text)
    # URL'leri temizleme
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Hashtag'leri temizleme (#hashtag)
    text = re.sub(r'#\w+', '', text)
    # Emojileri temizleme
    text = re.sub(r'[^\w\s]', '', text)
    # Sayıları temizleme
    text = re.sub(r'\d+', '', text)
    # HTML etiketlerini temizleme
    text = re.sub(r'<.*?>', '', text)
    # Fazla boşlukları temizleme
    text = " ".join(text.split())
    # Tek karakterli kelimeleri temizleme
    text = " ".join(word for word in text.split() if len(word) > 1)
    return text

# Durak kelimelerinin (Stop Words) Kaldırılması
stop_words = set(stopwords.words('english'))
def remove_stop_words(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

# Lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatization(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

print("Veri temizleme işlemleri uygulanıyor...")
dataset_preprocessed = dataset.copy()
dataset_preprocessed['tweet_text_processed'] = dataset_preprocessed['tweet_text'].apply(preprocess)
dataset_preprocessed['tweet_text_processed'] = dataset_preprocessed['tweet_text_processed'].apply(remove_stop_words)
dataset_preprocessed['tweet_text_processed'] = dataset_preprocessed['tweet_text_processed'].apply(lemmatization)

print("\nÖrnek Tweet (Ham):")
print(dataset['tweet_text'].iloc[0])
print("\nÖrnek Tweet (İşlenmiş):")
print(dataset_preprocessed['tweet_text_processed'].iloc[0])

# Çıktı örnekleri için Label Encoding
le = LabelEncoder()
dataset_preprocessed['cyberbullying_type_encoded'] = le.fit_transform(dataset_preprocessed['cyberbullying_type'])

label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("\nEtiket Eşleştirmeleri:")
print(label_mapping)

X = dataset_preprocessed['tweet_text_processed']
y = dataset_preprocessed['cyberbullying_type_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

print("Eğitim veri seti boyutu: ", X_train.shape[0])
print("Test veri seti boyutu: ", X_test.shape[0])

#%% 3- Öznitelik Çıkarımı
# En sık kullanılan kelimeler için analiz
def get_top_n_words(corpus, n=20):
    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

for cyberbullying_type in dataset_preprocessed['cyberbullying_type'].unique():
    print(f"\nEn sık kullanılan kelimeler ({cyberbullying_type}):")
    texts = dataset_preprocessed[dataset_preprocessed['cyberbullying_type'] == cyberbullying_type]['tweet_text_processed']
    top_words = get_top_n_words(texts, 10)
    for word, freq in top_words:
        print(f"{word}: {freq}")

print("\nTF-IDF Vektörizasyon uygulanıyor...")
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print("TF-IDF vektör boyutu: ", X_train_tfidf.shape)

print("\nBag of Words Vektörizasyon uygulanıyor...")
count_vectorizer = CountVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_bow = count_vectorizer.fit_transform(X_train)
X_test_bow = count_vectorizer.transform(X_test)
print(f"BoW vektör boyutu: {X_train_bow.shape}")
'''
#%% 4- Model Eğitimi, Değerlendirmesi ve Kaydedilmesi
# Model Eğitim Fonksiyonu
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# Model Değerlendirme Fonksiyonu
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{model_name} Sonuçları:")
    print(f"Doğruluk: {accuracy:.4f}")
    
    # Sınıflandırma Raporu
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print("\nSınıflandırma Raporu:")
    print(report)

    with open(f'{output_dir}/classification_report_{model_name}.txt', 'w') as file:
        file.write(report)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Karmaşıklık Matrisi - {model_name}')
    plt.savefig(f'{output_dir}/confusion_matrix_{model_name}.png')
    plt.close()

    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(le.classes_):
            fpr, tpr, _ = roc_curve(y_test == i, y_scores[:, i])
            plt.plot(fpr, tpr, label=f'{class_name}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC Eğrisi - {model_name}')
        plt.legend()
        plt.savefig(f'{output_dir}/roc_curve_{model_name}.png')
        plt.close()

# Model Listesi
models = [
    ("Lojistik Regresyon", LogisticRegression(max_iter=1000, class_weight='balanced')),
    ("SVM", SVC(kernel='linear', probability=True, class_weight='balanced')),
    ("YSA", MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, max_iter=300, early_stopping=True)),
    ("XGBoost", XGBClassifier(n_estimators=200, max_depth=6,))]

for model_name, model in models:
    # TF-IDF ile eğit
    trained_model_tfidf = train_model(model, X_train_tfidf, y_train)
    evaluate_model(trained_model_tfidf, X_test_tfidf, y_test, f"{model_name}_TFIDF")
    joblib.dump(trained_model_tfidf, f"{output_dir}/{model_name}_TFIDF_model.pkl")
    
    # BoW ile eğit
    trained_model_bow = train_model(model, X_train_bow, y_train)
    evaluate_model(trained_model_bow, X_test_bow, y_test, f"{model_name}_BoW")
    joblib.dump(trained_model_bow, f"{output_dir}/{model_name}_BoW_model.pkl")

joblib.dump(tfidf_vectorizer, f"{output_dir}/TFIDF_vectorizer.pkl")
joblib.dump(count_vectorizer, f"{output_dir}/BoW_vectorizer.pkl")
'''
#%% 5-  Kullanıcıdan Model ve Vektörizer Seçimiyle Tahmin
def predict_text(text, model, vectorizer, le):
    text_processed = lemmatization(remove_stop_words(preprocess(text)))
    vectorized_text = vectorizer.transform([text_processed])
    prediction = model.predict(vectorized_text)
    predicted_label = le.inverse_transform(prediction)[0]
    print(f"\nTahmin Edilen Siber Zorbalık Türü: {predicted_label}")

print("\n>>> MODELLER <<<")
model_names = ["LojistikRegresyon", "SVM", "YSA", "XGBoost"]
vectorizer_names = ["TFIDF", "BoW"]

for i, name in enumerate(model_names, 1):
    print(f"{i}. {name}")
model_choice = int(input("Bir model seçin (numara): "))
chosen_model_name = model_names[model_choice - 1]

print("\n>>> VEKTÖRİZERLER <<<")
for i, name in enumerate(vectorizer_names, 1):
    print(f"{i}. {name}")
vectorizer_choice = int(input("Bir vektörizer seçin (numara): "))
chosen_vectorizer_name = vectorizer_names[vectorizer_choice - 1]

# Dosya yolları
model_path = f"{output_dir}/{chosen_model_name}_{chosen_vectorizer_name}_model.pkl"
vectorizer_path = f"{output_dir}/{chosen_vectorizer_name}_vectorizer.pkl"

# Yükle
model_loaded = joblib.load(model_path)
vectorizer_loaded = joblib.load(vectorizer_path)

# Tahmin döngüsü
while True:
    user_input = input("\nBir tweet metni girin (çıkmak için 'q' yazın): ")
    if user_input.lower() == 'q':
        break
    predict_text(user_input, model_loaded, vectorizer_loaded, le)

