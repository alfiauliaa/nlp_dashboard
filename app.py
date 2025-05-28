from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from nltk.tokenize import word_tokenize
import nltk
from sklearn.preprocessing import LabelEncoder
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load model, vectorizer, dan selector
model_with_ratings = joblib.load('models/model_tfidf_with_discrit_logistic_regression.pkl')
model_text_only = joblib.load('models/model_tfidf_only_logistic_regression.pkl')
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
tfidf_selector = joblib.load('models/tfidf_selector.pkl')
le_makanan = joblib.load('models/label_encoder_Makanan_discrit.pkl')
le_layanan = joblib.load('models/label_encoder_Rate Layanan_discrit.pkl')
le_suasana = joblib.load('models/label_encoder_Suasana_discrit.pkl')

# Setup stopwords dan stemmer
nltk_stopwords = set(nltk.corpus.stopwords.words('indonesian'))
sastrawi_stopwords = set(StopWordRemoverFactory().get_stop_words())
custom_stopwords = set(['dan', 'atau'])  # Kurangi stopwords kustom
combined_stopwords = nltk_stopwords.union(sastrawi_stopwords).union(custom_stopwords)
stemmer = StemmerFactory().create_stemmer()

# Fungsi pembersihan teks
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in combined_stopwords]
    tokens = [stemmer.stem(word) for word in tokens]
    cleaned_text = ' '.join(tokens)
    return cleaned_text if cleaned_text.strip() else "empty"

# Fungsi untuk membuat fitur
def prepare_input(text, makanan=None, layanan=None, suasana=None):
    cleaned_text = clean_text(text)
    print(f"Input: {text}")
    print(f"Cleaned text: {cleaned_text}")  # Debugging
    if cleaned_text == "empty":
        raise ValueError("Ulasan tidak menghasilkan teks yang valid setelah pembersihan.")
    
    # Transformasi TF-IDF
    text_features = tfidf_vectorizer.transform([cleaned_text]).toarray()
    text_features_df = pd.DataFrame(text_features, columns=tfidf_vectorizer.get_feature_names_out())
    
    # Seleksi fitur
    selected_text_features = tfidf_selector.transform(text_features_df)
    selected_indices = tfidf_selector.get_support(indices=True)
    selected_feature_names = [tfidf_vectorizer.get_feature_names_out()[i] for i in selected_indices]
    selected_text_features_df = pd.DataFrame(selected_text_features, columns=selected_feature_names)
    print(f"Selected features shape: {selected_text_features_df.shape}")  # Debugging
    
    # Jika rating tidak ada
    if makanan is None or layanan is None or suasana is None:
        return selected_text_features_df, False  # Gunakan model hanya teks
    
    # Diskretisasi rating
    makanan_disc = 'Low' if makanan <= 2.5 else 'Medium' if makanan <= 3.5 else 'High'
    layanan_disc = 'Low' if layanan <= 2.5 else 'Medium' if layanan <= 3.5 else 'High'
    suasana_disc = 'Low' if suasana <= 2.5 else 'Medium' if suasana <= 3.5 else 'High'
    
    # Encode rating
    try:
        makanan_enc = le_makanan.transform([makanan_disc])[0]
        layanan_enc = le_layanan.transform([layanan_disc])[0]
        suasana_enc = le_suasana.transform([suasana_disc])[0]
    except ValueError as e:
        raise ValueError(f"Error saat encoding: {str(e)}. Pastikan rating valid.")
    
    # Buat DataFrame untuk fitur diskret
    discrit_features_df = pd.DataFrame({
        'Makanan_discrit': [makanan_enc],
        'Rate Layanan_discrit': [layanan_enc],
        'Suasana_discrit': [suasana_enc]
    })
    
    # Gabungkan fitur
    features_df = pd.concat([selected_text_features_df, discrit_features_df], axis=1)
    print(f"Final features shape: {features_df.shape}")  # Debugging
    return features_df, True  # Gunakan model dengan rating

# Route untuk index
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            review = request.form.get('review', '').strip()
            makanan = request.form.get('makanan', '')
            layanan = request.form.get('layanan', '')
            suasana = request.form.get('suasana', '')
            
            # Validasi ulasan
            if not review:
                error = "Ulasan tidak boleh kosong!"
            else:
                # Konversi rating
                makanan = float(makanan) if makanan else None
                layanan = float(layanan) if layanan else None
                suasana = float(suasana) if suasana else None
                
                # Validasi rating
                if any(r is not None for r in [makanan, layanan, suasana]):
                    if any(r is None for r in [makanan, layanan, suasana]):
                        error = "Jika memasukkan rating, semua harus diisi (Makanan, Layanan, Suasana)!"
                    elif any(r < 0 or r > 5 for r in [makanan, layanan, suasana] if r is not None):
                        error = "Rating harus antara 0 dan 5!"
                
                if not error:
                    # Siapkan fitur
                    features_df, use_ratings = prepare_input(review, makanan, layanan, suasana)
                    model = model_with_ratings if use_ratings else model_text_only
                    
                    # Prediksi
                    pred = model.predict(features_df)[0]
                    label_mapping = {0: 'Negatif', 1: 'Positif', 2: 'Netral', 3: 'Others'}
                    prediction = label_mapping[pred]
                    print(f"Prediksi: {prediction}")  # Debugging
        except ValueError as e:
            error = f"Terjadi kesalahan: {str(e)}"
        except Exception as e:
            error = f"Terjadi kesalahan tidak terduga: {str(e)}"
    
    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)