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
import os

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
custom_stopwords = set(['dan', 'atau'])
combined_stopwords = nltk_stopwords.union(sastrawi_stopwords).union(custom_stopwords)
stemmer = StemmerFactory().create_stemmer()

# Fungsi pembersihan teks
def clean_text(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return "empty"
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in combined_stopwords and len(word) > 1]
    tokens = [stemmer.stem(word) for word in tokens]
    cleaned_text = ' '.join(tokens)
    return cleaned_text if cleaned_text.strip() else "empty"

# Fungsi untuk membuat fitur teks
def prepare_text_features(text):
    cleaned_text = clean_text(text)
    print(f"Input: {text}")
    print(f"Cleaned text: {cleaned_text}")
    if cleaned_text == "empty":
        raise ValueError("Ulasan tidak menghasilkan teks yang valid setelah pembersihan.")
    
    text_features = tfidf_vectorizer.transform([cleaned_text]).toarray()
    text_features_df = pd.DataFrame(text_features, columns=tfidf_vectorizer.get_feature_names_out())
    
    selected_text_features = tfidf_selector.transform(text_features_df)
    selected_indices = tfidf_selector.get_support(indices=True)
    selected_feature_names = [tfidf_vectorizer.get_feature_names_out()[i] for i in selected_indices]
    selected_text_features_df = pd.DataFrame(selected_text_features, columns=selected_feature_names)
    print(f"Selected features shape: {selected_text_features_df.shape}")
    
    return selected_text_features_df

# Fungsi untuk membuat fitur lengkap (dengan rating)
def prepare_input(text, makanan=None, layanan=None, suasana=None):
    cleaned_text = clean_text(text)
    print(f"Input: {text}")
    print(f"Cleaned text: {cleaned_text}")
    if cleaned_text == "empty":
        raise ValueError("Ulasan tidak menghasilkan teks yang valid setelah pembersihan.")
    
    text_features = tfidf_vectorizer.transform([cleaned_text]).toarray()
    text_features_df = pd.DataFrame(text_features, columns=tfidf_vectorizer.get_feature_names_out())
    
    selected_text_features = tfidf_selector.transform(text_features_df)
    selected_indices = tfidf_selector.get_support(indices=True)
    selected_feature_names = [tfidf_vectorizer.get_feature_names_out()[i] for i in selected_indices]
    selected_text_features_df = pd.DataFrame(selected_text_features, columns=selected_feature_names)
    print(f"Selected features shape: {selected_text_features_df.shape}")
    
    if makanan is None or layanan is None or suasana is None:
        return selected_text_features_df, False
    
    makanan_disc = 'Low' if makanan <= 2.5 else 'Medium' if makanan <= 3.5 else 'High'
    layanan_disc = 'Low' if layanan <= 2.5 else 'Medium' if layanan <= 3.5 else 'High'
    suasana_disc = 'Low' if suasana <= 2.5 else 'Medium' if suasana <= 3.5 else 'High'
    
    try:
        makanan_enc = le_makanan.transform([makanan_disc])[0]
        layanan_enc = le_layanan.transform([layanan_disc])[0]
        suasana_enc = le_suasana.transform([suasana_disc])[0]
    except ValueError as e:
        raise ValueError(f"Error saat encoding: {str(e)}. Pastikan rating valid.")
    
    discrit_features_df = pd.DataFrame({
        'Makanan_discrit': [makanan_enc],
        'Rate Layanan_discrit': [layanan_enc],
        'Suasana_discrit': [suasana_enc]
    })
    
    features_df = pd.concat([selected_text_features_df, discrit_features_df], axis=1)
    print(f"Final features shape: {features_df.shape}")
    return features_df, True

# Route untuk index
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            review = request.form.get('review', '').strip()
            if not review:
                error = "Ulasan tidak boleh kosong!"
            else:
                features_df, use_ratings = prepare_input(review)
                model = model_text_only
                pred = model.predict(features_df)[0]
                label_mapping = {0: 'Negatif', 1: 'Positif', 2: 'Netral', 3: 'Others'}
                prediction = label_mapping[pred]
                print(f"Prediksi: {prediction}")
        except ValueError as e:
            error = f"Terjadi kesalahan: {str(e)}"
        except Exception as e:
            error = f"Terjadi kesalahan tidak terduga: {str(e)}"
    
    return render_template('index.html', prediction=prediction, error=error)

# Route untuk upload CSV
@app.route('/upload-csv', methods=['GET', 'POST'])
def upload_csv():
    results = None
    error = None

    if request.method == 'POST':
        if 'csv_file' not in request.files:
            error = "File CSV tidak ditemukan!"
        else:
            file = request.files['csv_file']
            if file.filename == '':
                error = "File CSV tidak dipilih!"
            elif not file.filename.endswith('.csv'):
                error = "File harus berformat CSV!"
            else:
                file_path = os.path.join('uploads', file.filename)
                os.makedirs('uploads', exist_ok=True)
                file.save(file_path)

                try:
                    # Baca file CSV
                    df = pd.read_csv(file_path, encoding='utf-8')
                    print(f"Loaded CSV columns: {df.columns.tolist()}")
                    print(f"Loaded CSV data (first 5 rows):\n{df.head().to_string()}")

                    # Pastikan kolom yang diperlukan ada
                    required_columns = ['Ulasan', 'Label']
                    if not all(col in df.columns for col in required_columns):
                        error = "File CSV harus memiliki kolom 'Ulasan' dan 'Label'!"
                        print(f"Error: Missing required columns. Found: {df.columns.tolist()}")
                    else:
                        # Pastikan kolom Ulasan dan Label tidak kosong
                        df = df.dropna(subset=['Ulasan', 'Label']).reset_index(drop=True)
                        print(f"Data after dropping NA (first 5 rows):\n{df.head().to_string()}")
                        if df.empty:
                            error = "File CSV tidak berisi data yang valid (kolom 'Ulasan' atau 'Label' kosong)!"
                        else:
                            # Bersihkan ulasan
                            df['Ulasan Bersih'] = df['Ulasan'].apply(clean_text)
                            print(f"Ulasan Bersih (first 5 rows):\n{df[['Ulasan', 'Ulasan Bersih']].head().to_string()}")

                            # Filter ulasan yang kosong setelah pembersihan
                            df = df[df['Ulasan Bersih'] != "empty"].copy().reset_index(drop=True)
                            print(f"Data after filtering 'empty' (first 5 rows):\n{df.head().to_string()}")
                            if df.empty:
                                error = "Tidak ada ulasan valid setelah pembersihan!"
                            else:
                                # Transformasi fitur teks
                                X_features = []
                                for text in df['Ulasan Bersih']:
                                    try:
                                        features = prepare_text_features(text)
                                        X_features.append(features)
                                    except ValueError as e:
                                        print(f"Skipping text due to error: {str(e)}")
                                        continue
                                if not X_features:
                                    error = "Tidak ada ulasan yang dapat diproses untuk prediksi!"
                                else:
                                    X = pd.concat(X_features, ignore_index=True)
                                    print(f"Feature matrix shape: {X.shape}")

                                    # Prediksi
                                    y_pred = model_text_only.predict(X)
                                    label_mapping = {0: 'Negatif', 1: 'Positif', 2: 'Netral', 3: 'Others'}
                                    df['predicted_label'] = [label_mapping[pred] for pred in y_pred]
                                    df['status'] = df.apply(
                                        lambda row: 'Benar' if str(row['Label']).strip() == str(row['predicted_label']).strip() else 'Salah',
                                        axis=1
                                    )
                                    print(f"Data with predictions (first 5 rows):\n{df[['Ulasan', 'Label', 'predicted_label', 'status']].head().to_string()}")

                                    # Siapkan hasil untuk ditampilkan dengan kunci yang sesuai dengan template
                                    results = df[['Ulasan', 'Label', 'predicted_label', 'status']].to_dict('records')
                                    # Ubah kunci agar sesuai dengan template
                                    results = [
                                        {
                                            'ulasan': row['Ulasan'],
                                            'true_label': row['Label'],
                                            'predicted_label': row['predicted_label'],
                                            'status': row['status']
                                        }
                                        for row in results
                                    ]
                                    print(f"Final results: {results}")

                except UnicodeDecodeError:
                    error = "File CSV tidak dapat dibaca. Pastikan file menggunakan encoding UTF-8!"
                except Exception as e:
                    error = f"Terjadi kesalahan saat memproses file: {str(e)}"
                    print(f"Error during processing: {str(e)}")
                finally:
                    if os.path.exists(file_path):
                        os.remove(file_path)

    return render_template('upload_csv.html', results=results, error=error)

if __name__ == '__main__':
    app.run(debug=True)