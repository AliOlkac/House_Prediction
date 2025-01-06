import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


class HousePriceModel:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.columns = [
            'Brüt_Metrekare',
            'Net_Metrekare',
            'Bina_Yaşı',
            'Oda_Sayısı',
            'Bulunduğu_Kat',
            'Banyo_Sayısı',
            'İl',
            'İlçe',
            'Mahalle',
            'Isıtma_Tipi',
            'Site_İçerisinde',
            'Krediye_Uygunluk',
            'Binanın_Kat_Sayısı'
        ]
        
    def preprocess_data(self, df):
        processed_data = pd.DataFrame()
        
        # Metrekare işlemleri
        processed_data['Brüt_Metrekare'] = pd.to_numeric(df['Brüt_Metrekare'], errors='coerce')
        processed_data['Net_Metrekare'] = pd.to_numeric(df['Net_Metrekare'], errors='coerce')
        processed_data['Alan_Oranı'] = processed_data['Net_Metrekare'] / processed_data['Brüt_Metrekare']
        
        # Bina Yaşı işleme
        def convert_age(age):
            if pd.isna(age):
                return None
            age = str(age).strip()
            if age == '0 (Yeni)':
                return 0
            elif '-' in age:
                start, end = map(int, age.split('-'))
                return (start + end) / 2
            elif 'Ve Üzeri' in age:
                return float(age.split()[0])
            return float(age)
        
        processed_data['Bina_Yaşı'] = df['Binanın_Yaşı'].apply(convert_age)
        
        # Oda Sayısı işleme
        def extract_room_number(room):
            if pd.isna(room):
                return None
            room = str(room).strip()
            try:
                if '+' in room:
                    parts = room.split('+')
                    total = 0
                    for part in parts:
                        if '.' in part:
                            total += float(part)
                        else:
                            total += int(part)
                    return total
                return float(room.split()[0]) if 'Oda' in room else float(room)
            except:
                return None
        
        processed_data['Oda_Sayısı'] = df['Oda_Sayısı'].apply(extract_room_number)
        
        # Banyo Sayısı - Düzeltilmiş kısım
        processed_data['Banyo_Sayısı'] = pd.to_numeric(df['Banyo_Sayısı'], errors='coerce')
        
        # Kategorik değişkenleri dönüştür
        categorical_columns = ['İl', 'İlçe', 'Mahalle', 'Isıtma_Tipi', 'Site_İçerisinde', 'Krediye_Uygunluk']
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                processed_data[col] = self.label_encoders[col].fit_transform(df[col].fillna('Bilinmiyor'))
            else:
                # Yeni kategorileri ele al
                unique_values = df[col].unique()
                known_values = self.label_encoders[col].classes_
                new_values = set(unique_values) - set(known_values)
                if new_values:
                    self.label_encoders[col].classes_ = np.concatenate([known_values, list(new_values)])
                processed_data[col] = self.label_encoders[col].transform(df[col].fillna('Bilinmiyor'))
        
        # Kat bilgisi işleme
        def process_floor(floor):
            if pd.isna(floor):
                return 0
            floor = str(floor).lower()
            if 'bodrum' in floor:
                return -1
            elif 'zemin' in floor or 'giriş' in floor:
                return 0
            elif 'çatı' in floor:
                return 99
            try:
                return int(''.join(filter(str.isdigit, floor)))
            except:
                return 0
        
        processed_data['Bulunduğu_Kat'] = df['Bulunduğu_Kat'].apply(process_floor)
        
        # Bina kat sayısı
        processed_data['Bina_Kat_Sayısı'] = pd.to_numeric(df['Binanın_Kat_Sayısı'], errors='coerce')
        
        # Eksik değerleri doldur
        numeric_columns = ['Brüt_Metrekare', 'Net_Metrekare', 'Alan_Oranı', 'Bina_Yaşı', 
                         'Oda_Sayısı', 'Banyo_Sayısı', 'Bina_Kat_Sayısı']
        
        # Sayısal değerleri doldur
        for col in numeric_columns:
            if col in processed_data.columns:
                processed_data[col] = processed_data[col].fillna(processed_data[col].median())
        
        return processed_data

    def train(self, data_path):
        # Veriyi oku
        df = pd.read_csv(data_path)
        
        # Aykırı değerleri temizle
        def remove_outliers(df, columns, n_std=3):
            for col in columns:
                mean = df[col].mean()
                std = df[col].std()
                df = df[~((df[col] - mean).abs() > n_std * std)]
            return df
        
        df = remove_outliers(df, ['Fiyat', 'Brüt_Metrekare', 'Net_Metrekare'])
        
        # Veriyi hazırla
        X = self.preprocess_data(df)
        y = df['Fiyat']
        
        # Veriyi ölçeklendir
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Model
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=4,
            random_state=42
        )
        
        # Model eğitimi
        self.model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        
        # Tahminler
        y_pred = self.model.predict(X_test)
        
        # Feature importance
        self.feature_names = X.columns
        self.feature_importances_ = self.model.feature_importances_
        
        # Metrikleri hesapla
        metrics = {
            'R2': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'CV_Score_Mean': cv_scores.mean(),
            'CV_Score_Std': cv_scores.std()
        }
        
        # En önemli özellikleri yazdır
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nEn Önemli Özellikler:")
        print(feature_importance.head())
        
        return metrics

    def predict(self, features):
        # DataFrame'i düzgün şekilde oluştur
        input_df = pd.DataFrame(features, index=[0])
        
        # Veriyi işle
        processed_features = self.preprocess_data(input_df)
        
        # Eksik sütunları kontrol et ve ekle
        for col in self.feature_names:
            if col not in processed_features.columns:
                processed_features[col] = 0
        
        # Sütunları doğru sıraya koy
        processed_features = processed_features[self.feature_names]
        
        # Ölçeklendir ve tahmin yap
        scaled_features = self.scaler.transform(processed_features)
        return self.model.predict(scaled_features)[0]

    def save_model(self, path='house_price_model.joblib'):
        joblib.dump({
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'feature_importances_': self.feature_importances_,
            'scaler': self.scaler,
            'columns': self.columns
        }, path)

    def load_model(self, path='house_price_model.joblib'):
        saved = joblib.load(path)
        self.model = saved['model']
        self.label_encoders = saved['label_encoders']
        self.feature_names = saved['feature_names']
        self.feature_importances_ = saved['feature_importances_']
        self.scaler = saved['scaler']
        self.columns = saved['columns']

if __name__ == "__main__":
    model = HousePriceModel()
    metrics = model.train("augmented_house_dataset.csv")
    print("\nModel Performans Metrikleri:")
    print(f"R² Skoru: {metrics['R2']:.4f}")
    print(f"Ortalama Mutlak Hata (MAE): {metrics['MAE']:,.0f} TL")
    print(f"Kök Ortalama Kare Hata (RMSE): {metrics['RMSE']:,.0f} TL")
    print(f"Cross-Validation R² Skoru: {metrics['CV_Score_Mean']:.4f} (±{metrics['CV_Score_Std']:.4f})")
    model.save_model()