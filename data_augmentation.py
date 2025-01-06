import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random

def augment_data(input_file, output_file, multiplication_factor=10):
    # Veri setini oku
    df = pd.read_csv(input_file)
    
    # Yeni veri çerçevesi oluştur
    augmented_data = []
    
    # Her satır için yeni örnekler oluştur
    for _, row in df.iterrows():
        # Orijinal satırı ekle
        augmented_data.append(row.to_dict())
        
        # Yeni örnekler oluştur
        for _ in range(multiplication_factor - 1):
            new_row = row.copy()
            
            # Metrekare değişimi (±10%)
            brut_m2 = float(new_row['Brüt_Metrekare'])
            net_m2 = float(new_row['Net_Metrekare'])
            m2_change = random.uniform(0.90, 1.10)
            new_row['Brüt_Metrekare'] = round(brut_m2 * m2_change)
            new_row['Net_Metrekare'] = round(net_m2 * m2_change)
            
            # Fiyat değişimi (±15%)
            price = float(new_row['Fiyat'])
            price_change = random.uniform(0.85, 1.15)
            new_row['Fiyat'] = round(price * price_change)
            
            # Bina yaşı değişimi
            age = str(new_row['Binanın_Yaşı'])
            if age != '0 (Yeni)':
                try:
                    if '-' in age:
                        start, end = map(int, age.split('-'))
                        new_age = random.randint(start, end)
                    elif 'Ve Üzeri' in age:
                        base = int(age.split()[0])
                        new_age = random.randint(base, base + 5)
                    else:
                        current_age = int(float(age))
                        new_age = max(0, current_age + random.randint(-2, 2))
                    new_row['Binanın_Yaşı'] = str(new_age)
                except:
                    pass
            
            # Kat değişimi
            floor = str(new_row['Bulunduğu_Kat'])
            if 'Kat' in floor and not any(x in floor.lower() for x in ['bodrum', 'zemin', 'giriş', 'çatı']):
                try:
                    current_floor = int(''.join(filter(str.isdigit, floor)))
                    new_floor = max(1, current_floor + random.randint(-2, 2))
                    new_row['Bulunduğu_Kat'] = f"{new_floor}. Kat"
                except:
                    pass
            
            augmented_data.append(new_row)
    
    # Yeni veri setini oluştur
    augmented_df = pd.DataFrame(augmented_data)
    
    # Sütun sırasını orijinal veri setiyle aynı yap
    augmented_df = augmented_df[df.columns]
    
    # Veriyi karıştır
    augmented_df = augmented_df.sample(frac=1).reset_index(drop=True)
    
    # Yeni veri setini kaydet
    augmented_df.to_csv(output_file, index=False)
    
    return len(augmented_df)

if __name__ == "__main__":
    input_file = "house_dataset.csv"
    output_file = "augmented_house_dataset.csv"
    multiplication_factor = 10
    
    total_rows = augment_data(input_file, output_file, multiplication_factor)
    print(f"Veri seti {multiplication_factor} kat büyütüldü.")
    print(f"Toplam satır sayısı: {total_rows}") 