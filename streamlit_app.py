import streamlit as st
import pandas as pd
from house_price_model import HousePriceModel
import plotly.express as px

def main():
    st.title("🏠 İstanbul Ev Fiyat Tahmini")
    
    # Sidebar
    st.sidebar.header("Ev Özellikleri")
    
    # Default değerler güncellendi
    brut = st.sidebar.number_input("Brüt Metrekare", min_value=40, max_value=1000, value=150)
    net = st.sidebar.number_input("Net Metrekare", min_value=35, max_value=900, value=135)
    
    yas_options = ['0 (Yeni)', '1', '2', '3', '4', '5-10', '11-15', '16-20', '21 Ve Üzeri']
    yas = st.sidebar.selectbox("Bina Yaşı", yas_options, index=2)  # 2 yaşında
    
    oda_options = ['1+1', '2+1', '3+1', '3+2', '4+1', '4+2', '5+1', '5+2', '6+1', '6+2']
    oda = st.sidebar.selectbox("Oda Sayısı", oda_options, index=3)  # 3+2
    
    kat_options = ['Bodrum', 'Zemin', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10+']
    kat = st.sidebar.selectbox("Bulunduğu Kat", kat_options, index=3)  # 2. kat
    
    banyo = st.sidebar.number_input("Banyo Sayısı", min_value=1, max_value=6, value=2)
    
    il = st.sidebar.selectbox("İl", ["İstanbul"])
    
    # İlçe seçenekleri ve default değerleri güncellendi
    ilce_options = {
        "Beşiktaş": {"default_mahalleler": ["Levent", "Etiler", "Ulus", "Bebek", "Arnavutköy", "Ortaköy"], "fiyat_katsayisi": 1.5},
        "Sarıyer": {"default_mahalleler": ["Maslak", "Tarabya", "Yenikoy", "İstinye", "Emirgan"], "fiyat_katsayisi": 1.3},
        "Kadıköy": {"default_mahalleler": ["Moda", "Fenerbahçe", "Suadiye", "Caddebostan", "Göztepe"], "fiyat_katsayisi": 1.2},
        "Üsküdar": {"default_mahalleler": ["Çengelköy", "Beylerbeyi", "Kuzguncuk", "Kandilli", "Acıbadem"], "fiyat_katsayisi": 1.1}
    }
    
    ilce = st.sidebar.selectbox("İlçe", list(ilce_options.keys()), index=0)  # Default: Beşiktaş
    mahalle = st.sidebar.selectbox("Mahalle", ilce_options[ilce]["default_mahalleler"], index=0)  # Default: İlçenin ilk mahallesi
    
    isitma_options = ["Doğalgaz", "Merkezi", "Kombi", "Klima", "Yerden Isıtma"]
    isitma = st.sidebar.selectbox("Isıtma Tipi", isitma_options, index=0)
    
    # Ek özellikler için input alanları
    site_icinde = st.sidebar.selectbox("Site İçerisinde", ["Evet", "Hayır"], index=1)
    kredi_uygun = st.sidebar.selectbox("Krediye Uygunluk", ["Evet", "Hayır"], index=0)
    bina_kat = st.sidebar.number_input("Bina Kat Sayısı", min_value=1, max_value=30, value=5)
    
    # Tahmin butonu
    if st.sidebar.button("Fiyat Tahmini Yap"):
        try:
            model = HousePriceModel()
            model.load_model()
            
            # Input verilerini hazırla
            input_data = {
                'Brüt_Metrekare': brut,
                'Net_Metrekare': net,
                'Binanın_Yaşı': yas,
                'Oda_Sayısı': oda,
                'Bulunduğu_Kat': kat,
                'Banyo_Sayısı': banyo,
                'İl': il,
                'İlçe': ilce,
                'Mahalle': mahalle,
                'Isıtma_Tipi': isitma,
                'Site_İçerisinde': site_icinde,
                'Krediye_Uygunluk': kredi_uygun,
                'Binanın_Kat_Sayısı': bina_kat
            }
            
            # Tahmin yap
            base_prediction = model.predict(input_data)
            
            # İlçe bazlı fiyat düzeltmesi
            adjusted_prediction = base_prediction * ilce_options[ilce]["fiyat_katsayisi"]
            
            # Sonuçları göster
            st.header("Tahmin Sonucu")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Tahmini Fiyat", f"{adjusted_prediction:,.0f} TL")
            
            with col2:
                st.metric("Metrekare Başına", f"{(adjusted_prediction/brut):,.0f} TL/m²")
            
            with col3:
                st.metric("Net Metrekare Başına", f"{(adjusted_prediction/net):,.0f} TL/m²")
            
            # Karşılaştırma bilgisi
            st.info(f"""
            💡 Karşılaştırma:
            - {ilce} ilçesi için ortalama fiyat katsayısı: {ilce_options[ilce]["fiyat_katsayisi"]}x
            - Baz fiyat: {base_prediction:,.0f} TL
            - Lokasyon düzeltmeli fiyat: {adjusted_prediction:,.0f} TL
            """)
            
            # Özellik önem grafiği
            st.subheader("Özellik Önem Dereceleri")
            importance_df = pd.DataFrame({
                'Özellik': model.feature_names,
                'Önem': model.feature_importances_
            }).sort_values('Önem', ascending=True)
            
            fig = px.bar(importance_df, 
                        x='Önem', 
                        y='Özellik',
                        orientation='h',
                        title="Özelliklerin Fiyata Etkisi")
            
            fig.update_layout(
                plot_bgcolor='white',
                showlegend=False
            )
            
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    main() 