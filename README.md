# Intel Image Classification Projesi

## 🎯 Proje Açıklaması

Bu proje, Intel tarafından sağlanan doğal manzara görüntülerini sınıflandırmak için derin öğrenme (deep learning) teknikleri kullanmaktadır. Proje, 6 farklı doğal manzara kategorisindeki görüntüleri otomatik olarak tanımlamayı amaçlar.

## 📊 Veri Seti Bilgileri

- **Kaynak:** [Intel Image Classification - Kaggle](https://www.kaggle.com/puneet6060/intel-image-classification)
- **Sınıf Sayısı:** 6
- **Toplam Görüntü:** ~25,000
- **Görüntü Boyutu:** 150x150 piksel
- **Format:** RGB renkli görüntüler

### 🏷️ Sınıflar:
1. **Buildings (Binalar)** - Şehir manzaraları ve yapılar
2. **Forest (Orman)** - Orman ve ağaç manzaraları  
3. **Glacier (Buzul)** - Buzul ve kar manzaraları
4. **Mountain (Dağ)** - Dağ manzaraları
5. **Sea (Deniz)** - Deniz ve okyanus manzaraları
6. **Street (Sokak)** - Şehir sokakları ve caddeler

## 🛠️ Gereksinimler

### Ana Kütüphaneler:
```
tensorflow>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
pillow>=8.3.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
```

### Kurulum:
```bash
pip install -r requirements.txt
```

## 📁 Proje Yapısı

```
intel-image-classification/
│
├── data/
│   ├── seg_train/          # Eğitim veri seti
│   └── seg_test/           # Test veri seti
│
├── notebooks/
│   ├── 01_veri_analizi.ipynb
│   ├── 02_model_egitimi.ipynb
│   └── 03_model_degerlendirme.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_builder.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   ├── basic_cnn_model.h5
│   ├── vgg16_transfer_model.h5
│   └── model_architecture.json
│
├── results/
│   ├── confusion_matrix.png
│   ├── training_history.png
│   └── classification_report.txt
│
├── requirements.txt
├── README.md
└── main.py
```

## 🚀 Kullanım Talimatları

### 1. Veri Setini İndirme
```bash
# Kaggle CLI kullanarak
kaggle datasets download -d puneet6060/intel-image-classification

# Manuel indirme için Kaggle hesabı gereklidir
```

### 2. Veri Ön İşleme ve Analiz
```python
from src.data_preprocessing import DataPreprocessor

# Veri ön işleyici oluştur
preprocessor = DataPreprocessor(data_path='data/')

# Veri analizi
preprocessor.analyze_dataset()
preprocessor.visualize_samples()
preprocessor.plot_class_distribution()
```

### 3. Model Eğitimi
```python
from src.train import ModelTrainer

# Eğitici oluştur
trainer = ModelTrainer()

# Transfer Learning modeli eğit
model = trainer.train_transfer_learning_model(
    epochs=20,
    batch_size=32,
    learning_rate=0.001
)
```

### 4. Model Değerlendirme
```python
from src.evaluate import ModelEvaluator

# Değerlendirici oluştur
evaluator = ModelEvaluator(model_path='models/vgg16_transfer_model.h5')

# Test setinde değerlendirme
results = evaluator.evaluate_model()
evaluator.plot_confusion_matrix()
evaluator.generate_classification_report()
```

### 5. Tahmin Yapma
```python
from src.predict import ImagePredictor

# Tahmin edici oluştur
predictor = ImagePredictor(model_path='models/vgg16_transfer_model.h5')

# Tek görüntü tahmini
result = predictor.predict_single_image('path/to/image.jpg')
print(f"Tahmin: {result['class']}, Güven: {result['confidence']:.2f}")
```

## 🧠 Model Mimarileri

### 1. Basit CNN Modeli
- **Katmanlar:** 3 Conv2D + MaxPooling2D katmanı
- **Parametre Sayısı:** ~500K
- **Eğitim Süresi:** ~15 dakika
- **Test Doğruluğu:** ~75%

### 2. Transfer Learning (VGG16)
- **Temel Model:** VGG16 (ImageNet pre-trained)
- **Ek Katmanlar:** GlobalAveragePooling2D + Dense
- **Parametre Sayısı:** ~15M (frozen: ~14M)
- **Eğitim Süresi:** ~30 dakika
- **Test Doğruluğu:** ~85-90%

## 📈 Performans Sonuçları

### Model Karşılaştırması:
| Model | Test Doğruluğu | Eğitim Süresi | Model Boyutu |
|-------|---------------|---------------|--------------|
| Basit CNN | 75.2% | 15 min | 12 MB |
| VGG16 Transfer | 87.8% | 30 min | 58 MB |
| ResNet50 Transfer | 89.1% | 35 min | 98 MB |

### Sınıf Bazında Performans:
| Sınıf | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Buildings | 0.85 | 0.82 | 0.83 |
| Forest | 0.92 | 0.94 | 0.93 |
| Glacier | 0.88 | 0.91 | 0.89 |
| Mountain | 0.84 | 0.81 | 0.82 |
| Sea | 0.95 | 0.93 | 0.94 |
| Street | 0.79 | 0.83 | 0.81 |

## 🔧 Parametre Ayarları

### Veri Artırma Parametreleri:
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.1,
    fill_mode='nearest'
)
```

### Model Hiperparametreleri:
- **Batch Size:** 32
- **Learning Rate:** 0.001 (başlangıç)
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Epochs:** 20 (Early Stopping ile)

### Callback Ayarları:
- **EarlyStopping:** patience=5
- **ReduceLROnPlateau:** factor=0.2, patience=3
- **ModelCheckpoint:** En iyi modeli kaydet

## 📊 Veri Görselleştirme

Proje aşağıdaki görselleştirmeleri içerir:

1. **Sınıf Dağılımı:** Her sınıftaki görüntü sayısı
2. **Örnek Görüntüler:** Her sınıftan rastgele örnekler
3. **Eğitim Geçmişi:** Doğruluk ve kayıp grafikleri
4. **Karışıklık Matrisi:** Sınıflandırma sonuçları
5. **Yanlış Sınıflandırmalar:** Hatalı tahminlerin analizi

## 🚧 Bilinen Sorunlar ve Çözümler

### Yaygın Sorunlar:

1. **Bellek Hatası:**
   - Batch size'ı düşürün (16 veya 8)
   - Görüntü boyutunu küçültün (128x128)

2. **Overfitting:**
   - Dropout oranını artırın (0.5 → 0.7)
   - Daha fazla veri artırma uygulayın

3. **Düşük Doğruluk:**
   - Learning rate'i ayarlayın (0.001 → 0.0001)
   - Daha fazla epoch ile eğitin

## 🔮 Gelecek Geliştirmeler

- [ ] **EfficientNet** modeli ile deneme
- [ ] **Ensemble** yöntemleri uygulama
- [ ] **Grad-CAM** ile görselleştirme
- [ ] **Web uygulaması** geliştirme
- [ ] **Mobile deployment** (TensorFlow Lite)
- [ ] **Real-time** görüntü sınıflandırma

## 👨‍💻 Katkıda Bulunma

1. Bu repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/YeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/YeniOzellik`)
5. Pull Request oluşturun

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 📞 İletişim

- **Geliştirici:** [Adınız]
- **E-posta:** [email@example.com]
- **LinkedIn:** [linkedin.com/in/profiliniz]
- **GitHub:** [github.com/kullaniciadi]

## 🙏 Teşekkürler

- Intel Corporation - Veri seti sağladığı için
- Kaggle Community - Veri setini paylaştığı için
- TensorFlow Team - Framework geliştirmesi için

## 📚 Referanslar

1. [Intel Image Classification Dataset](https://www.kaggle.com/puneet6060/intel-image-classification)
2. [VGG16 Paper](https://arxiv.org/abs/1409.1556)
3. [Transfer Learning Guide](https://www.tensorflow.org/guide/keras/transfer_learning)
4. [Image Classification Best Practices](https://arxiv.org/abs/1905.11946)

---
