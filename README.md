# Cats vs Dogs — Deep Learning Project 

## 🎯 Proje Amacı
Bu projede **Cats vs Dogs** veri seti kullanılarak kedi ve köpek görüntülerini sınıflandıran bir derin öğrenme modeli geliştirilmiştir.  
Amaç, sıfırdan basit bir CNN ile başlayıp, **Transfer Learning (MobileNetV2)** kullanarak performansı karşılaştırmaktır.  

## 🛠️ Kullanılan Yöntemler
- Veri önişleme 
- Data Augmentation (flip, rotation, zoom)
- CNN tabanlı Baseline model
- Transfer Learning (MobileNetV2, fine-tuning)
- Hiperparametre denemeleri (learning rate, dropout)
- Grad-CAM görselleştirme 

## 📎 Kaggle Notebook
👉 [Kaggle Notebook Linkim]((https://www.kaggle.com/code/yiitdoukanylmaz/proje))



 Özet
- **Veriseti:** [`tongpython/cat-and-dog`](https://www.kaggle.com/datasets/tongpython/cat-and-dog) (Kaggle, hazır train/test klasörleri)
- **Modeller:**
  - *Baseline CNN:* Sıfırdan küçük bir konvolüsyonel ağ
  - *MobileNetV2 Transfer Learning:* 
- **Sonuçlar :**
  - Baseline CNN → **Accuracy ≈ %66**, ROC‑AUC ≈ 0.72, Macro‑F1 ≈ 0.65
  - Transfer (MobileNetV2) → **Accuracy ≈ %99.0**, ROC‑AUC ≈ **0.9990**, AP ≈ 0.9986, **Macro‑F1 ≈ 0.996**

> Not: Sonuçlar benim koşumda elde edilmiştir. GPU, epoch sayısı, augmentation vb. değişirse metrikler farklılaşabilir.

---

##  İçerik
```
.
├─ notebooks/
│  ├─ cats_vs_dogs_kaggle_FIXED.ipynb   
│  └─ cats_vs_dogs_kaggle.ipynb         
├─ README.md
├─ requirements.txt                     
└─ .gitignore
```


```
# Python
__pycache__/
*.pyc

# Jupyter
.ipynb_checkpoints/

# Models & weights
*.keras
*.h5
*.pth
*.tflite

# Kaggle/Colab artifacts
working/
kaggle-working/
/kaggle/working/
/content/

# Data
data/
datasets/
/kaggle/input/
```

---

##  Hızlı Başlangıç (Kaggle)
1. Kaggle → **New Notebook** → **Upload Notebook** → `notebooks/cats_vs_dogs_kaggle_FIXED.ipynb`’yi yükle.
2. Sağ panel → **Add Data** → `tongpython/cat-and-dog` ekle.
3. **Settings → Accelerator: GPU** seç.
4. Hücreleri sırayla çalıştır:  
   - 1–3: Import + Dataset yolu + Pipeline  
   - 4–5: Baseline CNN eğit & değerlendir  
   - 6–9: MobileNetV2 eğit (top layers) → fine‑tune → test değerlendirme → modeli kaydet

> Notebook, `image_dataset_from_directory` ile `training_set/test_set` klasörlerinden otomatik veri yükler.

---

##  Lokal Çalıştırma (Opsiyonel)
```bash
# (1) Ortamı kur
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install -r requirements.txt   # yoksa: pip install tensorflow matplotlib scikit-learn pillow numpy

# (2) Veriyi indir & aç (zip'i data/ altına çıkar)
# training_set/ ve test_set/ klasörleri olacak şekilde düzenle
# (3) Notebook'u aç
jupyter lab  # veya jupyter notebook
```
Notebook içindeki `base_dir`’i lokal klasörüne göre değiştir:
```python
base_dir = Path("./data/cat-and-dog")
train_root = base_dir / "training_set" / "training_set"
test_root  = base_dir / "test_set" / "test_set"
```

---

##  Eğitim
- **Baseline CNN**: küçük bir mimari, hızlı deneme için.
- **Transfer Learning** (MobileNetV2):
  1) Base donuk (frozen), üst katmanları eğit.  
  2) Base’in son ~%30’unu aç, **düşük LR** ile fine‑tune.

Not: `IMG_SIZE`, `BATCH_SIZE`, epoch sayısı ve augmentation oranlarını değiştirerek performansı inceleyebilirsiniz.

---

##  Değerlendirme
Notebook aşağıdaki metrikleri üretir:
- Accuracy, Precision, Recall, **Macro‑F1**
- **Confusion Matrix**
- **ROC** ve **PR** eğrileri (ROC‑AUC, AP)

Örnek baseline → transfer karşılaştırması (benim koşum):
- Baseline: Acc~%66, ROC‑AUC~0.72, Macro‑F1~0.65
- Transfer: **Acc~%99**, **ROC‑AUC~0.999**, **Macro‑F1~0.996**

---

##  Inference (Tek görsel tahmini)
Aşağıdaki yardımcı fonksiyonu notebook sonunda kullanabilirsiniz:
```python
from PIL import Image
import numpy as np
from tensorflow import keras

IMG_SIZE = 224
class_names = ["cats", "dogs"]  # image_dataset_from_directory sırasına göre

def predict_image(model, image_path):
    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32")/255.0
    arr = np.expand_dims(arr, 0)
    prob = model.predict(arr, verbose=0).squeeze()
    pred = int(prob >= 0.5)
    return class_names[pred], float(prob)

# usage:
# model = keras.models.load_model("cats_vs_dogs_transfer.keras")
# label, prob = predict_image(model, "example.jpg")
# print(label, prob)
```

