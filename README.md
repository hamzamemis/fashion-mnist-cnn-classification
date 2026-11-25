# Fashion-MNIST Veri Seti ile CNN Tabanlı Görüntü Sınıflandırma ve Web Uygulaması

Bu depo, Derin Öğrenme dersi **Computer Vision Ödevi** kapsamında hazırlanmıştır.

* **Öğrenci Numarası:** `23120205011`

---

## 📋 Proje Raporu Şablonu

| Alan | Açıklama / Öğrenci Yanıtı |
|---|---|
| **Project Title** | Fashion-MNIST Veri Seti ile CNN Tabanlı Kıyafet Sınıflandırma ve Gradio Web Uygulaması |
| **Problem / Task** | Görüntü Sınıflandırma (Multi-Class Image Classification — 10 Sınıf) |
| **Dataset** | Fashion-MNIST (70.000 adet 28×28 gri seviye kıyafet/aksesuar görüntüsü; 50.000 Train / 10.000 Validation / 10.000 Test) |
| **Dataset URL** | https://github.com/zalandoresearch/fashion-mnist |
| **Model Architecture** | Özel Tasarlanmış CNN (`Conv2d(1→32, 3×3)` → `ReLU` → `MaxPool2d(2×2)` → `Conv2d(32→64, 3×3)` → `ReLU` → `MaxPool2d(2×2)` → `Flatten` → `Linear(3136→128)` → `ReLU` → `Linear(128→10)`) |
| **Input Size** | `1 × 28 × 28` (Gri seviye, tek kanallı normalize görüntü) |
| **Hyperparameters** | Learning Rate: `0.001`, Batch Size: `128`, Epoch: `6`, Seed: `42` |
| **Optimizer** | Adam (`lr=0.001`) |
| **Activation / Output Function** | Gizli katmanlarda **ReLU**; çıkışta **Softmax** (`CrossEntropyLoss` içinde Log-Softmax) |
| **Loss Function** | Cross-Entropy Loss (`nn.CrossEntropyLoss`) |
| **Evaluation Metrics** | Accuracy, Precision (Macro), Recall (Macro), F1-Score (Macro), Confusion Matrix |
| **Training Environment** | Google Colab (Python 3, PyTorch, CPU — Yüksek RAM) |
| **Best Performance** | Validation Acc: **%91.66** (Epoch 6, Val Loss: 0.2285) \| Test Acc: **%91.00**, Precision: **0.9105**, Recall: **0.9100**, F1-Score: **0.9089** \| Eğitim Süresi: **545.7 sn** |
| **Inference Time** | Tek görüntü ortalama: **0.737 ms** (10.000 test görüntüsü toplam: **5.151 sn**; gerçek fotoğraflarda ön işleme dâhil ~**1.44 ms**) |
| **Deployment Technology** | Gradio Web Arayüzü (`gradio.Interface`, canlı paylaşım linki ile) |
| **Repository** | *(GitHub depo bağlantınızı buraya ekleyin)* |

---

## 📁 Proje Dizin Yapısı

```text
├── fashion_mnist_cnn_odev.ipynb     # Eğitim, test, 5 gerçek fotoğraf analizi ve Gradio uygulamasını içeren ana notebook
├── my_photos/                       # Telefonla çekilen 5 gerçek kıyafet/çanta fotoğrafı (foto1.jpg - foto5.jpg)
├── results/                         # Eğitim/doğrulama eğrileri, confusion matrix, gerçek fotoğraf tahmin görseli ve results.md
└── README.md                        # Proje raporu ve dokümantasyon
```

---

## 🔬 Model Ayarları ve Seçim Gerekçeleri

1. **Veri Ayrımı:** 60.000 adet Fashion-MNIST eğitim verisi **50.000 Train** ve **10.000 Validation** olarak kesişmeyecek biçimde ayrılmıştır. Resmi **10.000 Test** kümesi eğitim ve model geliştirme sürecinde kesinlikle kullanılmamış, yalnızca final değerlendirmede kullanılmıştır.
2. **CNN Mimarisi:** İki evrişim katmanı (32 ve 64 filtre, 3×3 çekirdek) ile görüntülerdeki kenar, yaka, kol ve siluet gibi uzamsal (spatial) özellikler öğrenilmiş; `MaxPool2d` ile boyut indirgenerek öteleme değişmezliği güçlendirilmiştir.
3. **Aktivasyon & Kayıp Fonksiyonu:** Ara katmanlarda gradyan sönmesini önleyen **ReLU**; 10 sınıflı olasılık dağılımı optimizasyonu için **Cross-Entropy Loss** tercih edilmiştir.
4. **Optimizer:** Adaptif öğrenme oranları sayesinde hızlı ve kararlı yakınsama sağladığı için **Adam (`lr=0.001`)** kullanılmıştır.

---

## 📊 Performans Değerlendirmesi (Test Kümesi — 10.000 Örnek)

| Metrik | Değer |
|---|---|
| **Test Accuracy** | **%91.00** (`0.9100`) |
| **Precision (Macro)** | `0.9105` |
| **Recall (Macro)** | `0.9100` |
| **F1-Score (Macro)** | `0.9089` |
| **Eğitim Süresi (6 Epoch, CPU)** | `545.7 sn` |
| **Toplam Test Süresi (10.000 örnek)** | `5.151 sn` |
| **Tek Görüntü Tahmin Süresi (Ortalama)** | `0.737 ms` |

### Sınıf Bazlı Özet Bulgular
* **En yüksek başarı elde edilen sınıflar:** Pantolon (F1: `0.9875`), Çanta (F1: `0.9809`), Bot (F1: `0.9682`), Sandalet (F1: `0.9669`).
* **En çok karışan sınıflar:** Karmaşıklık matrisinde en düşük F1 skoru **Gömlek (`0.7413`)** sınıfında görülmüştür. Gömlek örnekleri en çok **Tişört (104)**, **Kazak (104)** ve **Mont (85)** sınıflarıyla karışmaktadır; bunun nedeni 28×28 gri seviye çözünürlükte üst giyim siluetlerinin birbirine çok benzemesidir.

---

## 📱 Gerçek Görüntülerle Test ve Değerlendirme (5 Telefon Fotoğrafı)

Kendi telefonumla çektiğim 5 farklı giyim/aksesuar fotoğrafı gri tonlama, arka plan temizleme, kare tamamlama ve 28×28 boyutlandırma adımlarından geçirilerek eğitilen CNN modeline verilmiştir:

| Fotoğraf | Gerçek Sınıf | Model Tahmini | Güven Skoru | Tahmin Süresi | Durum |
|---|---|---|---|---|---|
| `foto1.jpg` | Tişört | Gömlek | %63.5 | 1.49 ms | Yanlış ✗ |
| `foto2.jpg` | Pantolon | **Pantolon** | **%99.3** | 1.47 ms | **Doğru ✓** |
| `foto3.jpg` | Kazak | Mont | %55.8 | 1.42 ms | Yanlış ✗ |
| `foto4.jpg` | Mont | Çanta | %96.3 | 1.46 ms | Yanlış ✗ |
| `foto5.jpg` | Çanta | **Çanta** | **%98.1** | 1.38 ms | **Doğru ✓** |

* **Gerçek Fotoğraf Doğruluğu:** `2/5 (%40.0)` *(Resmi Fashion-MNIST Test Accuracy: `%91.00`)*

### Başarılı ve Başarısız Durumların Yorumlanması
* **Başarılı Durumlar (`Pantolon` %99.3, `Çanta` %98.1):** Pantolonun iki bacaklı belirgin geometrisi ve sırt çantasının üst saplı oval formu 28×28 boyutuna indirgendiğinde bile ayırt edici siluetini korumuştur. Arka planla kontrastı yüksek ve şekli özgün nesnelerde model çok yüksek güven skoruyla doğru tahmin üretmiştir.
* **Başarısız Durumlar ve Sebepleri (*Domain Shift*):**
  1. **Tişört → Gömlek (%63.5):** Test setinin karmaşıklık matrisinde de en çok karışan ikili Tişört ve Gömlek'tir. Düşük çözünürlükte (28×28) yaka ve düğme detayları kaybolduğundan model orta seviye bir güvenle (%63.5) Gömlek tahmininde bulunmuştur.
  2. **Kazak → Mont (%55.8):** Kapüşonlu sweatshirt'ün üstteki kapüşon çıkıntısı ve kalın kolları 28×28'de kalın dış giyim/mont formuna benzemiş; model de %55.8 gibi düşük bir güvenle iki sınıf arasında kararsız kalmıştır.
  3. **Mont → Çanta (%96.3):** 28×28 işlenmiş görüntü incelendiğinde, montun kollarının gövdeye bitişik durması nedeniyle dış hattın tek parça yuvarlak/kare bir bloğa dönüştüğü ve ortadaki parlak fermuar hattının çanta detayı gibi algılandığı görülmüştür.
* **Genel Çıkarım:** Fashion-MNIST görselleri stüdyo ışığında, tam beyaz/siyah arka plandan temizlenmiş katalog fotoğraflarıdır. Gerçek telefon fotoğraflarındaki çarşaf kırışıklıkları, gölgeler, kumaş katlanmaları ve 28×28'e küçültme kaynaklı bilgi kaybı (*distribution shift*) nedeniyle gerçek dünya başarısı test kümesine göre belirgin biçimde düşmektedir. Veri artırma (*data augmentation*), daha yüksek giriş çözünürlüğü (ör. 224×224) ve transfer öğrenme (ResNet vb.) bu farkı ciddi oranda azaltacaktır.
