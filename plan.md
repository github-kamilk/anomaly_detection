

| Algorytm                                                            | adbench | pyOD | AD-LLM | NLP-ADBench |
|----------------------------------------------------------------------|:-------:|:----:|:------:|:-----------:|
| Deep Support Vector Data Description (**DeepSVDD**)                  | ✔️      | ✔️   | ✔️     | ✔️          |
| Deep Autoencoding Gaussian Mixture Model (**DAGMM**)                 | ✔️      |      |        |             |
| Semi-Supervised Anomaly Detection via Adversarial Training (**GANomaly**) | ✔️      |      |        |             |
| Deep Semi-supervised Anomaly Detection (**DeepSAD**)                 | ✔️      |      |        |             |
| REPresentations for a random nEarest Neighbor (**REPEN**)            | ✔️      |      |        |             |
| Deviation Networks (**DevNet**)                                      | ✔️      | ✔️   |        |             |
| Pairwise Relation prediction-based ordinal regression Network (**PReNet**) | ✔️      |      |        |             |
| Feature Encoding With Autoencoders for Weakly Supervised AD (**FEAWAD**) | ✔️      |      |        |             |
| Residual Networks (**ResNet**)                                       | ✔️      |      |        |             |
| Feature Tokenizer + Transformer (**FTTransformer**)                  | ✔️      |      |        |             |
| AutoEncoder (**AE**)                                                 |         | ✔️   | ✔️     | ✔️          |
| Variational AutoEncoder (**VAE**)                                    |         | ✔️   | ✔️     | ✔️          |
| Beta-VAE                                                             |         | ✔️   |        |             |
| Single-Objective Generative Adversarial Active Learning (**SO_GAAL**) |         | ✔️   | ✔️     | ✔️          |
| Multiple-Objective Generative Adversarial Active Learning (**MO_GAAL**) |         | ✔️   |        |             |
| AnoGAN                                                               |         | ✔️   |        |             |
| Adversarially learned anomaly detection (**ALAD**)                   |         | ✔️   |        |             |
| Autoencoder-based One-class SVM (**AE1SVM**)                         |         | ✔️   |        |             |
| LUNAR (Graph Neural Networks)                                        |         | ✔️   |        | ✔️          |
| Context Vector Data Description (**CVDD**)                           |         |      | ✔️     |             |
| Detecting Anomalies in Text via Self-Supervision (**DATE**)          |         |      | ✔️     |             |

---



| Grupa podejść                                   | Wybrane algorytmy                                |
|-------------------------------------------------|--------------------------------------------------|
| Deep learning for feature extraction            | **FTTransformer** (Sup), **ResNet** (Sup)                    |
| Generic normality feature learning              | **AE** (Unsup), **VAE** (Unsup), **GANomaly** (Semi)         |
| Distance-based measures                         | **REPEN** (Semi)                                      |
| One-class classification measures               | **DeepSVDD** (Unsup), **DeepSAD** (Semi)                        |
| Clustering-based measures                       | **DAGMM** (Unsup), **SO_GAAL** (Unsup)                                        |
| Ranking models                                  | **DevNet** (Semi), **FEAWAD** (Semi)|

**PODSTAWA**: FTTransformer, AE, VAE, REPEN, DeepSVDD, DAGMM

**ROZSZERZENIE**: DevNet, ResNet, GANomaly, DeepSAD, SO_GAAL, FEAWAD


# Wybór zbiorów danych 


- **Sieciowe**: NSL-KDD [https://github.com/jmnwong/NSL-KDD-Dataset] [https://web.archive.org/web/20150205070216/http://nsl.cs.unb.ca/NSL-KDD/] [https://www.researchgate.net/publication/283185453_Analysis_of_KDD_Dataset_Attributes_-_Class_wise_for_Intrusion_Detection]
- **Finansowe**: Credit Card Fraud (ADBench)
- **Medyczne**: AnnThyroid (ADBench)
- **Text**: EmailSpam (OpenAI embeddings from NLP ADBench)


----------------

# Plan 

- 16.03 Wybór zbiorów danych
- 21.03 Analiza zbiorów danych (podstawowe statystyki, redukcja wymiarowosci)
- 28.03 Uruchomienie treniningu do Eksperymentu 1
- 30.03 Przygotowanie zbiorów syntetycznych
- 04.04 Uruchomienie treniningu do Eksperymentu 2
