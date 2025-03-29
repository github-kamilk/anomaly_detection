

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
| Deep learning for feature extraction            | **FTTransformer** (Sup, ADB), **ResNet** (Sup, ADB)                    |
| Generic normality feature learning              | **AE** (Unsup, PyOD), **VAE** (Unsup, PyOD), **GANomaly** (Semi, ADB)         |
| Distance-based measures                         | **REPEN** (Semi, ADB)                                      |
| One-class classification measures               | **DeepSVDD** (Unsup, PyOD), **DeepSAD** (Semi, ADB)                        |
| Clustering-based measures                       | **DAGMM** (Unsup, ADB), **SO_GAAL** (Unsup, PyOD)                                        |
| Ranking models                                  | **DevNet** (Semi, PyOD), **FEAWAD** (Semi, ADB)|

**PODSTAWA**: FTTransformer, AE, VAE, ~~REPEN~~, DeepSVDD, DAGMM, LUNAR
**ROZSZERZENIE**: DevNet, ~~ResNet~~, GANomaly, ~~DeepSAD~~, SO_GAAL, FEAWAD


Supervised: **FTTransformer**
Semi-Supervised (unlabeled to głównie normal + zaznaczone pojedyńcze anomalie): **FEAWAD**, **DevNet**? na razie nie działa na irysach
Unsupervised (uczone tylko na zbiorze normalnym): **AE**, **VAE**, **DeepSVDD**, **DAGMM**, **SO_GAAL**, **LUNAR**, **GANomaly**

Przetestować DeepSAD

# Wybór zbiorów danych 


- **Sieciowe**: NSL-KDD [https://github.com/jmnwong/NSL-KDD-Dataset] [https://web.archive.org/web/20150205070216/http://nsl.cs.unb.ca/NSL-KDD/] [https://www.researchgate.net/publication/283185453_Analysis_of_KDD_Dataset_Attributes_-_Class_wise_for_Intrusion_Detection]
- **Finansowe**: Credit Card Fraud (ADBench)
- **Medyczne**: AnnThyroid (ADBench)
- **Text**: EmailSpam (OpenAI embeddings from NLP ADBench)


----------------
AUC ROC
AUC PR/REC
macierz pomylek!  zwrócic uwagę jaki jest threshold
czas


# Plan 

- 16.03 Wybór zbiorów danych
- 21.03 Analiza zbiorów danych (podstawowe statystyki, redukcja wymiarowosci)
- 28.03 Uruchomienie treniningu do Eksperymentu 1
- 30.03 Przygotowanie zbiorów syntetycznych
- 04.04 Uruchomienie treniningu do Eksperymentu 2





# Eksperyment 1

-Opisać dokładnie jakie podejście było w ADB, z podziałem na unsup, sup, semisup

"Thus, we use 70% data for training and the remaining 30%
as the test set. We use stratified sampling to keep the anomaly ratio consistent. We repeat each
experiment 3 times and report the averag"

"Hyperparameter Settings. For all the algorithms in ADBench, we use their default hyperparameter
(HP) settings in the original paper for a fair comparis"


# Eksperyment 2

"NLP datasets are mainly considered for evaluating algorithm performance on the public datasets and are not included in the
experiments of different types of anomalies and algorithm robustness, since such high-dimensional
data could make it hard to generate synthetic anomalies, or introduce too much noise in input data."



