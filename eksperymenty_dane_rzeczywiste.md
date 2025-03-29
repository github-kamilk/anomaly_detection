
# 🧪 **Eksperyment 1:** Porównanie metod **supervised** i **semi-supervised** w zależności od liczby oznaczonych anomalii w zbiorze treningowym

## 🎯 **Cel eksperymentu:**

- Zbadanie, jak liczba poprawnie oznaczonych przykładów anomalnych w treningu wpływa na jakość detekcji anomalii metod **supervised** oraz **semi-supervised**.

## ✅ **Modele objęte eksperymentem:**

- **Supervised:**  
  - FTTransformer

- **Semi-supervised:**  
  - FEAWAD, DevNet

## 🛠️ **Założenia eksperymentu:**

- **Stała liczba przykładów normalnych** w zbiorze treningowym (np. **5000**).
- Zmienna liczba oznaczonych przykładów anomalnych w zbiorze treningowym np:

| Scenariusz | Liczba anomalii w treningu |
|------------|-----------------------------|
| A          | 0  (może tutaj dać jako baseline unsupervised?)                         |
| B          | 25 (0,5%)                   |
| C          | 50 (1%)                     |
| D          | 250 (5%)                    |
| E          | 500 (10%)                   |

- **Stały zbiór testowy** (np. **2000 normalnych + 200 anomalnych**, może 200+200?).

## 🔍 **Przykładowe pytania badawcze:**

- Czy modele supervised (FTTransformer) szybko osiągają wysoką skuteczność już przy niskim procencie anomalii?
- Czy modele semi-supervised (DevNet, FEAWAD) potrzebują większej liczby anomalii, aby zbliżyć się jakością do supervised?



## 🔸 NSL-KDD *(67343 normalne, 58630 anomalne)*

| Scenariusz | Normalne | Anomalne (oznaczone) | Test (stały)            |
|------------|----------|----------------------|-------------------------|
| A          | 5000     | 0                    | 2000 normalnych + 200 anomalnych |
| B          | 5000     | 25 (0,5%)            | ↳                        |
| C          | 5000     | 50 (1%)              | ↳                        |
| D          | 5000     | 150 (5%)             | ↳                        |
| E          | 5000     | 300 (6%)            | ↳                        |

---

## 🔸 Credit Card Fraud *(284315 normalne, 492 anomalne)*

| Scenariusz | Normalne | Anomalne (oznaczone) | Test (stały)             |
|------------|----------|----------------------|--------------------------|
| A          | 5000     | 0                    | 2000 normalnych + 200 anomalnych |
| B          | 5000     | 25 (0,5%)            | ↳                         |
| C          | 5000     | 50 (1%)              | ↳                         |
| D          | 5000     | 150 (3%)<br>*(ograniczenie ze względu na małą liczbę anomalii)* | ↳                         |
| E          | 5000     | 292 (6%)<br>*(ograniczenie)* | ↳                         |


---

## 🔸 AnnThyroid *(6666 normalne, 534 anomalne)*

| Scenariusz | Normalne | Anomalne (oznaczone) | Test (stały)             |
|------------|----------|----------------------|--------------------------|
| A          | 5000     | 0                    | 1000 normalnych + 100 anomalnych |
| B          | 5000     | 25 (0,5%)            | ↳                         |
| C          | 5000     | 50 (1%)              | ↳                         |
| D          | 5000     | 150 (5%)             | ↳                         |
| E          | 5000     | 300 (6%)<br>*(max.)*| ↳                         |


---

## 🔸 EmailSpam *(3578 normalne, 146 anomalne)*

*(Najmniejszy zbiór, dlatego mniejsze liczby)*

| Scenariusz | Normalne | Anomalne (oznaczone) | Test (stały)             |
|------------|----------|----------------------|--------------------------|
| A          | 2000     | 0                    | 1000 normalnych + 46 anomalnych |
| B          | 2000     | 10 (0,5%)            | ↳                         |
| C          | 2000     | 20 (1%)              | ↳                         |
| D          | 2000     | 50 (2,5%)<br>*(ograniczenie ze względu na liczbę)*| ↳                         |
| E          | 2000     | 100 (5%)<br>*(max.)* | ↳                         |

*(Pozostaje ~46 anomalnych do testu.)*


---

# 🧪 **Eksperyment 2:** Analiza odporności modeli **supervised**, **semi-supervised** oraz **unsupervised** na błędne oznaczenia anomalii jako dane normalne

## 🎯 **Cel eksperymentu:**

- Zbadanie odporności modeli supervised, semi-supervised oraz unsupervised na sytuację, gdy anomalie trafiają do zbioru treningowego błędnie oznaczone jako dane normalne (tzw. contamination lub noisy labels).

## ✅ **Modele objęte eksperymentem:**

- **Supervised:**  
  - FTTransformer

- **Semi-supervised:**  
  - FEAWAD, DevNet

- **Unsupervised:**  
  - AE, VAE, DeepSVDD, DAGMM, SO_GAAL, LUNAR, GANomaly

## 🛠️ **Założenia eksperymentu:**

- **Stała liczba przykładów normalnych** w treningu (np. **5000**).
- Wprowadzenie **błędnie oznaczonych anomalii (oznaczonych jako normalne)** w różnych ilościach:

| Scenariusz | Liczba błędnie oznaczonych anomalii (oznaczone jako normalne) |
|------------|----------------------------------------------------------------|
| A          | 0 (czyste dane)                                                |
| B          | 25 (0,5%)                                                      |
| C          | 50 (1%)                                                        |
| D          | 250 (5%)                                                       |
| E          | 500 (10%)                                                      |


- Na podstawie Eksperymentu 1 **ustalony znany % anomalii** dla modeli supervised i semi-supervised dla wszystkich scenariuszy
- **Stały zbiór testowy** (identyczny jak w Eksperymencie 1).

## 🔍 **Pytania badawcze:**

- Jak odporne są poszczególne typy modeli (supervised, semi-supervised, unsupervised) na błędne oznaczenia danych?
- Która grupa modeli jest najbardziej odporna na sytuację, w której zbiór treningowy zawiera anomalie błędnie oznaczone jako dane normalne?
- Czy unsupervised modele radzą sobie lepiej w sytuacji błędnych oznaczeń niż supervised lub semi-supervised?
