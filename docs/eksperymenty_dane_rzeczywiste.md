
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
| D          | 5000     | 250 (5%)             | ↳                        |
| E          | 5000     | 500 (10%)            | ↳                        |

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
| A          | 3000     | 0                    | 1000 normalnych + 100 anomalnych |
| B          | 3000     | 15 (0,5%)            | ↳                         |
| C          | 3000     | 30 (1%)              | ↳                         |
| D          | 3000     | 150 (5%)             | ↳                         |
| E          | 3000     | 300 (10%)            | ↳                         |


---

## 🔸 EmailSpam *(3578 normalne, 146 anomalne)*

*(Najmniejszy zbiór, dlatego mniejsze liczby)*

| Scenariusz | Normalne | Anomalne (oznaczone) | Test (stały)             |
|------------|----------|----------------------|--------------------------|
| A          | 2000     | 0                    | 1000 normalnych + 46 anomalnych |
| B          | 2000     | 10 (0,5%)            | ↳                         |
| C          | 2000     | 20 (1%)              | ↳                         |
| D          | 2000     | 50 (2,5%)            | ↳                         |
| E          | 2000     | 100 (5%)             | ↳                         |

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



## 🔸 NSL-KDD *(67343 normalne, 58630 anomalne)*

Test stały (jak w EX1): 2000 normalnych + 200 anomalnych

| Scenariusz | Normalne | Anomalne (oznaczone, tylko dla semi/supervised) | Anomalne (błędne)            |
|------------|----------|----------------------|-------------------------|
| A          | 5000     | 250                    | 25 |
| B          | 5000     | 250             | 50                        |
| C          | 5000     | 250               | 125                        |
| D          | 5000     | 250              | 250                        |

---

## 🔸 Credit Card Fraud *(284315 normalne, 492 anomalne)*

Test stały (jak w EX1): 2000 normalnych + 200 anomalnych

| Scenariusz | Normalne | Anomalne (oznaczone, tylko dla semi/supervised) | Anomalne (błędne)             |
|------------|----------|----------------------|--------------------------|
| A          | 5000     | 150                    | 15 |
| B          | 5000     | 150             | 30                         |
| C          | 5000     | 150               | 75                         |
| D          | 5000     | 150  | 142                         |


---

## 🔸 AnnThyroid *(6666 normalne, 534 anomalne)*

Test stały (jak w EX1): 1000 normalnych + 100 anomalnych


| Scenariusz | Normalne | Anomalne (oznaczone, tylko dla semi/supervised) | Anomalne (błędne)             |
|------------|----------|----------------------|--------------------------|
| A          | 3000     | 150                    | 15 |
| B          | 3000     | 150            | 30                         |
| C          | 3000     | 150              | 75                         |
| D          | 3000     | 150             | 150                         |


---

## 🔸 EmailSpam *(3578 normalne, 146 anomalne)*

Test stały (jak w EX1): 1000 normalnych + 46 anomalnych

| Scenariusz | Normalne | Anomalne (oznaczone, tylko dla semi/supervised) | Anomalne (błędne)             |
|------------|----------|----------------------|--------------------------|
| A          | 2000     | 50                    | 5 |
| B          | 2000     | 50             | 10                         |
| C          | 2000     | 50              | 25                         |
| D          | 2000     | 50             | 50                         |








# 🧪 **Eksperyment 3: Wpływ typu oraz trudności syntetycznych anomalii na skuteczność modeli deep learning**

## 🎯 Cel eksperymentu:

**Zbadanie:**

- jak różne typy anomalii (local, global, dependency, clustered) wpływają na jakość detekcji modeli supervised, semi-supervised oraz unsupervised,
- jak różne poziomy trudności generowanych anomalii wpływają na wyniki modeli.

## ✅ Modele wykorzystane w eksperymencie:

(Wszystkie)

- **Supervised:** FTTransformer  
- **Semi-supervised:** FEAWAD, DevNet  
- **Unsupervised:** AE, VAE, DeepSVDD, DAGMM, SO_GAAL, LUNAR, GANomaly  

---

## 🛠️ **Metoda generowania anomalii**:

Każdy zbiór danych (NSL-KDD, Credit Card, AnnThyroid, EmailSpam) poddajemy następującej procedurze:

### **Krok 1: Trenowanie modelu bazowego na danych normalnych**

- **Model bazowy:** Gaussian Mixture Model (GMM).
- Trenowany tylko na danych klasy „0” (normalnych).

---

### **Krok 2: Generowanie anomalii – 4 typy wg. ADBench**

Dla każdego typu anomalii wygenerujemy trzy poziomy trudności:

| Typ anomalii  | Opis i parametry bazowe                               | Poziomy trudności (parametry α) |
|---------------|--------------------------------------------------------|----------------------------------|
| **Local**     | GMM ze skalowaną kowariancją: `Σ' = αΣ`                | łatwy: α=2<br>średni: α=5<br>trudny: α=10 | 15
| **Global**    | Uniform(min-max), skalowane zakresy: `α × zakres`      | łatwy: α=1.1<br>średni: α=1.25<br>trudny: α=1.5 | 2
| **Dependency**| Vine Copula + KDE (zaburzenie zależności między cechami)| łatwy: 35% cech zaburzonych <br>średni: 70% cech zaburzonych<br>trudny: 100% cech zaburzonych |
| **Clustered** | GMM wokół średniej z większą odległością od normalnych | łatwy: α=2<br>średni: α=5<br>trudny: α=10 | 15


Trenowanie GMM/Vine Copula: 5000 (3500 dla email) normalnych próbek

Generowanie:
Trening -  5000 normalnych, 250 anomalii (tylko dla semi/supervised)
Test - 2000 normalnych, 200 anomalii

---

## 📊 **Analiza wyników (propozycja):**

Porównaj wyniki na dwóch płaszczyznach:

- **Typ anomalii**:  
  Które anomalie są najtrudniejsze dla modeli supervised/semi-supervised/unsupervised?

- **Poziom trudności**:  
  Jak poziom trudności generowanych anomalii wpływa na pogorszenie wyników?



## 📝 **Przykładowe pytania badawcze eksperymentu 3:**

- Czy modele supervised (np. FTTransformer) radzą sobie lepiej z określonym typem syntetycznych anomalii niż modele unsupervised (np. VAE)?
- Czy anomalie lokalne są generalnie trudniejsze dla modeli unsupervised?
- Jak silny jest wpływ wzrostu trudności (np. wzrost współczynnika α) na pogorszenie wyników poszczególnych modeli?

