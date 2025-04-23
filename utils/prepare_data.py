import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from copulas.multivariate import VineCopula
from copulas.univariate import GaussianKDE
import pandas as pd


def get_data(dataset):
    datasets_paths = {
        'NSL-KDD': 'nsl_kdd.npz',
        'CreditCard': 'fraud.npz',
        'AnnThyroid': 'annthyroid.npz',
        'EmailSpam': 'email.npz',
        'EmailSpam-bert': 'email_bert.npz'
    }
    
    data = np.load(f"datasets/preprocessed_datasets/{datasets_paths[dataset]}")
    
    X = data['X']
    y = data['y']
    
    return X, y

def prepare_data_e1(dataset, seed=42, save_dataset=True):
    X, y = get_data(dataset)
    
    np.random.seed(seed)
    
    # Scenarios definition for each dataset
    scenarios = {
        'NSL-KDD': {'normal':5000, 'anomalies':[0,25,50,250,500], 'test_n':2000, 'test_a':200},
        'CreditCard': {'normal':5000, 'anomalies':[0,25,50,150,292], 'test_n':2000, 'test_a':200},
        'AnnThyroid': {'normal':3000, 'anomalies':[0,15,30,150,300], 'test_n':1000, 'test_a':100},
        'EmailSpam': {'normal':2000, 'anomalies':[0,10,20,50,100], 'test_n':1000, 'test_a':46},
        'EmailSpam-bert': {'normal':2000, 'anomalies':[0,10,20,50,100], 'test_n':1000, 'test_a':46},
    }
    
    params = scenarios[dataset]

    X_normal = X[y == 0]
    X_anomalies = X[y == 1]

    # First, create a fixed test set
    X_n_train, X_n_test = train_test_split(X_normal, test_size=params['test_n'], random_state=seed)
    X_a_train, X_a_test = train_test_split(X_anomalies, test_size=params['test_a'], random_state=seed)

    X_test = np.vstack((X_n_test, X_a_test))
    y_test = np.hstack((np.zeros(len(X_n_test)), np.ones(len(X_a_test))))

    # Prepare normal samples once for comparability
    X_norm_sample = X_n_train[np.random.choice(len(X_n_train), params['normal'], replace=False)]

    # Prepare anomaly indices for incremental inclusion
    max_anomalies = max(params['anomalies'])
    anom_indices = np.random.choice(len(X_a_train), max_anomalies, replace=False)

    # Prepare each training scenario
    scenario_labels = ['A', 'B', 'C', 'D', 'E']

    data_dict = {'X_test': X_test, 'y_test': y_test}

    for sc_label, anomaly_count in zip(scenario_labels, params['anomalies']):

        if anomaly_count > 0:
            X_anom_sample = X_a_train[anom_indices[:anomaly_count]]
            X_train = np.vstack((X_norm_sample, X_anom_sample))
            y_train = np.hstack((np.zeros(len(X_norm_sample)), np.ones(len(X_anom_sample))))
        else:
            X_train = X_norm_sample
            y_train = np.zeros(len(X_norm_sample))

        # Shuffle training set
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train, y_train = X_train[indices], y_train[indices]

        # Save datasets into dictionary
        data_dict[f'X_train_{sc_label}'] = X_train
        data_dict[f'y_train_{sc_label}'] = y_train

    # Save all data to a single npz file
    if save_dataset:
        np.savez(f'datasets/E1/E1_{dataset}_{seed}_data.npz', **data_dict)

    return data_dict

def prepare_data_e2(dataset, seed=42, save_dataset=True):
    X, y = get_data(dataset)
    
    np.random.seed(seed)
    
    # Scenarios definition for each dataset
    scenarios = {
        'NSL-KDD': {'normal':5000, 'anomalies':250, 'anomalies_misslabeled':[25,50,125,250], 'test_n':2000, 'test_a':200},
        'CreditCard': {'normal':5000, 'anomalies':150, 'anomalies_misslabeled':[15,30,75,142], 'test_n':2000, 'test_a':200},
        'AnnThyroid': {'normal':3000, 'anomalies':150, 'anomalies_misslabeled':[15,30,75,150], 'test_n':1000, 'test_a':100},
        'EmailSpam-bert': {'normal':2000, 'anomalies':50, 'anomalies_misslabeled':[5,10,25,50], 'test_n':1000, 'test_a':46},
    }
    
    params = scenarios[dataset]

    X_normal = X[y == 0]
    X_anomalies = X[y == 1]

    # First, create a fixed test set
    X_n_train, X_n_test = train_test_split(X_normal, test_size=params['test_n'], random_state=seed)
    X_a_train, X_a_test = train_test_split(X_anomalies, test_size=params['test_a'], random_state=seed)

    X_test = np.vstack((X_n_test, X_a_test))
    y_test = np.hstack((np.zeros(len(X_n_test)), np.ones(len(X_a_test))))

    # Prepare normal samples once for comparability
    X_norm_sample = X_n_train[np.random.choice(len(X_n_train), params['normal'], replace=False)]

    # Prepare anomaly indices for incremental inclusion
    max_anomalies_misslabeled = max(params['anomalies_misslabeled'])
    anom_indices = np.random.choice(len(X_a_train), max_anomalies_misslabeled+params['anomalies'], replace=False)

    # Prepare each training scenario
    scenario_labels = ['A', 'B', 'C', 'D']

    data_dict = {'X_test': X_test, 'y_test': y_test}

    for sc_label, anomaly_misslabeled_count in zip(scenario_labels, params['anomalies_misslabeled']):
        total_anomaly_count = anomaly_misslabeled_count + params['anomalies']
        X_anom_sample = X_a_train[anom_indices[:total_anomaly_count]]
        X_train = np.vstack((X_norm_sample, X_anom_sample))
        y_train = np.hstack((np.zeros(len(X_norm_sample)), np.ones(params['anomalies']), np.zeros(anomaly_misslabeled_count)))

        # Shuffle training set
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train, y_train = X_train[indices], y_train[indices]

        # Save datasets into dictionary
        data_dict[f'X_train_{sc_label}'] = X_train
        data_dict[f'y_train_{sc_label}'] = y_train

    # Save all data to a single npz file
    if save_dataset:
        np.savez(f'datasets/E2/E2_{dataset}_{seed}_data.npz', **data_dict)

    return data_dict

def prepare_data_e3(dataset, anomaly_type, seed=42, save_dataset=True):
    X, y = get_data(dataset)
    
    # Używamy tylko normalnych danych jako podstawy
    X_normal = X[y == 0]
    
    np.random.seed(seed)
    
    # Parametry dla każdego zbioru danych
    scenarios = {
        'NSL-KDD': {'train_n': 5000, 'train_a': 250, 'test_n': 2000, 'test_a': 200},
        'CreditCard': {'train_n': 5000, 'train_a': 250, 'test_n': 2000, 'test_a': 200},
        'AnnThyroid': {'train_n': 5000, 'train_a': 250, 'test_n': 2000, 'test_a': 200},
        'EmailSpam': {'train_n': 3500, 'train_a': 250, 'test_n': 2000, 'test_a': 200},
        'EmailSpam-bert': {'train_n': 3500, 'train_a': 250, 'test_n': 2000, 'test_a': 200}
    }
    
    # Poziomy trudności dla różnych typów anomalii
    difficulty_params = {
        'local': [2, 5, 10, 15],  # Parametr alpha dla lokalnych anomalii
        'global': [1.1, 1.25, 1.5, 2],  # Parametr alpha dla globalnych anomalii
        'clustered': [2, 5, 10, 15],  # Parametr alpha dla klastrowych anomalii
        'dependency': [0.35, 0.7, 1.0]  # Procent zaburzonych cech
    }
    
    # Pobierz parametry dla danego zbioru danych
    params = scenarios[dataset]
    difficulty_levels = difficulty_params[anomaly_type]
    
        
    # Podstawowe wartości
    train_normal_samples = params['train_n']
    train_anom_samples = params['train_a']
    test_normal_samples = params['test_n']
    test_anom_samples = params['test_a']
    
    # Indeksy dla normalnych danych do podziału na zbiór treningowy i testowy
    normal_indices = np.random.permutation(len(X_normal))
    train_indices = normal_indices[:train_normal_samples]
    test_indices = normal_indices[train_normal_samples:train_normal_samples+test_normal_samples]
    
    X_train_normal = X_normal[train_indices]
    # X_test_normal = X_normal[test_indices]
    
    data_dict = {}
    
    # Wygeneruj dane dla każdego poziomu trudności
    scenario_labels = ['small', 'medium', 'large']
    
    for i, (sc_label, difficulty_param) in enumerate(zip(scenario_labels, difficulty_levels)):
        # Generowanie syntetycznych normalnych danych i anomalii
        if anomaly_type in ['local', 'clustered', 'global']:
            # Wybierz najlepszą liczbę komponentów na podstawie BIC
            metric_list = []
            n_components_list = list(np.arange(1, 10))
            
            for n_components in n_components_list:
                gm = GaussianMixture(n_components=n_components, random_state=seed).fit(X_train_normal)
                metric_list.append(gm.bic(X_train_normal))
                
            best_n_components = n_components_list[np.argmin(metric_list)]
            
            # Dopasuj model dla najlepszej liczby komponentów
            gm = GaussianMixture(n_components=best_n_components, random_state=seed).fit(X_train_normal)
            
            # Generuj syntetyczne normalne dane dla zbioru treningowego
            X_synthetic_train_normal = gm.sample(train_normal_samples)[0]
            X_synthetic_test_normal = gm.sample(test_anom_samples)[0]
            
            # Generuj syntetyczne anomalie
            if anomaly_type == 'local':
                # Lokalne anomalie - skalowana kowariancja
                gm_anomaly = GaussianMixture(n_components=best_n_components, random_state=seed).fit(X_train_normal)
                gm_anomaly.covariances_ = difficulty_param * gm_anomaly.covariances_
                X_synthetic_train_anomalies = gm_anomaly.sample(train_anom_samples)[0]
                X_synthetic_test_anomalies = gm_anomaly.sample(test_anom_samples)[0]
                
            elif anomaly_type == 'clustered':
                # Klastrowe anomalie - skalowana średnia
                gm_anomaly = GaussianMixture(n_components=best_n_components, random_state=seed).fit(X_train_normal)
                gm_anomaly.means_ = difficulty_param * gm_anomaly.means_
                X_synthetic_train_anomalies = gm_anomaly.sample(train_anom_samples)[0]
                X_synthetic_test_anomalies = gm_anomaly.sample(test_anom_samples)[0]
                
            elif anomaly_type == 'global':
                # Globalne anomalie - równomierny rozkład poza zakresem normalnych
                X_synthetic_train_anomalies = np.zeros((train_anom_samples, X_train_normal.shape[1]))
                X_synthetic_test_anomalies = np.zeros((test_anom_samples, X_train_normal.shape[1]))
                
                for j in range(X_train_normal.shape[1]):
                    low = np.min(X_train_normal[:, j]) * difficulty_param
                    high = np.max(X_train_normal[:, j]) * difficulty_param
                    
                    X_synthetic_train_anomalies[:, j] = np.random.uniform(low=low, high=high, size=train_anom_samples)
                    X_synthetic_test_anomalies[:, j] = np.random.uniform(low=low, high=high, size=test_anom_samples)
                
        elif anomaly_type == 'dependency':
            # Ogranicz wymiarowość jeśli potrzeba (dla wydajności)
            feature_dim = X_train_normal.shape[1]
            if feature_dim > 50:
                selected_features = np.random.choice(feature_dim, 50, replace=False)
                X_train_normal_subset = X_train_normal[:, selected_features]
                feature_dim = 50
            else:
                selected_features = np.arange(feature_dim)
                X_train_normal_subset = X_train_normal
            
            # Dopasuj Vine Copula na normalnych danych
            copula = VineCopula('center')  # C-vine copula
            copula.fit(pd.DataFrame(X_train_normal_subset))
            
            # Generuj syntetyczne normalne dane
            X_synthetic_train_normal = copula.sample(train_normal_samples).values
            X_synthetic_test_normal = copula.sample(test_anom_samples).values
            
            # Dla anomalii typu dependency, zaburzamy określony procent cech
            num_features_to_disturb = int(feature_dim * difficulty_param)
            disturbed_features = np.random.choice(feature_dim, num_features_to_disturb, replace=False)
            
            # Generuj anomalie z niezależnymi cechami
            X_synthetic_train_anomalies = np.zeros((train_anom_samples, feature_dim))
            X_synthetic_test_anomalies = np.zeros((test_anom_samples, feature_dim))
            
            for j in range(feature_dim):
                kde = GaussianKDE()
                kde.fit(X_train_normal_subset[:, j])
                
                if j in disturbed_features:
                    # Zaburzone cechy - użyj KDE dla niezależnej generacji
                    X_synthetic_train_anomalies[:, j] = kde.sample(train_anom_samples)
                    X_synthetic_test_anomalies[:, j] = kde.sample(test_anom_samples)
                else:
                    # Niezaburzone cechy - użyj tego samego rozkładu co dla normalnych
                    sampled_data = copula.sample(train_anom_samples + test_anom_samples).values[:, j]
                    X_synthetic_train_anomalies[:, j] = sampled_data[:train_anom_samples]
                    X_synthetic_test_anomalies[:, j] = sampled_data[train_anom_samples:]
            
            # Jeśli używaliśmy podzbioru cech, musimy odtworzyć pełny wymiar
            if feature_dim < X_train_normal.shape[1]:
                full_X_synthetic_train_normal = np.zeros((train_normal_samples, X_train_normal.shape[1]))
                full_X_synthetic_train_anomalies = np.zeros((train_anom_samples, X_train_normal.shape[1]))
                full_X_synthetic_test_anomalies = np.zeros((test_anom_samples, X_train_normal.shape[1]))
                
                for j, orig_j in enumerate(selected_features):
                    full_X_synthetic_train_normal[:, orig_j] = X_synthetic_train_normal[:, j]
                    full_X_synthetic_train_anomalies[:, orig_j] = X_synthetic_train_anomalies[:, j]
                    full_X_synthetic_test_anomalies[:, orig_j] = X_synthetic_test_anomalies[:, j]
                
                X_synthetic_train_normal = full_X_synthetic_train_normal
                X_synthetic_train_anomalies = full_X_synthetic_train_anomalies
                X_synthetic_test_anomalies = full_X_synthetic_test_anomalies
        
        # Łączenie danych treningowych i testowych
        X_train = np.vstack((X_synthetic_train_normal, X_synthetic_train_anomalies))
        y_train = np.hstack((np.zeros(len(X_synthetic_train_normal)), np.ones(len(X_synthetic_train_anomalies))))
        
        # Dodanie normalnych danych do zbioru testowego i sztcznych anomalii
        X_test = np.vstack((X_synthetic_test_normal, X_synthetic_test_anomalies))
        y_test = np.hstack((np.zeros(len(X_synthetic_test_normal)), np.ones(len(X_synthetic_test_anomalies))))
        
        # Tasowanie indeksów dla zbiorów treningowych
        train_shuffle_idx = np.random.permutation(len(X_train))
        X_train = X_train[train_shuffle_idx]
        y_train = y_train[train_shuffle_idx]
        
        # Tasowanie indeksów dla zbiorów testowych
        test_shuffle_idx = np.random.permutation(len(X_test))
        X_test = X_test[test_shuffle_idx]
        y_test = y_test[test_shuffle_idx]
        
        # Zapisz dane do słownika
        data_dict[f'X_train_{sc_label}'] = X_train
        data_dict[f'y_train_{sc_label}'] = y_train
        data_dict[f'X_test_{sc_label}'] = X_test
        data_dict[f'y_test_{sc_label}'] = y_test
    
    # Zapisz wszystkie dane do pliku npz
    if save_dataset:
        np.savez(f'datasets/E3/E3_{dataset}-{anomaly_type}_{seed}_data.npz', **data_dict)
    
    return data_dict
  


    
if __name__ == "__main__":
    prepare_data_e2('NSL-KDD')
    prepare_data_e2('CreditCard')
    prepare_data_e2('AnnThyroid')
    prepare_data_e2('EmailSpam-bert')    