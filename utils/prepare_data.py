import numpy as np
from sklearn.model_selection import train_test_split



def get_data(dataset):
    datasets_paths = {
        'NSL-KDD': 'nsl_kdd.npz',
        'CreditCard': 'fraud.npz',
        'AnnThyroid': 'annthyroid.npz',
        'EmailSpam': 'email.npz'
    }
    
    data = np.load(f"datasets/preprocessed_datasets/{datasets_paths[dataset]}")
    
    X = data['X']
    y = data['y']
    
    return X, y

def prepare_data_e1(dataset, seed=42):
    X, y = get_data(dataset)
    
    np.random.seed(seed)
    
    # Scenarios definition for each dataset
    scenarios = {
        'NSL-KDD': {'normal':5000, 'anomalies':[0,25,50,250,500], 'test_n':2000, 'test_a':200},
        'CreditCard': {'normal':5000, 'anomalies':[0,25,50,150,292], 'test_n':2000, 'test_a':200},
        'AnnThyroid': {'normal':3000, 'anomalies':[0,15,30,150,300], 'test_n':1000, 'test_a':100},
        'EmailSpam': {'normal':2000, 'anomalies':[0,10,20,50,100], 'test_n':1000, 'test_a':46}
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
    np.savez(f'datasets/E1/E1_{dataset}_{seed}_data.npz', **data_dict)

# Example of usage:
# prepare_data_e1(X, y, 'NSL-KDD', seed=123)
