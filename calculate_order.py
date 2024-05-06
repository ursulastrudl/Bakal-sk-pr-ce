import pandas as pd
from scipy import stats
import numpy as np

def load_data():
    # Load data from CSV file with specified separator and using the first column as index
    data_file_path = session.get('uploaded_data_file_path', None)
    data = pd.read_csv(data_file_path, sep=';', index_col=0, header=None)
    return data

def compute_statistics(data):
    # Výpočet statistik pro každý produkt
    for product, values in data.iterrows():
        mean = np.mean(values)
        std_dev = np.std(values, ddof=1)
        print(f'Produkt: {product}, Střední hodnota: {mean}, Výběrová směrodatná odchylka: {std_dev}')

def determine_distribution(data):
    # Určení nejlepšího rozdělení pravděpodobnosti pro každý produkt
    np.random.seed(42) 
    distributions = [stats.norm, stats.poisson, stats.uniform]
    products = data.index  # Získat názvy produktů z indexu DataFrame
    for product in products:
        best_fit = None
        best_p_value = 0
        for distribution in distributions:
            if distribution.name == 'norm':
                # Pro normální rozdělení potřebujeme střední hodnotu (mu) a výběrovou směrodatnou odchylku (scale)
                sample = distribution.rvs(loc=np.mean(data.loc[product]),
                scale=np.std(data.loc[product], ddof=1), size=len(data.loc[product]))
            elif distribution.name == 'poisson':
                # Pro Poissonovo rozdělení potřebujeme intenzitu (mu)
                sample = distribution.rvs(mu=np.mean(data.loc[product]), size=len(data.loc[product]))
            else:
                # Pro uniformní rozdělení potřebujeme minimální a maximální hodnotu intervalu
                sample = distribution.rvs(loc=np.min(data.loc[product]), scale=np.max(data.loc[product]) - np.min(data.loc[product]), size=len(data.loc[product]))
            
            # Test dobré shody (Kolmogorov-Smirnov) s použitím náhodných vzorků
            D, p = stats.kstest(data.loc[product], distribution.name, args=(sample,))
            if p > best_p_value:
                best_p_value = p
                best_fit = distribution.name
        print(f'Produkt: {product}, Nejlepší rozdělení: {best_fit}')

