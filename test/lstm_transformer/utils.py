import pandas as pd
import numpy as np

def generate_sample_data(start_date='2023-01-01', days=100, num_features=10):
    features = {f'value_{i}': np.random.rand(days) for i in range(1, num_features + 1)}

    data = {
        'timestamp': pd.date_range(start=start_date, periods=days, freq='d'),
        **features
    }
    
    df = pd.DataFrame(data)
    return df


if __name__ == '__main__':
    sample_data = generate_sample_data()
    print(sample_data.head())