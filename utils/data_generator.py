
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataGenerator:
    def __init__(self):
        np.random.seed(42)

    def generate_sample_data(self, data_type):
        if data_type == "linear_regression":
            return self.generate_linear_data()
        elif data_type == "time_series":
            return self.generate_time_series_data()
        elif data_type == "panel_data":
            return self.generate_panel_data()
        elif data_type == "classification":
            return self.generate_classification_data()
        else:
            return self.generate_linear_data()

    def generate_linear_data(self, n=500):
        """Generate data for linear regression"""
        np.random.seed(42)

        # Generate features
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        x3 = np.random.uniform(-2, 2, n)
        x4 = x1 + np.random.normal(0, 0.5, n)  # Correlated with x1

        # Generate target with linear relationship
        y = 2 + 3*x1 + 1.5*x2 - 2*x3 + 0.5*x4 + np.random.normal(0, 1, n)

        return pd.DataFrame({
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'x4': x4,
            'y': y
        })

    def generate_time_series_data(self, n=365):
        """Generate time series data with trend and seasonality"""
        np.random.seed(42)

        dates = pd.date_range(start='2022-01-01', periods=n, freq='D')

        # Components
        trend = np.linspace(100, 150, n)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 365.25)
        noise = np.random.normal(0, 5, n)

        # Combine components
        values = trend + seasonal + noise

        # Add some AR component
        for i in range(1, len(values)):
            values[i] = 0.3 * values[i-1] + 0.7 * values[i]

        return pd.DataFrame({
            'date': dates,
            'value': values,
            'trend': trend,
            'seasonal': seasonal
        })

    def generate_panel_data(self):
        """Generate panel data for fixed/random effects models"""
        np.random.seed(42)

        entities = ['Company_' + str(i) for i in range(1, 21)]
        years = range(2015, 2024)

        data = []

        for entity in entities:
            # Entity-specific effect
            entity_effect = np.random.normal(0, 2)

            for year in years:
                # Time effect
                time_effect = 0.5 * (year - 2015)

                # Generate variables
                x1 = np.random.normal(5, 1)
                x2 = np.random.normal(10, 2)
                x3 = np.random.uniform(0, 1)

                # Generate outcome
                y = (entity_effect + time_effect + 
                      2*x1 + 1.5*x2 + 3*x3 + 
                      np.random.normal(0, 1))

                data.append({
                    'entity': entity,
                    'year': year,
                    'revenue': y * 1000000,  # Scale to millions
                    'employees': int(x1 * 100),
                    'rd_spending': x2 * 100000,
                    'market_share': x3
                })

        return pd.DataFrame(data)

    def generate_classification_data(self, n=500):
        """Generate data for logistic regression"""
        np.random.seed(42)

        # Generate features
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        x3 = np.random.uniform(-2, 2, n)

        # Generate probabilities
        logits = -1 + 2*x1 + 1.5*x2 - x3
        probs = 1 / (1 + np.exp(-logits))

        # Generate binary outcome
        y = (probs > np.random.uniform(0, 1, n)).astype(int)

        return pd.DataFrame({
            'feature1': x1,
            'feature2': x2,
            'feature3': x3,
            'target': y
        })
