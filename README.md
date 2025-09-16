
# Advanced Econometric Simulator

A comprehensive, professional-grade econometric analysis platform built with Python and Streamlit. This application provides interactive tools for statistical modeling, time series analysis, panel data econometrics, and advanced statistical testing.

## 🚀 Features

### Econometric Models
- **Linear Models**: OLS, Ridge, Lasso, Elastic Net, Polynomial Regression
- **Time Series**: ARIMA, SARIMA, VAR, GARCH, Exponential Smoothing, Holt-Winters
- **Panel Data**: Fixed Effects, Random Effects, Pooled OLS, Difference-in-Differences
- **Advanced Models**: Logistic, Probit, Tobit, Ordered Models, Quantile Regression

### Statistical Tests
- **Unit Root Tests**: ADF, KPSS, Phillips-Perron
- **Cointegration**: Engle-Granger, Johansen
- **Diagnostic Tests**: Durbin-Watson, Breusch-Pagan, Jarque-Bera, RESET
- **Causality Tests**: Granger Causality, Toda-Yamamoto

### Interactive Features
- Real-time parameter adjustment
- Animated visualizations
- Bootstrap confidence intervals
- 3D surface plots
- Interactive time series with range selectors

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/econometric-simulator.git
cd econometric-simulator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

Run the application:
```bash
streamlit run main.py
```

The app will open in your browser at http://localhost:8501

## 📁 Project Structure

```
econometric-simulator/
│
├── main.py                 # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # Documentation
│
├── models/                # Model implementations
│   ├── __init__.py
│   ├── linear_models.py   # Linear regression models
│   ├── time_series_models.py  # Time series models
│   ├── panel_models.py    # Panel data models
│   └── advanced_models.py # Advanced econometric models
│
└── utils/                 # Utility modules
    ├── __init__.py
    ├── data_generator.py  # Sample data generation
    ├── statistical_tests.py  # Statistical testing suite
    └── visualizations.py  # Visualization utilities
```

## 📊 Data Input Options

- **Sample Data**: Pre-generated datasets for testing
- **File Upload**: Support for CSV and Excel files
- **Manual Entry**: Direct data input interface
- **API Integration**: Connect to economic data sources (FRED, World Bank)

## 🎯 Quick Start Guide

1. **Select Model Category** from the sidebar
2. **Load Data** using one of the input methods
3. **Configure Parameters** using interactive controls
4. **Run Analysis** and view results
5. **Export Results** for further analysis

## 🔧 Model Examples

### Linear Regression
```python
# Simple OLS with regularization options
- Ridge (L2): Prevents overfitting
- Lasso (L1): Feature selection
- Elastic Net: Combines L1 and L2
```

### Time Series
```python
# ARIMA modeling with automatic parameter selection
- Trend analysis
- Seasonal decomposition
- Forecasting with confidence intervals
```

### Panel Data
```python
# Fixed and Random Effects
- Entity and time effects
- Hausman test for model selection
- Clustered standard errors
```

## 📈 Key Features

### Interactive Visualizations
- Real-time plot updates
- Hover tooltips with detailed information
- Zoom and pan capabilities
- Export plots as images

### Model Diagnostics
- Residual analysis
- Q-Q plots
- Heteroskedasticity tests
- Autocorrelation checks

### Statistical Testing
- Comprehensive test suite
- Automatic interpretation
- Visual test results
- P-value highlighting

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 👨‍💻 Author

HAMDI Boulanouar
Email: contact@example.com
LinkedIn: Profile
GitHub: Profile

## 🙏 Acknowledgments

- Streamlit for the amazing web framework
- Statsmodels for econometric implementations
- Plotly for interactive visualizations
- The open-source community

## 📚 Documentation

For detailed documentation, visit our Wiki

## 🐛 Bug Reports

Please report bugs through GitHub Issues

---

Version: 1.0.0
Last Updated: 2024
Python: 3.8+
Streamlit: 1.28.0+
