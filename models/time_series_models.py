
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, acf, pacf
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesModule:
    def __init__(self):
        self.models = {
            "ARIMA": self.arima_model,
            "SARIMA": self.sarima_model,
            "VAR": self.var_model,
            "GARCH": self.garch_model,
            "Exponential Smoothing": self.exp_smoothing,
            "Holt-Winters": self.holt_winters
        }

    def run(self, data):
        st.markdown('<div class="section-header">⏰ Time Series Analysis</div>',
                    unsafe_allow_html=True)

        if data is None:
            st.warning("Please load time series data from the sidebar.")
            return

        # Model selection
        model_type = st.selectbox(
            "Select Time Series Model:",
            list(self.models.keys())
        )

        # Data preparation
        st.markdown("### Data Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Check for datetime column
            date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                date_col = st.selectbox("Date Column:", date_cols)
                data[date_col] = pd.to_datetime(data[date_col])
                data = data.sort_values(date_col)
            else:
                st.info("No date column detected. Using index as time.")

        with col2:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            value_col = st.selectbox("Value Column:", numeric_cols)

        with col3:
            if model_type == "VAR":
                additional_cols = st.multiselect(
                    "Additional Variables (for VAR):",
                    [col for col in numeric_cols if col != value_col]
                )

        # Display time series plot
        st.markdown("### Time Series Visualization")

        fig = go.Figure()
        if date_cols:
            fig.add_trace(go.Scatter(
                x=data[date_col], y=data[value_col],
                mode='lines', name='Time Series',
                line=dict(color='blue', width=2)
            ))
            fig.update_xaxes(title_text="Date")
        else:
            fig.add_trace(go.Scatter(
                y=data[value_col], mode='lines',
                name='Time Series', line=dict(color='blue', width=2)
            ))
            fig.update_xaxes(title_text="Time Index")

        fig.update_yaxes(title_text=value_col)
        fig.update_layout(title="Time Series Plot", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Stationarity test
        with st.expander("📊 Stationarity Analysis", expanded=True):
            self.stationarity_analysis(data[value_col].dropna())

        # Model parameters
        params = self.get_ts_parameters(model_type)

        # Forecast settings
        st.markdown("### Forecast Settings")
        col1, col2 = st.columns(2)

        with col1:
            forecast_periods = st.slider("Forecast Periods:", 1, 100, 30)

        with col2:
            confidence_level = st.slider("Confidence Level:", 0.80, 0.99, 0.95)

        # Run model
        if st.button("🚀 Run Time Series Model", type="primary"):
            self.models[model_type](data, value_col, params, forecast_periods, confidence_level)

    def get_ts_parameters(self, model_type):
        params = {}

        st.markdown("### Model Parameters")

        if model_type == "ARIMA":
            col1, col2, col3 = st.columns(3)
            with col1:
                params['p'] = st.slider("AR order (p):", 0, 5, 1)
            with col2:
                params['d'] = st.slider("Differencing (d):", 0, 2, 1)
            with col3:
                params['q'] = st.slider("MA order (q):", 0, 5, 1)

        elif model_type == "SARIMA":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Non-seasonal parameters**")
                params['p'] = st.slider("AR order (p):", 0, 3, 1)
                params['d'] = st.slider("Differencing (d):", 0, 2, 1)
                params['q'] = st.slider("MA order (q):", 0, 3, 1)
            with col2:
                st.markdown("**Seasonal parameters**")
                params['P'] = st.slider("Seasonal AR (P):", 0, 2, 1)
                params['D'] = st.slider("Seasonal diff (D):", 0, 1, 0)
                params['Q'] = st.slider("Seasonal MA (Q):", 0, 2, 1)
                params['s'] = st.slider("Seasonal period:", 4, 52, 12)

        elif model_type == "VAR":
            params['maxlags'] = st.slider("Maximum Lags:", 1, 20, 10)
            params['ic'] = st.selectbox("Information Criterion:", ['aic', 'bic', 'hqic', 'fpe'])

        elif model_type == "GARCH":
            col1, col2 = st.columns(2)
            with col1:
                params['p'] = st.slider("GARCH order (p):", 1, 5, 1)
                params['q'] = st.slider("ARCH order (q):", 1, 5, 1)
            with col2:
                params['dist'] = st.selectbox("Distribution:", ['normal', 't', 'skewt'])
                params['vol'] = st.selectbox("Volatility:", ['GARCH', 'EGARCH', 'TGARCH'])

        elif model_type in ["Exponential Smoothing", "Holt-Winters"]:
            col1, col2 = st.columns(2)
            with col1:
                params['trend'] = st.selectbox("Trend:", [None, 'add', 'mul'])
                params['seasonal'] = st.selectbox("Seasonal:", [None, 'add', 'mul'])
            with col2:
                if params['seasonal']:
                    params['seasonal_periods'] = st.slider("Seasonal Periods:", 2, 52, 12)
                params['damped'] = st.checkbox("Damped Trend", False)

        return params

    def stationarity_analysis(self, series):
        # ADF Test
        adf_result = adfuller(series)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Augmented Dickey-Fuller Test**")
            st.write(f"ADF Statistic: {adf_result[0]:.4f}")
            st.write(f"p-value: {adf_result[1]:.4f}")

            if adf_result[1] <= 0.05:
                st.success("✅ Series is stationary")
            else:
                st.warning("⚠️ Series is non-stationary")

        with col2:
            # ACF and PACF plots
            fig = make_subplots(rows=1, cols=2,
                              subplot_titles=('ACF Plot', 'PACF Plot'))

            # Calculate appropriate max lag (50% of sample size)
            max_lag = min(20, len(series) // 2 - 1)
            if max_lag <= 0:
                max_lag = 1
                
            acf_values = acf(series, nlags=max_lag)
            pacf_values = pacf(series, nlags=max_lag)

            fig.add_trace(go.Bar(y=acf_values, name='ACF'), row=1, col=1)
            fig.add_trace(go.Bar(y=pacf_values, name='PACF'), row=1, col=2)

            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    def arima_model(self, data, value_col, params, forecast_periods, confidence_level):
        try:
            # Prepare data
            ts_data = data[value_col].dropna()

            # Fit ARIMA model
            model = ARIMA(ts_data, order=(params['p'], params['d'], params['q']))
            fitted_model = model.fit()

            # Make predictions
            forecast = fitted_model.forecast(steps=forecast_periods)
            forecast_df = pd.DataFrame({
                'forecast': forecast,
                'index': range(len(ts_data), len(ts_data) + forecast_periods)
            })

            # Get prediction intervals
            forecast_result = fitted_model.get_forecast(steps=forecast_periods)
            conf_int = forecast_result.conf_int(alpha=1-confidence_level)

            # Display results
            st.success("✅ ARIMA model fitted successfully!")

            # Model summary
            with st.expander("Model Summary", expanded=True):
                st.text(str(fitted_model.summary()))

            # Forecast plot
            st.markdown("### Forecast Results")

            fig = go.Figure()

            # Historical data
            fig.add_trace(go.Scatter(
                y=ts_data.values, mode='lines',
                name='Historical', line=dict(color='blue')
            ))

            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_df['index'], y=forecast_df['forecast'],
                mode='lines', name='Forecast',
                line=dict(color='red', dash='dash')
            ))

            # Confidence intervals
            fig.add_trace(go.Scatter(
                x=list(forecast_df['index']) + list(forecast_df['index'][::-1]),
                y=list(conf_int.iloc[:, 0]) + list(conf_int.iloc[:, 1][::-1]),
                fill='toself', fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{int(confidence_level*100)}% CI'
            ))

            fig.update_layout(
                title=f"ARIMA({params['p']},{params['d']},{params['q']}) Forecast",
                xaxis_title="Time Index",
                yaxis_title=value_col,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Residual diagnostics
            self.plot_residual_diagnostics(fitted_model.resid)

        except Exception as e:
            st.error(f"Error fitting ARIMA model: {str(e)}")

    def sarima_model(self, data, value_col, params, forecast_periods, confidence_level):
        try:
            # Prepare data
            ts_data = data[value_col].dropna()

            # Fit SARIMA model
            model = SARIMAX(
                ts_data,
                order=(params['p'], params['d'], params['q']),
                seasonal_order=(params['P'], params['D'], params['Q'], params['s'])
            )
            fitted_model = model.fit(disp=False)

            # Make predictions
            forecast = fitted_model.forecast(steps=forecast_periods)

            # Get prediction intervals
            forecast_result = fitted_model.get_forecast(steps=forecast_periods)
            conf_int = forecast_result.conf_int(alpha=1-confidence_level)

            # Display results
            st.success("✅ SARIMA model fitted successfully!")

            # Model metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("AIC", f"{fitted_model.aic:.2f}")
            with col2:
                st.metric("BIC", f"{fitted_model.bic:.2f}")
            with col3:
                st.metric("HQIC", f"{fitted_model.hqic:.2f}")
            with col4:
                st.metric("Log Likelihood", f"{fitted_model.llf:.2f}")

            # Forecast plot
            self.plot_forecast(ts_data, forecast, conf_int, value_col, "SARIMA")

            # Model diagnostics
            self.plot_model_diagnostics(fitted_model)

        except Exception as e:
            st.error(f"Error fitting SARIMA model: {str(e)}")

    def garch_model(self, data, value_col, params, forecast_periods, confidence_level):
        try:
            # Prepare returns data
            ts_data = data[value_col].dropna()
            returns = 100 * ts_data.pct_change().dropna()

            # Fit GARCH model
            if params['vol'] == 'GARCH':
                model = arch_model(returns, vol='Garch', p=params['p'], q=params['q'],
                                  dist=params['dist'])
            elif params['vol'] == 'EGARCH':
                model = arch_model(returns, vol='EGARCH', p=params['p'], q=params['q'],
                                  dist=params['dist'])
            else:  # TGARCH
                model = arch_model(returns, vol='GARCH', p=params['p'], q=params['q'],
                                  dist=params['dist'], o=1)

            fitted_model = model.fit(disp='off')

            # Forecast
            forecast = fitted_model.forecast(horizon=forecast_periods)

            # Display results
            st.success(f"✅ {params['vol']} model fitted successfully!")

            # Model summary
            with st.expander("Model Summary", expanded=True):
                st.text(str(fitted_model.summary()))

            # Volatility plot
            st.markdown("### Conditional Volatility")

            fig = go.Figure()

            # Historical volatility
            fig.add_trace(go.Scatter(
                y=fitted_model.conditional_volatility,
                mode='lines', name='Conditional Volatility',
                line=dict(color='blue')
            ))

            # Forecast volatility
            forecast_var = forecast.variance.values[-1, :]
            forecast_vol = np.sqrt(forecast_var)

            fig.add_trace(go.Scatter(
                x=list(range(len(returns), len(returns) + forecast_periods)),
                y=forecast_vol, mode='lines',
                name='Forecast Volatility',
                line=dict(color='red', dash='dash')
            ))

            fig.update_layout(
                title=f"{params['vol']} Model - Conditional Volatility",
                xaxis_title="Time Index",
                yaxis_title="Volatility",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            # Returns vs Volatility
            self.plot_returns_volatility(returns, fitted_model.conditional_volatility)

        except Exception as e:
            st.error(f"Error fitting GARCH model: {str(e)}")

    def var_model(self, data, value_col, params, forecast_periods, confidence_level):
        try:
            st.info("VAR model requires multiple time series variables.")
            # Implementation would go here

        except Exception as e:
            st.error(f"Error fitting VAR model: {str(e)}")

    def exp_smoothing(self, data, value_col, params, forecast_periods, confidence_level):
        try:
            # Prepare data
            ts_data = data[value_col].dropna()

            # Fit Exponential Smoothing model
            model = ExponentialSmoothing(
                ts_data,
                trend=params.get('trend'),
                seasonal=params.get('seasonal'),
                seasonal_periods=params.get('seasonal_periods', 12) if params.get('seasonal') else None,
                damped_trend=params.get('damped', False) if params.get('trend') else False
            )
            fitted_model = model.fit()

            # Forecast
            forecast = fitted_model.forecast(steps=forecast_periods)

            # Display results
            st.success("✅ Exponential Smoothing model fitted successfully!")

            # Model parameters
            st.markdown("### Smoothing Parameters")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Alpha (level)", f"{fitted_model.params['smoothing_level']:.4f}")
            with col2:
                if params.get('trend'):
                    st.metric("Beta (trend)", f"{fitted_model.params.get('smoothing_trend', 'N/A'):.4f}")
            with col3:
                if params.get('seasonal'):
                    st.metric("Gamma (seasonal)", f"{fitted_model.params.get('smoothing_seasonal', 'N/A'):.4f}")

            # Forecast plot
            self.plot_simple_forecast(ts_data, forecast, value_col, "Exponential Smoothing")

        except Exception as e:
            st.error(f"Error fitting Exponential Smoothing model: {str(e)}")

    def holt_winters(self, data, value_col, params, forecast_periods, confidence_level):
        try:
            # Similar to exponential smoothing but with specific Holt-Winters implementation
            self.exp_smoothing(data, value_col, params, forecast_periods, confidence_level)

        except Exception as e:
            st.error(f"Error fitting Holt-Winters model: {str(e)}")

    def plot_forecast(self, historical, forecast, conf_int, value_col, model_name):
        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            y=historical.values, mode='lines',
            name='Historical', line=dict(color='blue', width=2)
        ))

        # Forecast
        forecast_index = range(len(historical), len(historical) + len(forecast))
        fig.add_trace(go.Scatter(
            x=list(forecast_index), y=forecast.values,
            mode='lines', name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))

        # Confidence intervals
        if conf_int is not None:
            fig.add_trace(go.Scatter(
                x=list(forecast_index) + list(forecast_index)[::-1],
                y=list(conf_int.iloc[:, 0]) + list(conf_int.iloc[:, 1])[::-1],
                fill='toself', fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))

        fig.update_layout(
            title=f"{model_name} Forecast",
            xaxis_title="Time Index",
            yaxis_title=value_col,
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

    def plot_simple_forecast(self, historical, forecast, value_col, model_name):
        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            y=historical.values, mode='lines',
            name='Historical', line=dict(color='blue', width=2)
        ))

        # Forecast
        forecast_index = range(len(historical), len(historical) + len(forecast))
        fig.add_trace(go.Scatter(
            x=list(forecast_index), y=forecast.values if hasattr(forecast, 'values') else forecast,
            mode='lines', name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig.update_layout(
            title=f"{model_name} Forecast",
            xaxis_title="Time Index",
            yaxis_title=value_col,
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    def plot_residual_diagnostics(self, residuals):
        st.markdown("### Residual Diagnostics")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residuals', 'Residuals ACF', 
                          'Residuals Distribution', 'Q-Q Plot')
        )

        # Residuals plot
        fig.add_trace(go.Scatter(y=residuals, mode='lines', name='Residuals'),
                     row=1, col=1)

        # ACF of residuals
        # Calculate appropriate max lag (50% of sample size)
        max_lag = min(20, len(residuals) // 2 - 1)
        if max_lag <= 0:
            max_lag = 1
        acf_values = acf(residuals, nlags=max_lag)
        fig.add_trace(go.Bar(y=acf_values, name='ACF'), row=1, col=2)

        # Residuals distribution
        fig.add_trace(go.Histogram(x=residuals, nbinsx=30, name='Distribution'),
                     row=2, col=1)

        # Q-Q plot
        from scipy import stats
        qq_data = stats.probplot(residuals, dist="norm", plot=None)
        fig.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[0][1],
                                mode='markers', name='Q-Q'),
                     row=2, col=2)

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    def plot_model_diagnostics(self, fitted_model):
        st.markdown("### Model Diagnostics")

        # Create diagnostic plots using statsmodels
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Standardized residuals
        fitted_model.plot_diagnostics(fig=fig)

        st.pyplot(fig)

    def plot_returns_volatility(self, returns, volatility):
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Returns', 'Conditional Volatility'),
            row_heights=[0.5, 0.5]
        )

        # Returns
        fig.add_trace(go.Scatter(y=returns, mode='lines', name='Returns',
                                line=dict(color='blue', width=1)),
                     row=1, col=1)

        # Volatility
        fig.add_trace(go.Scatter(y=volatility, mode='lines', name='Volatility',
                                line=dict(color='red', width=2)),
                     row=2, col=1)

        fig.update_layout(height=600, showlegend=False)
        fig.update_xaxes(title_text="Time Index", row=2, col=1)
        fig.update_yaxes(title_text="Returns (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)
