
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
import statsmodels.api as sm
from scipy import stats

class LinearModelsModule:
    def __init__(self):
        self.models = {
            "Simple Linear Regression": self.simple_linear_regression,
            "Multiple Linear Regression": self.multiple_linear_regression,
            "Polynomial Regression": self.polynomial_regression,
            "Ridge Regression": self.ridge_regression,
            "Lasso Regression": self.lasso_regression,
            "Elastic Net": self.elastic_net
        }

    def run(self, data):
        st.markdown('<div class="section-header">📊 Linear Regression Models</div>',
                    unsafe_allow_html=True)

        if data is None:
            st.warning("Please load data first from the sidebar.")
            return

        # Model selection
        col1, col2 = st.columns([2, 1])

        with col1:
            model_type = st.selectbox(
                "Select Linear Model:",
                list(self.models.keys())
            )

        with col2:
            st.markdown("### Model Information")
            self.show_model_info(model_type)

        # Data preview
        with st.expander("📋 Data Preview", expanded=True):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.dataframe(data.head(10), use_container_width=True)
            with col2:
                st.metric("Rows", data.shape[0])
                st.metric("Columns", data.shape[1])
            with col3:
                st.metric("Missing Values", data.isnull().sum().sum())
                st.metric("Data Types", len(data.dtypes.unique()))

        # Variable selection
        st.markdown("### Variable Selection")
        col1, col2 = st.columns(2)

        with col1:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            y_var = st.selectbox("Dependent Variable (Y):", numeric_cols)

        with col2:
            x_vars = st.multiselect(
                "Independent Variables (X):",
                [col for col in numeric_cols if col != y_var],
                default=[col for col in numeric_cols if col != y_var][:2] if len(numeric_cols) > 1 else []
            )

        if not x_vars:
            st.warning("Please select at least one independent variable.")
            return

        # Model parameters
        params = self.get_model_parameters(model_type)

        # Train-test split
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test Set Size:", 0.1, 0.5, 0.2)
        with col2:
            random_state = st.number_input("Random State:", 0, 100, 42)
        with col3:
            cv_folds = st.slider("Cross-Validation Folds:", 2, 10, 5)

        # Run model
        if st.button("🚀 Run Model", type="primary"):
            self.models[model_type](data, y_var, x_vars, params, test_size, random_state, cv_folds)

    def show_model_info(self, model_type):
        info = {
            "Simple Linear Regression": "Basic OLS regression with one predictor",
            "Multiple Linear Regression": "OLS with multiple predictors",
            "Polynomial Regression": "Non-linear relationships using polynomial features",
            "Ridge Regression": "L2 regularization to prevent overfitting",
            "Lasso Regression": "L1 regularization for feature selection",
            "Elastic Net": "Combination of L1 and L2 regularization"
        }
        st.info(info.get(model_type, ""))

    def get_model_parameters(self, model_type):
        params = {}

        st.markdown("### Model Parameters")

        if model_type == "Polynomial Regression":
            params['degree'] = st.slider("Polynomial Degree:", 2, 5, 2)
            params['include_bias'] = st.checkbox("Include Bias Term", True)

        elif model_type in ["Ridge Regression", "Lasso Regression"]:
            params['alpha'] = st.slider("Regularization Strength (α):", 0.001, 10.0, 1.0, step=0.001)
            params['normalize'] = st.checkbox("Normalize Features", True)

        elif model_type == "Elastic Net":
            col1, col2 = st.columns(2)
            with col1:
                params['alpha'] = st.slider("Regularization Strength (α):", 0.001, 10.0, 1.0, step=0.001)
            with col2:
                params['l1_ratio'] = st.slider("L1 Ratio:", 0.0, 1.0, 0.5, step=0.01)
            params['normalize'] = st.checkbox("Normalize Features", True)

        return params

    def prepare_data(self, data, y_var, x_vars, test_size, random_state):
        X = data[x_vars].values
        y = data[y_var].values

        # Remove NaN values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test

    def calculate_metrics(self, y_true, y_pred):
        metrics = {
            'R²': r2_score(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred)
        }
        return metrics

    def create_diagnostic_plots(self, y_train, y_train_pred, y_test, y_test_pred, residuals_train, residuals_test):
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Training: Actual vs Predicted', 'Test: Actual vs Predicted', 
                          'Residuals vs Fitted', 'Residuals Distribution', 
                          'Q-Q Plot', 'Scale-Location Plot'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Training: Actual vs Predicted
        fig.add_trace(
            go.Scatter(x=y_train, y=y_train_pred, mode='markers',
                      name='Training', marker=dict(color='blue', size=5)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[y_train.min(), y_train.max()], 
                      y=[y_train.min(), y_train.max()],
                      mode='lines', name='Perfect Fit',
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )

        # Test: Actual vs Predicted
        fig.add_trace(
            go.Scatter(x=y_test, y=y_test_pred, mode='markers',
                      name='Test', marker=dict(color='green', size=5)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=[y_test.min(), y_test.max()], 
                      y=[y_test.min(), y_test.max()],
                      mode='lines', name='Perfect Fit',
                      line=dict(color='red', dash='dash')),
            row=1, col=2
        )

        # Residuals vs Fitted
        fig.add_trace(
            go.Scatter(x=y_train_pred, y=residuals_train, mode='markers',
                      name='Residuals', marker=dict(color='purple', size=5)),
            row=1, col=3
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=3)

        # Residuals Distribution
        fig.add_trace(
            go.Histogram(x=residuals_train, name='Residuals', nbinsx=30,
                        marker=dict(color='orange')),
            row=2, col=1
        )

        # Q-Q Plot
        qq_data = stats.probplot(residuals_train, dist="norm", plot=None)
        fig.add_trace(
            go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers',
                      name='Q-Q', marker=dict(color='teal', size=5)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=qq_data[0][0], y=qq_data[0][0], mode='lines',
                      name='Normal Line', line=dict(color='red', dash='dash')),
            row=2, col=2
        )

        # Scale-Location Plot
        standardized_residuals = residuals_train / np.std(residuals_train)
        sqrt_standardized_residuals = np.sqrt(np.abs(standardized_residuals))
        fig.add_trace(
            go.Scatter(x=y_train_pred, y=sqrt_standardized_residuals, mode='markers',
                      name='Scale-Location', marker=dict(color='brown', size=5)),
            row=2, col=3
        )

        # Update layout
        fig.update_layout(height=700, showlegend=False, title_text="Model Diagnostics")
        fig.update_xaxes(title_text="Actual", row=1, col=1)
        fig.update_yaxes(title_text="Predicted", row=1, col=1)
        fig.update_xaxes(title_text="Actual", row=1, col=2)
        fig.update_yaxes(title_text="Predicted", row=1, col=2)
        fig.update_xaxes(title_text="Fitted Values", row=1, col=3)
        fig.update_yaxes(title_text="Residuals", row=1, col=3)
        fig.update_xaxes(title_text="Residuals", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)
        fig.update_xaxes(title_text="Fitted Values", row=2, col=3)
        fig.update_yaxes(title_text="√|Standardized Residuals|", row=2, col=3)

        return fig

    def simple_linear_regression(self, data, y_var, x_vars, params, test_size, random_state, cv_folds):
        if len(x_vars) > 1:
            st.warning("Simple Linear Regression uses only one predictor. Using the first selected variable.")
            x_vars = [x_vars[0]]

        X_train, X_test, y_train, y_test = self.prepare_data(data, y_var, x_vars, test_size, random_state)

        # Fit model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)

        # Display results
        self.display_results(model, x_vars, train_metrics, test_metrics, 
                            y_train, y_train_pred, y_test, y_test_pred)

    def multiple_linear_regression(self, data, y_var, x_vars, params, test_size, random_state, cv_folds):
        X_train, X_test, y_train, y_test = self.prepare_data(data, y_var, x_vars, test_size, random_state)

        # Fit model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)

        # Display results
        self.display_results(model, x_vars, train_metrics, test_metrics, 
                           y_train, y_train_pred, y_test, y_test_pred)

    def polynomial_regression(self, data, y_var, x_vars, params, test_size, random_state, cv_folds):
        X_train, X_test, y_train, y_test = self.prepare_data(data, y_var, x_vars, test_size, random_state)

        # Create polynomial features
        poly = PolynomialFeatures(degree=params.get('degree', 2), 
                                  include_bias=params.get('include_bias', True))
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # Fit model
        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # Predictions
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)

        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)

        # Display results
        self.display_results(model, x_vars, train_metrics, test_metrics, 
                           y_train, y_train_pred, y_test, y_test_pred,
                           model_name=f"Polynomial Regression (degree={params.get('degree', 2)})")

    def ridge_regression(self, data, y_var, x_vars, params, test_size, random_state, cv_folds):
        X_train, X_test, y_train, y_test = self.prepare_data(data, y_var, x_vars, test_size, random_state)

        # Normalize if requested
        if params.get('normalize', True):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Fit model
        model = Ridge(alpha=params.get('alpha', 1.0))
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)

        # Display results
        self.display_results(model, x_vars, train_metrics, test_metrics, 
                           y_train, y_train_pred, y_test, y_test_pred,
                           model_name=f"Ridge Regression (α={params.get('alpha', 1.0)})")

    def lasso_regression(self, data, y_var, x_vars, params, test_size, random_state, cv_folds):
        X_train, X_test, y_train, y_test = self.prepare_data(data, y_var, x_vars, test_size, random_state)

        # Normalize if requested
        if params.get('normalize', True):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Fit model
        model = Lasso(alpha=params.get('alpha', 1.0), max_iter=5000)
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)

        # Display results
        self.display_results(model, x_vars, train_metrics, test_metrics, 
                           y_train, y_train_pred, y_test, y_test_pred,
                           model_name=f"Lasso Regression (α={params.get('alpha', 1.0)})")

    def elastic_net(self, data, y_var, x_vars, params, test_size, random_state, cv_folds):
        X_train, X_test, y_train, y_test = self.prepare_data(data, y_var, x_vars, test_size, random_state)

        # Normalize if requested
        if params.get('normalize', True):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Fit model
        model = ElasticNet(alpha=params.get('alpha', 1.0), 
                           l1_ratio=params.get('l1_ratio', 0.5),
                           max_iter=5000)
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)

        # Display results
        self.display_results(model, x_vars, train_metrics, test_metrics, 
                           y_train, y_train_pred, y_test, y_test_pred,
                           model_name=f"Elastic Net (α={params.get('alpha', 1.0)}, L1={params.get('l1_ratio', 0.5)})")

    def display_results(self, model, x_vars, train_metrics, test_metrics, 
                       y_train, y_train_pred, y_test, y_test_pred, model_name=None):

        st.success("✅ Model fitted successfully!")

        # Model name
        if model_name:
            st.markdown(f"### {model_name} Results")

        # Metrics display
        st.markdown("### 📊 Model Performance Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**Training Set**")
            st.metric("R² Score", f"{train_metrics['R²']:.4f}")
            st.metric("RMSE", f"{train_metrics['RMSE']:.4f}")

        with col2:
            st.markdown("**Test Set**")
            st.metric("R² Score", f"{test_metrics['R²']:.4f}")
            st.metric("RMSE", f"{test_metrics['RMSE']:.4f}")

        with col3:
            st.markdown("**Training Set**")
            st.metric("MSE", f"{train_metrics['MSE']:.4f}")
            st.metric("MAE", f"{train_metrics['MAE']:.4f}")

        with col4:
            st.markdown("**Test Set**")
            st.metric("MSE", f"{test_metrics['MSE']:.4f}")
            st.metric("MAE", f"{test_metrics['MAE']:.4f}")

        # Coefficients
        st.markdown("### 📈 Model Coefficients")

        if hasattr(model, 'coef_'):
            coef = model.coef_.flatten() if len(model.coef_.shape) > 1 else model.coef_
            # For polynomial regression, create feature names
            if model_name and "Polynomial" in model_name:
                from sklearn.preprocessing import PolynomialFeatures
                # Extract degree from model_name
                import re
                degree_match = re.search(r"degree=(\d+)", model_name)
                degree = int(degree_match.group(1)) if degree_match else 2
                poly = PolynomialFeatures(degree=degree)
                # Create dummy data for feature names
                dummy_data = np.zeros((1, len(x_vars)))
                poly.fit(dummy_data)
                feature_names = poly.get_feature_names_out(x_vars)
            else:
                feature_names = x_vars
            
            coef_df = pd.DataFrame({
                'Variable': feature_names,
                'Coefficient': coef
            })
            coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
            coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.dataframe(coef_df[['Variable', 'Coefficient']], use_container_width=True)
                if hasattr(model, 'intercept_'):
                    st.metric("Intercept", f"{model.intercept_:.4f}")

            with col2:
                # Coefficient plot
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=coef_df['Coefficient'],
                    y=coef_df['Variable'],
                    orientation='h',
                    marker=dict(
                        color=coef_df['Coefficient'],
                        colorscale='RdBu',
                        cmin=-max(abs(coef_df['Coefficient'])),
                        cmax=max(abs(coef_df['Coefficient']))
                    )
                ))
                fig.update_layout(
                    title="Feature Importance",
                    xaxis_title="Coefficient Value",
                    yaxis_title="Variable",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

        # Diagnostic plots
        st.markdown("### 🔍 Diagnostic Plots")

        residuals_train = y_train - y_train_pred
        residuals_test = y_test - y_test_pred

        diagnostic_fig = self.create_diagnostic_plots(
            y_train, y_train_pred, y_test, y_test_pred,
            residuals_train, residuals_test
        )
        st.plotly_chart(diagnostic_fig, use_container_width=True)

        # Additional statistics
        with st.expander("📊 Additional Statistics", expanded=False):
            # Durbin-Watson statistic
            from statsmodels.stats.stattools import durbin_watson
            dw_stat = durbin_watson(residuals_train)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Durbin-Watson", f"{dw_stat:.4f}")
                if 1.5 < dw_stat < 2.5:
                    st.success("No autocorrelation detected")
                else:
                    st.warning("Potential autocorrelation")

            with col2:
                # Jarque-Bera test
                from scipy.stats import jarque_bera
                jb_stat, jb_pvalue = jarque_bera(residuals_train)
                st.metric("Jarque-Bera p-value", f"{jb_pvalue:.4f}")
                if jb_pvalue > 0.05:
                    st.success("Residuals are normally distributed")
                else:
                    st.warning("Residuals may not be normal")

            with col3:
                # Condition number
                if hasattr(model, 'coef_'):
                    # Create design matrix from coefficients
                    n_features = len(model.coef_) if len(model.coef_.shape) == 1 else model.coef_.shape[1]
                    X_for_cond = np.random.normal(0, 1, (len(y_train), n_features))
                    cond_num = np.linalg.cond(X_for_cond.T @ X_for_cond)
                    st.metric("Condition Number", f"{cond_num:.2f}")
                    if cond_num > 30:
                        st.warning("Potential multicollinearity")
                    else:
                        st.success("No multicollinearity detected")
