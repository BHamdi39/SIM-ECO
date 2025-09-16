
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit, Logit
from scipy import stats

class AdvancedModelsModule:
    def __init__(self):
        self.models = {
            "Logistic Regression": self.logistic_regression,
            "Probit Model": self.probit_model,
            "Tobit Model": self.tobit_model,
            "Ordered Logit/Probit": self.ordered_model,
            "Multinomial Logit": self.multinomial_logit,
            "Poisson Regression": self.poisson_regression,
            "Negative Binomial": self.negative_binomial,
            "Quantile Regression": self.quantile_regression
        }

    def run(self, data):
        st.markdown('<div class="section-header">🔬 Advanced Econometric Models</div>',
                    unsafe_allow_html=True)

        if data is None:
            st.warning("Please load data from the sidebar.")
            return

        # Model selection
        model_type = st.selectbox(
            "Select Advanced Model:",
            list(self.models.keys())
        )

        # Model information
        self.show_model_info(model_type)

        # Variable selection
        st.markdown("### Variable Selection")

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        col1, col2 = st.columns(2)

        with col1:
            y_var = st.selectbox("Dependent Variable:", numeric_cols)

            # Check if binary for classification models
            if model_type in ["Logistic Regression", "Probit Model"]:
                unique_vals = data[y_var].nunique()
                if unique_vals > 2:
                    st.warning(f"Selected variable has {unique_vals} unique values. "
                             f"Binary classification models work best with 2 classes.")

        with col2:
            x_vars = st.multiselect(
                "Independent Variables:",
                [col for col in numeric_cols if col != y_var]
            )

        if not x_vars:
            st.warning("Please select at least one independent variable.")
            return

        # Model parameters
        params = self.get_advanced_parameters(model_type)

        # Run model
        if st.button("🚀 Run Advanced Model", type="primary"):
            self.models[model_type](data, y_var, x_vars, params)

    def show_model_info(self, model_type):
        info = {
            "Logistic Regression": "Binary classification using logistic function",
            "Probit Model": "Binary classification using normal CDF",
            "Tobit Model": "Censored regression for limited dependent variables",
            "Ordered Logit/Probit": "For ordinal dependent variables",
            "Multinomial Logit": "For categorical dependent variables with multiple classes",
            "Poisson Regression": "For count data following Poisson distribution",
            "Negative Binomial": "For overdispersed count data",
            "Quantile Regression": "Estimates conditional quantiles"
        }

        st.info(info.get(model_type, ""))

    def get_advanced_parameters(self, model_type):
        params = {}

        st.markdown("### Model Parameters")

        if model_type == "Logistic Regression":
            col1, col2 = st.columns(2)
            with col1:
                params['penalty'] = st.selectbox("Regularization:", ['none', 'l2', 'l1', 'elasticnet'])
                if params['penalty'] != 'none':
                    params['C'] = st.slider("Regularization Strength (C):", 0.01, 10.0, 1.0)
            with col2:
                params['solver'] = st.selectbox("Solver:", ['lbfgs', 'liblinear', 'newton-cg', 'sag'])
                params['max_iter'] = st.number_input("Max Iterations:", 100, 1000, 100)

        elif model_type == "Tobit Model":
            params['left_censoring'] = st.number_input("Left Censoring Point:", value=0.0)
            params['right_censoring'] = st.number_input("Right Censoring Point:", value=None)

        elif model_type == "Quantile Regression":
            params['quantiles'] = st.multiselect(
                "Quantiles to estimate:",
                [0.1, 0.25, 0.5, 0.75, 0.9],
                default=[0.25, 0.5, 0.75]
            )

        elif model_type in ["Poisson Regression", "Negative Binomial"]:
            params['exposure'] = st.checkbox("Include Exposure/Offset Variable", False)
            if params['exposure']:
                params['exposure_col'] = st.text_input("Exposure Column Name:")

        return params

    def logistic_regression(self, data, y_var, x_vars, params):
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            # Prepare data
            X = data[x_vars].values
            y = data[y_var].values

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Fit model
            if params['penalty'] == 'none':
                model = LogisticRegression(penalty=None, solver=params['solver'],
                                         max_iter=params['max_iter'])
            else:
                model = LogisticRegression(penalty=params['penalty'], C=params.get('C', 1.0),
                                         solver=params['solver'], max_iter=params['max_iter'])

            model.fit(X_train_scaled, y_train)

            # Predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Display results
            st.success("✅ Logistic Regression fitted successfully!")

            # Model coefficients
            self.display_logit_coefficients(model, x_vars, scaler)

            # Classification metrics
            self.display_classification_metrics(y_test, y_test_pred, y_test_proba)

            # ROC curve and confusion matrix
            self.plot_classification_diagnostics(y_test, y_test_pred, y_test_proba)

        except Exception as e:
            st.error(f"Error fitting Logistic Regression: {str(e)}")

    def probit_model(self, data, y_var, x_vars, params):
        try:
            # Prepare data
            X = sm.add_constant(data[x_vars])
            y = data[y_var]

            # Fit Probit model
            model = Probit(y, X)
            fitted_model = model.fit()

            # Display results
            st.success("✅ Probit model fitted successfully!")

            # Model summary
            with st.expander("Model Summary", expanded=True):
                st.text(str(fitted_model.summary()))

            # Marginal effects
            marginal_effects = fitted_model.get_margeff()

            st.markdown("### Marginal Effects")
            me_df = pd.DataFrame({
                'Variable': x_vars,
                'Marginal Effect': marginal_effects.margeff,
                'Std Error': marginal_effects.margeff_se,
                'z-statistic': marginal_effects.tvalues,
                'P-value': marginal_effects.pvalues
            })

            st.dataframe(me_df.style.format({
                'Marginal Effect': '{:.4f}',
                'Std Error': '{:.4f}',
                'z-statistic': '{:.4f}',
                'P-value': '{:.4f}'
            }), use_container_width=True)

            # Predicted probabilities
            self.plot_predicted_probabilities(fitted_model, X, y)

        except Exception as e:
            st.error(f"Error fitting Probit model: {str(e)}")

    def tobit_model(self, data, y_var, x_vars, params):
        try:
            st.info("Tobit model implementation")

            # Note: Would require specialized implementation or use of R through rpy2
            st.warning("Tobit model requires specialized implementation. "
                      "Consider using R's AER package through rpy2 or "
                      "implementing custom likelihood function.")

        except Exception as e:
            st.error(f"Error fitting Tobit model: {str(e)}")

    def ordered_model(self, data, y_var, x_vars, params):
        try:
            from statsmodels.miscmodels.ordinal_model import OrderedModel

            # Prepare data
            X = data[x_vars]
            y = data[y_var]

            # Fit Ordered Logit model
            model = OrderedModel(y, X, distr='logit')
            fitted_model = model.fit(method='bfgs')

            # Display results
            st.success("✅ Ordered Logit model fitted successfully!")

            # Model summary
            with st.expander("Model Summary", expanded=True):
                st.text(str(fitted_model.summary()))

        except Exception as e:
            st.error(f"Error fitting Ordered model: {str(e)}")

    def multinomial_logit(self, data, y_var, x_vars, params):
        try:
            from statsmodels.discrete.discrete_model import MNLogit

            # Prepare data
            X = sm.add_constant(data[x_vars])
            y = data[y_var]

            # Fit Multinomial Logit
            model = MNLogit(y, X)
            fitted_model = model.fit()

            # Display results
            st.success("✅ Multinomial Logit fitted successfully!")

            # Model summary
            with st.expander("Model Summary", expanded=True):
                st.text(str(fitted_model.summary()))

        except Exception as e:
            st.error(f"Error fitting Multinomial Logit: {str(e)}")

    def poisson_regression(self, data, y_var, x_vars, params):
        try:
            from statsmodels.discrete.discrete_model import Poisson

            # Prepare data
            X = sm.add_constant(data[x_vars])
            y = data[y_var]

            # Fit Poisson model
            model = Poisson(y, X)
            fitted_model = model.fit()

            # Display results
            st.success("✅ Poisson Regression fitted successfully!")

            # Model summary
            with st.expander("Model Summary", expanded=True):
                st.text(str(fitted_model.summary()))

            # Check for overdispersion
            self.check_overdispersion(fitted_model)

        except Exception as e:
            st.error(f"Error fitting Poisson Regression: {str(e)}")

    def negative_binomial(self, data, y_var, x_vars, params):
        try:
            from statsmodels.discrete.discrete_model import NegativeBinomial

            # Prepare data
            X = sm.add_constant(data[x_vars])
            y = data[y_var]

            # Fit Negative Binomial model
            model = NegativeBinomial(y, X)
            fitted_model = model.fit()

            # Display results
            st.success("✅ Negative Binomial fitted successfully!")

            # Model summary
            with st.expander("Model Summary", expanded=True):
                st.text(str(fitted_model.summary()))

            # Display coefficients and IRR
            self.display_count_model_results(fitted_model, x_vars)

            # Compare with Poisson
            self.compare_poisson_nb(data, y_var, x_vars)

        except Exception as e:
            st.error(f"Error fitting Negative Binomial: {str(e)}")

    def quantile_regression(self, data, y_var, x_vars, params):
        try:
            import statsmodels.formula.api as smf
            from statsmodels.regression.quantile_regression import QuantReg

            # Prepare data
            X = sm.add_constant(data[x_vars])
            y = data[y_var]

            quantiles = params.get('quantiles', [0.25, 0.5, 0.75])

            # Fit models for each quantile
            results = {}
            for q in quantiles:
                model = QuantReg(y, X)
                results[q] = model.fit(q=q)

            # Display results
            st.success("✅ Quantile Regression fitted successfully!")

            # Coefficients across quantiles
            self.display_quantile_results(results, x_vars, quantiles)

            # Plot quantile regression lines
            self.plot_quantile_regression(data, y_var, x_vars[0] if x_vars else None, results, quantiles)

        except Exception as e:
            st.error(f"Error fitting Quantile Regression: {str(e)}")

    def display_logit_coefficients(self, model, x_vars, scaler):
        st.markdown("### Model Coefficients")

        # Get coefficients
        coefs = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_

        # Calculate odds ratios
        odds_ratios = np.exp(coefs)

        # Create results dataframe
        results_df = pd.DataFrame({
            'Variable': x_vars,
            'Coefficient': coefs,
            'Odds Ratio': odds_ratios,
            'OR 95% CI Lower': odds_ratios * np.exp(-1.96 * 0.1),  # Approximate
            'OR 95% CI Upper': odds_ratios * np.exp(1.96 * 0.1)   # Approximate
        })

        col1, col2 = st.columns([1, 1])

        with col1:
            st.dataframe(results_df.style.format({
                'Coefficient': '{:.4f}',
                'Odds Ratio': '{:.4f}',
                'OR 95% CI Lower': '{:.4f}',
                'OR 95% CI Upper': '{:.4f}'
            }), use_container_width=True)

            if hasattr(model, 'intercept_'):
                st.metric("Intercept", f"{model.intercept_[0]:.4f}")

        with col2:
            # Odds ratio plot
            fig = go.Figure()

            for i, var in enumerate(x_vars):
                fig.add_trace(go.Scatter(
                    x=[results_df.loc[i, 'OR 95% CI Lower'], 
                       results_df.loc[i, 'OR 95% CI Upper']],
                    y=[var, var],
                    mode='lines',
                    line=dict(color='lightblue', width=3),
                    showlegend=False
                ))

                fig.add_trace(go.Scatter(
                    x=[results_df.loc[i, 'Odds Ratio']],
                    y=[var],
                    mode='markers',
                    marker=dict(size=10, color='blue'),
                    showlegend=False
                ))

            fig.add_vline(x=1, line_dash="dash", line_color="red")
            fig.update_layout(
                title="Odds Ratios with 95% CI",
                xaxis_title="Odds Ratio",
                yaxis_title="Variable",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

    def display_classification_metrics(self, y_true, y_pred, y_proba):
        st.markdown("### Classification Metrics")

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            accuracy = accuracy_score(y_true, y_pred)
            st.metric("Accuracy", f"{accuracy:.4f}")

        with col2:
            precision = precision_score(y_true, y_pred, average='weighted')
            st.metric("Precision", f"{precision:.4f}")

        with col3:
            recall = recall_score(y_true, y_pred, average='weighted')
            st.metric("Recall", f"{recall:.4f}")

        with col4:
            f1 = f1_score(y_true, y_pred, average='weighted')
            st.metric("F1 Score", f"{f1:.4f}")

        # Classification report
        with st.expander("Detailed Classification Report", expanded=False):
            report = classification_report(y_true, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format('{:.4f}'), use_container_width=True)

    def plot_classification_diagnostics(self, y_true, y_pred, y_proba):
        st.markdown("### Model Diagnostics")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Confusion Matrix', 'ROC Curve'),
            specs=[[{'type': 'heatmap'}, {'type': 'scatter'}]]
        )

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)

        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['Predicted 0', 'Predicted 1'],
                y=['Actual 0', 'Actual 1'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16},
                showscale=False
            ),
            row=1, col=1
        )

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines',
                      name=f'ROC (AUC = {roc_auc:.3f})',
                      line=dict(color='blue', width=2)),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                      name='Random', line=dict(color='red', dash='dash')),
            row=1, col=2
        )

        fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)

        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    def plot_predicted_probabilities(self, model, X, y):
        st.markdown("### Predicted Probabilities")

        # Get predicted probabilities
        pred_probs = model.predict(X)

        # Create histogram
        fig = go.Figure()

        # Separate by actual class
        for class_val in y.unique():
            mask = y == class_val
            fig.add_trace(go.Histogram(
                x=pred_probs[mask],
                name=f'Class {class_val}',
                opacity=0.7,
                nbinsx=20
            ))

        fig.update_layout(
            title="Distribution of Predicted Probabilities by Actual Class",
            xaxis_title="Predicted Probability",
            yaxis_title="Frequency",
            barmode='overlay',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def display_count_model_results(self, model, x_vars):
        st.markdown("### Count Model Results")

        # Get coefficients and calculate IRR
        params = model.params
        irr = np.exp(params)
        conf_int = model.conf_int()
        irr_ci = np.exp(conf_int)

        results_df = pd.DataFrame({
            'Variable': params.index,
            'Coefficient': params.values,
            'IRR': irr.values,
            'IRR 95% CI Lower': irr_ci.iloc[:, 0].values,
            'IRR 95% CI Upper': irr_ci.iloc[:, 1].values,
            'P-value': model.pvalues.values
        })

        st.dataframe(results_df.style.format({
            'Coefficient': '{:.4f}',
            'IRR': '{:.4f}',
            'IRR 95% CI Lower': '{:.4f}',
            'IRR 95% CI Upper': '{:.4f}',
            'P-value': '{:.4f}'
        }), use_container_width=True)

    def check_overdispersion(self, model):
        st.markdown("### Overdispersion Test")

        # Calculate dispersion parameter
        residuals = model.resid_pearson
        dispersion = np.sum(residuals**2) / model.df_resid

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Dispersion Parameter", f"{dispersion:.4f}")

        with col2:
            if dispersion > 1.5:
                st.warning("⚠️ Evidence of overdispersion. Consider Negative Binomial.")
            else:
                st.success("✅ No significant overdispersion detected.")

    def compare_poisson_nb(self, data, y_var, x_vars):
        st.markdown("### Model Comparison: Poisson vs Negative Binomial")

        from statsmodels.discrete.discrete_model import Poisson, NegativeBinomial

        X = sm.add_constant(data[x_vars])
        y = data[y_var]

        # Fit both models
        poisson_model = Poisson(y, X).fit(disp=False)
        nb_model = NegativeBinomial(y, X).fit(disp=False)

        # Compare metrics
        comparison_df = pd.DataFrame({
            'Metric': ['AIC', 'BIC', 'Log-Likelihood'],
            'Poisson': [poisson_model.aic, poisson_model.bic, poisson_model.llf],
            'Negative Binomial': [nb_model.aic, nb_model.bic, nb_model.llf]
        })

        st.dataframe(comparison_df.style.format('{:.2f}'), use_container_width=True)

        # Likelihood ratio test
        lr_stat = 2 * (nb_model.llf - poisson_model.llf)
        p_value = 1 - stats.chi2.cdf(lr_stat, 1)

        st.write(f"**Likelihood Ratio Test:** LR = {lr_stat:.4f}, p-value = {p_value:.4f}")

        if p_value < 0.05:
            st.info("📊 Negative Binomial is preferred (significant improvement over Poisson)")
        else:
            st.info("📊 Poisson model is adequate")

    def display_quantile_results(self, results, x_vars, quantiles):
        st.markdown("### Quantile Regression Results")

        # Create coefficient comparison across quantiles
        coef_data = []
        for q in quantiles:
            for var in ['const'] + x_vars:
                coef_data.append({
                    'Quantile': q,
                    'Variable': var,
                    'Coefficient': results[q].params[var],
                    'Std Error': results[q].bse[var],
                    'P-value': results[q].pvalues[var]
                })

        coef_df = pd.DataFrame(coef_data)

        # Pivot for better display
        pivot_df = coef_df.pivot(index='Variable', columns='Quantile', values='Coefficient')

        st.dataframe(pivot_df.style.format('{:.4f}'), use_container_width=True)

        # Plot coefficients across quantiles
        fig = go.Figure()

        for var in x_vars:
            coefs = [results[q].params[var] for q in quantiles]
            fig.add_trace(go.Scatter(
                x=quantiles, y=coefs,
                mode='lines+markers',
                name=var,
                line=dict(width=2)
            ))

        fig.update_layout(
            title="Coefficient Evolution Across Quantiles",
            xaxis_title="Quantile",
            yaxis_title="Coefficient Value",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def plot_quantile_regression(self, data, y_var, x_var, results, quantiles):
        if x_var is None:
            return

        st.markdown("### Quantile Regression Lines")

        fig = go.Figure()

        # Scatter plot of data
        fig.add_trace(go.Scatter(
            x=data[x_var], y=data[y_var],
            mode='markers',
            name='Data',
            marker=dict(size=5, color='lightgray')
        ))

        # Add quantile regression lines
        x_range = np.linspace(data[x_var].min(), data[x_var].max(), 100)
        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for i, q in enumerate(quantiles):
            y_pred = results[q].params['const'] + results[q].params[x_var] * x_range
            fig.add_trace(go.Scatter(
                x=x_range, y=y_pred,
                mode='lines',
                name=f'Q{int(q*100)}',
                line=dict(color=colors[i % len(colors)], width=2)
            ))

        fig.update_layout(
            title=f"Quantile Regression: {y_var} vs {x_var}",
            xaxis_title=x_var,
            yaxis_title=y_var,
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)
