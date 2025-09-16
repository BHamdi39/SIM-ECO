
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from linearmodels import PanelOLS, RandomEffects, PooledOLS
from linearmodels.panel import compare
import statsmodels.api as sm
from scipy import stats

class PanelDataModule:
    def __init__(self):
        self.models = {
            "Fixed Effects": self.fixed_effects,
            "Random Effects": self.random_effects,
            "Pooled OLS": self.pooled_ols,
            "Difference-in-Differences": self.diff_in_diff,
            "Instrumental Variables": self.instrumental_variables
        }

    def run(self, data):
        st.markdown('<div class="section-header">📋 Panel Data Models</div>',
                    unsafe_allow_html=True)

        if data is None:
            st.warning("Please load panel data from the sidebar.")
            st.info("Panel data should have entity and time identifiers.")
            return

        # Model selection
        model_type = st.selectbox(
            "Select Panel Data Model:",
            list(self.models.keys())
        )

        # Panel data configuration
        st.markdown("### Panel Data Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            entity_col = st.selectbox(
                "Entity/Individual Column:",
                data.columns.tolist()
            )

        with col2:
            time_col = st.selectbox(
                "Time Period Column:",
                [col for col in data.columns if col != entity_col]
            )

        with col3:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            y_var = st.selectbox(
                "Dependent Variable:",
                [col for col in numeric_cols if col not in [entity_col, time_col]]
            )

        # Independent variables
        x_vars = st.multiselect(
            "Independent Variables:",
            [col for col in numeric_cols if col not in [entity_col, time_col, y_var]]
        )

        if not x_vars:
            st.warning("Please select at least one independent variable.")
            return

        # Set panel index
        try:
            data = data.set_index([entity_col, time_col])

            # Display panel structure
            with st.expander("📊 Panel Data Structure", expanded=True):
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Entities", data.index.get_level_values(0).nunique())
                with col2:
                    st.metric("Time Periods", data.index.get_level_values(1).nunique())
                with col3:
                    st.metric("Total Observations", len(data))
                with col4:
                    st.metric("Variables", len(x_vars) + 1)

                # Balance check
                balance = data.groupby(level=0).size()
                if balance.nunique() == 1:
                    st.success("✅ Panel is balanced")
                else:
                    st.warning("⚠️ Panel is unbalanced")
                    st.write(f"Min obs per entity: {balance.min()}, Max: {balance.max()}")

        except Exception as e:
            st.error(f"Error setting panel index: {str(e)}")
            return

        # Model parameters
        params = self.get_panel_parameters(model_type)

        # Run model
        if st.button("🚀 Run Panel Model", type="primary"):
            self.models[model_type](data, y_var, x_vars, params)

    def get_panel_parameters(self, model_type):
        params = {}

        st.markdown("### Model Parameters")

        if model_type == "Fixed Effects":
            col1, col2 = st.columns(2)
            with col1:
                params['entity_effects'] = st.checkbox("Entity Effects", True)
                params['time_effects'] = st.checkbox("Time Effects", False)
            with col2:
                params['drop_absorbed'] = st.checkbox("Drop Absorbed", True)
                params['use_lsdv'] = st.checkbox("Use LSDV", False)

        elif model_type == "Random Effects":
            params['small_sample'] = st.checkbox("Small Sample Correction", False)
            params['cov_type'] = st.selectbox(
                "Covariance Type:",
                ['unadjusted', 'homoskedastic', 'robust', 'clustered']
            )

        elif model_type == "Difference-in-Differences":
            col1, col2 = st.columns(2)
            with col1:
                params['treatment_col'] = st.text_input("Treatment Column Name:")
                params['post_col'] = st.text_input("Post-Period Column Name:")
            with col2:
                params['parallel_trends'] = st.checkbox("Test Parallel Trends", True)
                params['placebo_test'] = st.checkbox("Run Placebo Test", False)

        elif model_type == "Instrumental Variables":
            params['instruments'] = st.multiselect(
                "Select Instrumental Variables:",
                data.columns.tolist() if 'data' in locals() else []
            )
            params['test_weak'] = st.checkbox("Test for Weak Instruments", True)
            params['test_overid'] = st.checkbox("Test Overidentification", True)

        return params

    def fixed_effects(self, data, y_var, x_vars, params):
        try:
            # Prepare data
            y = data[y_var]
            X = data[x_vars]

            # Add constant
            X = sm.add_constant(X)

            # Fit Fixed Effects model
            model = PanelOLS(y, X, entity_effects=params.get('entity_effects', True),
                           time_effects=params.get('time_effects', False))
            fitted_model = model.fit(cov_type='clustered', cluster_entity=True)

            # Display results
            st.success("✅ Fixed Effects model fitted successfully!")

            # Model summary
            with st.expander("Model Summary", expanded=True):
                st.text(str(fitted_model))

            # Coefficients and statistics
            self.display_panel_results(fitted_model, x_vars)

            # Entity fixed effects
            if params.get('entity_effects', True):
                self.plot_entity_effects(fitted_model)

            # Diagnostic tests
            self.panel_diagnostics(fitted_model)

        except Exception as e:
            st.error(f"Error fitting Fixed Effects model: {str(e)}")

    def random_effects(self, data, y_var, x_vars, params):
        try:
            # Prepare data
            y = data[y_var]
            X = data[x_vars]
            X = sm.add_constant(X)

            # Fit Random Effects model
            model = RandomEffects(y, X)
            fitted_model = model.fit(cov_type=params.get('cov_type', 'unadjusted'))

            # Display results
            st.success("✅ Random Effects model fitted successfully!")

            # Model summary
            with st.expander("Model Summary", expanded=True):
                st.text(str(fitted_model))

            # Results display
            self.display_panel_results(fitted_model, x_vars)

            # Hausman test
            self.hausman_test(data, y_var, x_vars)

        except Exception as e:
            st.error(f"Error fitting Random Effects model: {str(e)}")

    def pooled_ols(self, data, y_var, x_vars, params):
        try:
            # Prepare data
            y = data[y_var]
            X = data[x_vars]
            X = sm.add_constant(X)

            # Fit Pooled OLS model
            model = PooledOLS(y, X)
            fitted_model = model.fit(cov_type='clustered', cluster_entity=True)

            # Display results
            st.success("✅ Pooled OLS model fitted successfully!")

            # Model summary
            with st.expander("Model Summary", expanded=True):
                st.text(str(fitted_model))

            # Results display
            self.display_panel_results(fitted_model, x_vars)

        except Exception as e:
            st.error(f"Error fitting Pooled OLS model: {str(e)}")

    def diff_in_diff(self, data, y_var, x_vars, params):
        try:
            st.info("Difference-in-Differences analysis")

            # Check for treatment and post columns
            if not params.get('treatment_col') or not params.get('post_col'):
                st.warning("Please specify treatment and post-period columns in parameters.")
                return

            # Create DiD interaction term
            data['did_interaction'] = data[params['treatment_col']] * data[params['post_col']]

            # Run regression with DiD specification
            X = data[x_vars + [params['treatment_col'], params['post_col'], 'did_interaction']]
            y = data[y_var]
            X = sm.add_constant(X)

            model = sm.OLS(y, X)
            fitted_model = model.fit(cov_type='HC1')

            # Display results
            st.success("✅ Difference-in-Differences model fitted successfully!")

            # DiD coefficient
            did_coef = fitted_model.params['did_interaction']
            did_se = fitted_model.bse['did_interaction']
            did_pval = fitted_model.pvalues['did_interaction']

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("DiD Coefficient", f"{did_coef:.4f}")
            with col2:
                st.metric("Standard Error", f"{did_se:.4f}")
            with col3:
                st.metric("P-value", f"{did_pval:.4f}")
                if did_pval < 0.05:
                    st.success("Significant treatment effect")
                else:
                    st.warning("No significant treatment effect")

            # Parallel trends visualization
            if params.get('parallel_trends', True):
                self.plot_parallel_trends(data, y_var, params['treatment_col'], params['post_col'])

        except Exception as e:
            st.error(f"Error fitting DiD model: {str(e)}")

    def instrumental_variables(self, data, y_var, x_vars, params):
        try:
            st.info("Instrumental Variables estimation")

            if not params.get('instruments'):
                st.warning("Please select instrumental variables.")
                return

            # Implementation would use linearmodels IV2SLS
            st.info("IV estimation would be implemented here using linearmodels.IV2SLS")

        except Exception as e:
            st.error(f"Error fitting IV model: {str(e)}")

    def display_panel_results(self, fitted_model, x_vars):
        st.markdown("### Model Results")

        # Coefficients table
        coef_df = pd.DataFrame({
            'Variable': ['const'] + x_vars,
            'Coefficient': fitted_model.params.values,
            'Std Error': fitted_model.std_errors.values,
            't-statistic': fitted_model.tstats.values,
            'P-value': fitted_model.pvalues.values
        })

        # Format p-values
        coef_df['Significance'] = coef_df['P-value'].apply(
            lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        )

        st.dataframe(coef_df.style.format({
            'Coefficient': '{:.4f}',
            'Std Error': '{:.4f}',
            't-statistic': '{:.4f}',
            'P-value': '{:.4f}'
        }), use_container_width=True)

        # Model statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("R-squared", f"{fitted_model.rsquared:.4f}")
        with col2:
            st.metric("Within R²", f"{fitted_model.rsquared_within:.4f}"
                      if hasattr(fitted_model, 'rsquared_within') else "N/A")
        with col3:
            st.metric("F-statistic", f"{fitted_model.f_statistic.stat:.4f}"
                     if hasattr(fitted_model, 'f_statistic') else "N/A")
        with col4:
            st.metric("Observations", f"{fitted_model.nobs:.0f}")

        # Coefficient plot
        fig = go.Figure()

        # Add coefficients with confidence intervals
        for i, var in enumerate(x_vars):
            coef = fitted_model.params[var]
            se = fitted_model.std_errors[var]
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se

            fig.add_trace(go.Scatter(
                x=[ci_lower, ci_upper],
                y=[var, var],
                mode='lines',
                line=dict(color='lightblue', width=3),
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=[coef],
                y=[var],
                mode='markers',
                marker=dict(size=10, color='blue'),
                showlegend=False
            ))

        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title="Coefficient Plot with 95% Confidence Intervals",
            xaxis_title="Coefficient Value",
            yaxis_title="Variable",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def plot_entity_effects(self, fitted_model):
        st.markdown("### Entity Fixed Effects")

        if hasattr(fitted_model, 'estimated_effects'):
            effects = fitted_model.estimated_effects.entity_effects

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=effects.index,
                y=effects.values.flatten(),
                marker=dict(color=effects.values.flatten(), colorscale='RdBu')
            ))

            fig.update_layout(
                title="Entity Fixed Effects",
                xaxis_title="Entity",
                yaxis_title="Fixed Effect",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

    def hausman_test(self, data, y_var, x_vars):
        st.markdown("### Hausman Test")

        try:
            # Fit both FE and RE models
            y = data[y_var]
            X = sm.add_constant(data[x_vars])

            fe_model = PanelOLS(y, X, entity_effects=True).fit()
            re_model = RandomEffects(y, X).fit()

            # Calculate Hausman statistic
            b_fe = fe_model.params[x_vars]
            b_re = re_model.params[x_vars]
            var_fe = fe_model.cov[x_vars].loc[x_vars]
            var_re = re_model.cov[x_vars].loc[x_vars]

            diff = b_fe - b_re
            var_diff = var_fe - var_re

            hausman_stat = diff.T @ np.linalg.inv(var_diff) @ diff
            p_value = 1 - stats.chi2.cdf(hausman_stat, len(x_vars))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Hausman Statistic", f"{hausman_stat:.4f}")
            with col2:
                st.metric("P-value", f"{p_value:.4f}")

            if p_value < 0.05:
                st.info("📊 Reject null hypothesis: Use Fixed Effects")
            else:
                st.info("📊 Fail to reject null: Random Effects is consistent")

        except Exception as e:
            st.warning(f"Could not perform Hausman test: {str(e)}")

    def panel_diagnostics(self, fitted_model):
        st.markdown("### Panel Diagnostics")

        residuals = fitted_model.resids

        # Create diagnostic plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residuals Over Time', 'Residuals by Entity', 
                          'Residuals Distribution', 'Residuals vs Fitted')
        )

        # Residuals over time
        fig.add_trace(go.Scatter(
            y=residuals.values, mode='lines',
            line=dict(color='blue', width=1)
        ), row=1, col=1)

        # Residuals by entity
        entity_means = residuals.groupby(level=0).mean()
        fig.add_trace(go.Bar(
            x=entity_means.index,
            y=entity_means.values,
            marker=dict(color='green')
        ), row=1, col=2)

        # Residuals distribution
        fig.add_trace(go.Histogram(
            x=residuals.values, nbinsx=30,
            marker=dict(color='orange')
        ), row=2, col=1)

        # Residuals vs fitted
        fitted_values = fitted_model.fitted_values
        fig.add_trace(go.Scatter(
            x=fitted_values.values, y=residuals.values,
            mode='markers', marker=dict(size=3, color='purple')
        ), row=2, col=2)

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    def plot_parallel_trends(self, data, y_var, treatment_col, post_col):
        st.markdown("### Parallel Trends Assumption")

        # Group by treatment and time
        treated = data[data[treatment_col] == 1].groupby(level=1)[y_var].mean()
        control = data[data[treatment_col] == 0].groupby(level=1)[y_var].mean()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=treated.index, y=treated.values,
            mode='lines+markers', name='Treated',
            line=dict(color='red', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=control.index, y=control.values,
            mode='lines+markers', name='Control',
            line=dict(color='blue', width=2)
        ))

        # Add vertical line at treatment time
        treatment_time = data[data[post_col] == 1].index.get_level_values(1).min()
        fig.add_vline(x=treatment_time, line_dash="dash", line_color="gray",
                     annotation_text="Treatment Start")

        fig.update_layout(
            title="Parallel Trends Test",
            xaxis_title="Time Period",
            yaxis_title=y_var,
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)
