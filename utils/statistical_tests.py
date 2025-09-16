
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import grangercausalitytests, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class StatisticalTestsModule:
    def __init__(self):
        self.test_categories = {
            "Unit Root Tests": self.unit_root_tests,
            "Cointegration Tests": self.cointegration_tests,
            "Diagnostic Tests": self.diagnostic_tests,
            "Causality Tests": self.causality_tests,
            "Structural Break Tests": self.structural_break_tests,
            "Normality Tests": self.normality_tests,
            "Heteroskedasticity Tests": self.heteroskedasticity_tests,
            "Autocorrelation Tests": self.autocorrelation_tests
        }

    def run(self, data):
        st.markdown('<div class="section-header">🧪 Statistical Tests</div>',
                    unsafe_allow_html=True)

        if data is None:
            st.warning("Please load data from the sidebar.")
            return

        # Test category selection
        test_category = st.selectbox(
            "Select Test Category:",
            list(self.test_categories.keys())
        )

        # Run selected test category
        self.test_categories[test_category](data)

    def unit_root_tests(self, data):
        st.markdown("### Unit Root Tests")
        st.info("Unit root tests check whether a time series is stationary or non-stationary.")

        # Select variable
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        test_var = st.selectbox("Select variable to test:", numeric_cols)

        # Test selection
        tests = st.multiselect(
            "Select tests to run:",
            ["Augmented Dickey-Fuller (ADF)", "KPSS", "Phillips-Perron"],
            default=["Augmented Dickey-Fuller (ADF)"]
        )

        if st.button("Run Unit Root Tests"):
            series = data[test_var].dropna()

            results = {}

            # ADF Test
            if "Augmented Dickey-Fuller (ADF)" in tests:
                st.markdown("#### Augmented Dickey-Fuller Test")

                # Test parameters
                col1, col2, col3 = st.columns(3)
                with col1:
                    regression = st.selectbox("Regression:", ['c', 'ct', 'ctt', 'n'])
                with col2:
                    autolag = st.selectbox("Auto lag:", ['AIC', 'BIC', 't-stat', None])
                with col3:
                    maxlag = st.number_input("Max lag:", 1, 20, 10)

                adf_result = adfuller(series, regression=regression, autolag=autolag)

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("ADF Statistic", f"{adf_result[0]:.4f}")
                    st.metric("P-value", f"{adf_result[1]:.4f}")
                    st.metric("Lags Used", f"{adf_result[2]}")
                    st.metric("Observations", f"{adf_result[3]}")

                with col2:
                    st.write("**Critical Values:**")
                    for key, value in adf_result[4].items():
                        st.write(f"{key}: {value:.4f}")

                if adf_result[1] <= 0.05:
                    st.success("✅ Reject H0: Series is stationary")
                else:
                    st.warning("⚠️ Fail to reject H0: Series has unit root")

                results['ADF'] = adf_result

            # KPSS Test
            if "KPSS" in tests:
                st.markdown("#### KPSS Test")

                regression_kpss = st.selectbox("KPSS Regression:", ['c', 'ct'])
                kpss_result = kpss(series, regression=regression_kpss)

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("KPSS Statistic", f"{kpss_result[0]:.4f}")
                    st.metric("P-value", f"{kpss_result[1]:.4f}")
                    st.metric("Lags Used", f"{kpss_result[2]}")

                with col2:
                    st.write("**Critical Values:**")
                    for key, value in kpss_result[3].items():
                        st.write(f"{key}: {value:.4f}")

                if kpss_result[1] > 0.05:
                    st.success("✅ Fail to reject H0: Series is stationary")
                else:
                    st.warning("⚠️ Reject H0: Series has unit root")

                results['KPSS'] = kpss_result

            # Phillips-Perron Test
            if "Phillips-Perron" in tests:
                st.markdown("#### Phillips-Perron Test")

                pp = PhillipsPerron(series)
                pp_result = pp.stat, pp.pvalue

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("PP Statistic", f"{pp_result[0]:.4f}")
                    st.metric("P-value", f"{pp_result[1]:.4f}")

                with col2:
                    if pp_result[1] <= 0.05:
                        st.success("✅ Reject H0: Series is stationary")
                    else:
                        st.warning("⚠️ Fail to reject H0: Series has unit root")

                results['PP'] = pp_result

            # Visualize series
            self.plot_series_with_tests(series, results)

    def cointegration_tests(self, data):
        st.markdown("### Cointegration Tests")
        st.info("Cointegration tests check for long-run relationships between non-stationary series.")

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            var1 = st.selectbox("First variable:", numeric_cols)
        with col2:
            var2 = st.selectbox("Second variable:",
                               [col for col in numeric_cols if col != var1])

        test_type = st.selectbox(
            "Select test:",
            ["Engle-Granger", "Johansen", "Phillips-Ouliaris"]
        )

        if st.button("Run Cointegration Test"):
            series1 = data[var1].dropna()
            series2 = data[var2].dropna()

            # Align series
            min_len = min(len(series1), len(series2))
            series1 = series1[:min_len]
            series2 = series2[:min_len]

            if test_type == "Engle-Granger":
                self.engle_granger_test(series1, series2, var1, var2)

            elif test_type == "Johansen":
                self.johansen_test(pd.DataFrame({var1: series1, var2: series2}))

    def diagnostic_tests(self, data):
        st.markdown("### Regression Diagnostic Tests")
        st.info("These tests check regression assumptions and model specification.")

        # Variable selection
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            y_var = st.selectbox("Dependent variable:", numeric_cols)
        with col2:
            x_vars = st.multiselect("Independent variables:",
                                   [col for col in numeric_cols if col != y_var])

        if not x_vars:
            st.warning("Please select at least one independent variable.")
            return

        # Run regression first
        import statsmodels.api as sm

        X = sm.add_constant(data[x_vars])
        y = data[y_var]

        # Remove NaN
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]

        model = sm.OLS(y, X).fit()

        # Display regression summary
        with st.expander("Regression Summary", expanded=False):
            st.text(str(model.summary()))

        # Diagnostic tests
        st.markdown("### Diagnostic Test Results")

        # Create tabs for different tests
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Autocorrelation", "Heteroskedasticity", "Normality", "Specification"]
        )

        with tab1:
            self.run_autocorrelation_tests(model)

        with tab2:
            self.run_heteroskedasticity_tests(model, X)

        with tab3:
            self.run_normality_tests(model)

        with tab4:
            self.run_specification_tests(model, X, y)

    def causality_tests(self, data):
        st.markdown("### Causality Tests")
        st.info("Test for causal relationships between time series variables.")

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            var1 = st.selectbox("Variable 1:", numeric_cols)
        with col2:
            var2 = st.selectbox("Variable 2:",
                              [col for col in numeric_cols if col != var1])

        max_lag = st.slider("Maximum lag to test:", 1, 20, 5)

        if st.button("Run Granger Causality Test"):
            # Prepare data
            test_data = data[[var1, var2]].dropna()

            # Run Granger causality test
            results = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)

            # Display results
            st.markdown("#### Granger Causality Test Results")
            st.write(f"Testing: Does {var2} Granger-cause {var1}?")

            # Create results table
            results_data = []
            for lag in range(1, max_lag + 1):
                ssr_ftest = results[lag][0]['ssr_ftest']
                ssr_chi2test = results[lag][0]['ssr_chi2test']

                results_data.append({
                    'Lag': lag,
                    'F-statistic': ssr_ftest[0],
                    'F p-value': ssr_ftest[1],
                    'Chi2-statistic': ssr_chi2test[0],
                    'Chi2 p-value': ssr_chi2test[1]
                })

            results_df = pd.DataFrame(results_data)

            # Highlight significant results
            def highlight_significant(val):
                if isinstance(val, float) and val < 0.05:
                    return 'background-color: lightgreen'
                return ''

            st.dataframe(
                results_df.style.applymap(
                    highlight_significant,
                    subset=['F p-value', 'Chi2 p-value']
                ).format({
                    'F-statistic': '{:.4f}',
                    'F p-value': '{:.4f}',
                    'Chi2-statistic': '{:.4f}',
                    'Chi2 p-value': '{:.4f}'
                }),
                use_container_width=True
            )

            # Plot p-values
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results_df['Lag'],
                y=results_df['F p-value'],
                mode='lines+markers',
                name='F-test p-value',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=results_df['Lag'],
                y=results_df['Chi2 p-value'],
                mode='lines+markers',
                name='Chi2-test p-value',
                line=dict(color='red')
            ))
            fig.add_hline(y=0.05, line_dash="dash", line_color="gray",
                         annotation_text="5% significance level")

            fig.update_layout(
                title="Granger Causality P-values by Lag",
                xaxis_title="Lag",
                yaxis_title="P-value",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

    def structural_break_tests(self, data):
        st.markdown("### Structural Break Tests")
        st.info("Test for structural breaks in time series data.")

        # Chow test implementation
        st.warning("Structural break tests (Chow, CUSUM, etc.) require specialized implementation.")

    def normality_tests(self, data):
        st.markdown("### Normality Tests")

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        test_var = st.selectbox("Select variable to test:", numeric_cols)

        if st.button("Run Normality Tests"):
            series = data[test_var].dropna()

            # Multiple normality tests
            col1, col2, col3 = st.columns(3)

            with col1:
                # Jarque-Bera test
                jb_result = jarque_bera(series)
                jb_stat = jb_result[0]
                jb_pval = jb_result[1]
                st.markdown("**Jarque-Bera Test**")
                st.metric("Statistic", f"{jb_stat:.4f}")
                st.metric("P-value", f"{jb_pval:.4f}")
                if jb_pval > 0.05:
                    st.success("Normal")
                else:
                    st.warning("Not Normal")

            with col2:
                # Shapiro-Wilk test
                sw_stat, sw_pval = stats.shapiro(series)
                st.markdown("**Shapiro-Wilk Test**")
                st.metric("Statistic", f"{sw_stat:.4f}")
                st.metric("P-value", f"{sw_pval:.4f}")
                if sw_pval > 0.05:
                    st.success("Normal")
                else:
                    st.warning("Not Normal")

            with col3:
                # Anderson-Darling test
                ad_result = stats.anderson(series)
                st.markdown("**Anderson-Darling Test**")
                st.metric("Statistic", f"{ad_result.statistic:.4f}")
                critical_5 = ad_result.critical_values[2]  # 5% level
                st.metric("Critical (5%)", f"{critical_5:.4f}")
                if ad_result.statistic < critical_5:
                    st.success("Normal")
                else:
                    st.warning("Not Normal")

            # Visualization
            self.plot_normality_diagnostics(series)

    def heteroskedasticity_tests(self, data):
        st.markdown("### Heteroskedasticity Tests")
        st.info("Heteroskedasticity refers to situations where the variance of the residuals is unequal over a range of measured values [[1]]. These tests check for constant variance assumption in regression models.")

        # Variable selection
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            y_var = st.selectbox("Dependent variable:", numeric_cols)
        with col2:
            x_vars = st.multiselect("Independent variables:",
                                   [col for col in numeric_cols if col != y_var])

        if not x_vars:
            st.warning("Please select at least one independent variable.")
            return

        if st.button("Run Heteroskedasticity Tests"):
            import statsmodels.api as sm

            # Prepare data
            X = sm.add_constant(data[x_vars])
            y = data[y_var]

            # Remove NaN
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]

            # Fit model
            model = sm.OLS(y, X).fit()

            # Run multiple tests
            col1, col2, col3 = st.columns(3)

            with col1:
                # Breusch-Pagan test
                st.markdown("**Breusch-Pagan Test**")
                bp_stat, bp_pval, _, _ = het_breuschpagan(model.resid, X)
                st.metric("LM Statistic", f"{bp_stat:.4f}")
                st.metric("P-value", f"{bp_pval:.4f}")
                if bp_pval > 0.05:
                    st.success("✅ Homoskedastic")
                else:
                    st.warning("⚠️ Heteroskedastic")

            with col2:
                # White test
                st.markdown("**White Test**")
                white_stat, white_pval, _, _ = het_white(model.resid, X)
                st.metric("LM Statistic", f"{white_stat:.4f}")
                st.metric("P-value", f"{white_pval:.4f}")
                if white_pval > 0.05:
                    st.success("✅ Homoskedastic")
                else:
                    st.warning("⚠️ Heteroskedastic")

            with col3:
                # Goldfeld-Quandt test
                st.markdown("**Goldfeld-Quandt Test**")
                from statsmodels.stats.diagnostic import het_goldfeldquandt
                gq_stat, gq_pval, _ = het_goldfeldquandt(model.resid, X)
                st.metric("F Statistic", f"{gq_stat:.4f}")
                st.metric("P-value", f"{gq_pval:.4f}")
                if gq_pval > 0.05:
                    st.success("✅ Homoskedastic")
                else:
                    st.warning("⚠️ Heteroskedastic")

            # Visualization
            self.plot_heteroskedasticity_diagnostics(model, X)

    def autocorrelation_tests(self, data):
        st.markdown("### Autocorrelation Tests")
        st.info("Test for serial correlation in residuals.")

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        test_var = st.selectbox("Select variable to test:", numeric_cols)

        if st.button("Run Autocorrelation Tests"):
            series = data[test_var].dropna()

            # Durbin-Watson test
            dw_stat = durbin_watson(series)

            # Ljung-Box test
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(series, lags=10, return_df=True)

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Durbin-Watson Test**")
                st.metric("DW Statistic", f"{dw_stat:.4f}")
                if 1.5 < dw_stat < 2.5:
                    st.success("✅ No autocorrelation")
                else:
                    st.warning("⚠️ Autocorrelation detected")

            with col2:
                st.markdown("**Ljung-Box Test**")
                st.dataframe(lb_result[['lb_stat', 'lb_pvalue']].style.format({
                    'lb_stat': '{:.4f}',
                    'lb_pvalue': '{:.4f}'
                }), use_container_width=True)

            # ACF/PACF plots
            self.plot_acf_pacf(series)

    def plot_series_with_tests(self, series, test_results):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Time Series', 'First Difference', 
                          'ACF', 'PACF')
        )

        # Original series
        fig.add_trace(go.Scatter(y=series.values, mode='lines',
                                line=dict(color='blue')),
                     row=1, col=1)

        # First difference
        diff_series = series.diff().dropna()
        fig.add_trace(go.Scatter(y=diff_series.values, mode='lines',
                                line=dict(color='green')),
                     row=1, col=2)

        # ACF
        from statsmodels.tsa.stattools import acf, pacf
        acf_values = acf(series, nlags=20)
        fig.add_trace(go.Bar(y=acf_values, marker=dict(color='orange')),
                     row=2, col=1)

        # PACF
        # Calculate appropriate max lag (50% of sample size)
        max_lag = min(20, len(series) // 2 - 1)
        if max_lag <= 0:
            max_lag = 1
        pacf_values = pacf(series, nlags=max_lag)
        fig.add_trace(go.Bar(y=pacf_values, marker=dict(color='purple')),
                     row=2, col=2)

        fig.update_layout(height=600, showlegend=False,
                        title="Unit Root Test Diagnostics")
        st.plotly_chart(fig, use_container_width=True)

    def engle_granger_test(self, series1, series2, var1, var2):
        st.markdown("#### Engle-Granger Cointegration Test")

        # Step 1: Check if both series are I(1)
        adf1 = adfuller(series1)
        adf2 = adfuller(series2)

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**{var1} Unit Root Test**")
            st.write(f"ADF Statistic: {adf1[0]:.4f}")
            st.write(f"P-value: {adf1[1]:.4f}")

        with col2:
            st.write(f"**{var2} Unit Root Test**")
            st.write(f"ADF Statistic: {adf2[0]:.4f}")
            st.write(f"P-value: {adf2[1]:.4f}")

        # Step 2: Run cointegration test
        result = coint(series1, series2)

        st.markdown("**Cointegration Test Result**")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Test Statistic", f"{result[0]:.4f}")
        with col2:
            st.metric("P-value", f"{result[1]:.4f}")
        with col3:
            if result[1] < 0.05:
                st.success("✅ Series are cointegrated")
            else:
                st.warning("⚠️ No cointegration found")

        # Critical values
        st.write("**Critical Values:**")
        crit_vals = result[2]
        st.write(f"1%: {crit_vals[0]:.4f}, 5%: {crit_vals[1]:.4f}, 10%: {crit_vals[2]:.4f}")

    def johansen_test(self, data):
        st.markdown("#### Johansen Cointegration Test")

        # Run Johansen test
        result = coint_johansen(data, det_order=0, k_ar_diff=1)

        # Display results
        st.write("**Trace Statistics:**")
        trace_df = pd.DataFrame({
            'r': range(len(result.lr1)),
            'Trace Statistic': result.lr1,
            'Critical Value (5%)': result.cvt[:, 1],
            'Cointegrated': result.lr1 > result.cvt[:, 1]
        })
        st.dataframe(trace_df, use_container_width=True)

        st.write("**Maximum Eigenvalue Statistics:**")
        max_eig_df = pd.DataFrame({
            'r': range(len(result.lr2)),
            'Max Eigenvalue': result.lr2,
            'Critical Value (5%)': result.cvm[:, 1],
            'Cointegrated': result.lr2 > result.cvm[:, 1]
        })
        st.dataframe(max_eig_df, use_container_width=True)

        # Number of cointegrating relationships
        n_coint = sum(result.lr1 > result.cvt[:, 1])
        st.info(f"Number of cointegrating relationships: {n_coint}")

    def run_autocorrelation_tests(self, model):
        st.markdown("#### Autocorrelation Tests")

        residuals = model.resid

        # Durbin-Watson
        dw_stat = durbin_watson(residuals)

        # Breusch-Godfrey
        from statsmodels.stats.diagnostic import acorr_breusch_godfrey
        bg_result = acorr_breusch_godfrey(model, nlags=4)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Durbin-Watson Test**")
            st.metric("DW Statistic", f"{dw_stat:.4f}")
            if 1.5 < dw_stat < 2.5:
                st.success("✅ No autocorrelation")
            else:
                st.warning("⚠️ Potential autocorrelation")

        with col2:
            st.markdown("**Breusch-Godfrey Test**")
            st.metric("LM Statistic", f"{bg_result[0]:.4f}")
            st.metric("P-value", f"{bg_result[1]:.4f}")
            if bg_result[1] > 0.05:
                st.success("✅ No autocorrelation")
            else:
                st.warning("⚠️ Autocorrelation detected")

    def run_heteroskedasticity_tests(self, model, X):
        st.markdown("#### Heteroskedasticity Tests")

        residuals = model.resid

        # Multiple tests
        tests_results = []

        # Breusch-Pagan
        bp_stat, bp_pval, _, _ = het_breuschpagan(residuals, X)
        tests_results.append(("Breusch-Pagan", bp_stat, bp_pval))

        # White
        white_stat, white_pval, _, _ = het_white(residuals, X)
        tests_results.append(("White", white_stat, white_pval))

        # Display results
        for test_name, stat, pval in tests_results:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**{test_name} Test**")
            with col2:
                st.metric("Statistic", f"{stat:.4f}")
            with col3:
                st.metric("P-value", f"{pval:.4f}")
                if pval > 0.05:
                    st.success("Homoskedastic")
                else:
                    st.warning("Heteroskedastic")

    def run_normality_tests(self, model):
        st.markdown("#### Normality Tests")

        residuals = model.resid

        # Multiple normality tests
        col1, col2, col3 = st.columns(3)

        with col1:
            jb_result = jarque_bera(residuals)
            jb_stat = jb_result[0]
            jb_pval = jb_result[1]
            st.markdown("**Jarque-Bera**")
            st.metric("Statistic", f"{jb_stat:.4f}")
            st.metric("P-value", f"{jb_pval:.4f}")
            if jb_pval > 0.05:
                st.success("Normal")
            else:
                st.warning("Not Normal")

        with col2:
            sw_stat, sw_pval = stats.shapiro(residuals)
            st.markdown("**Shapiro-Wilk**")
            st.metric("Statistic", f"{sw_stat:.4f}")
            st.metric("P-value", f"{sw_pval:.4f}")
            if sw_pval > 0.05:
                st.success("Normal")
            else:
                st.warning("Not Normal")

        with col3:
            # Skewness and Kurtosis
            skew = stats.skew(residuals)
            kurt = stats.kurtosis(residuals)
            st.markdown("**Moments**")
            st.metric("Skewness", f"{skew:.4f}")
            st.metric("Kurtosis", f"{kurt:.4f}")

    def run_specification_tests(self, model, X, y):
        st.markdown("#### Specification Tests")

        # RESET test
        from statsmodels.stats.diagnostic import linear_reset
        # Ensure model.fittedvalues is a numpy array to avoid indexing issues
        if hasattr(model, "fittedvalues"):
            original_fittedvalues = model.fittedvalues
            model.fittedvalues = np.array(model.fittedvalues)
            
        reset_result = linear_reset(model, power=3, use_f=True)
        
        # Restore original fittedvalues
        if hasattr(model, "fittedvalues"):
            model.fittedvalues = original_fittedvalues

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Ramsey RESET Test**")
            st.metric("F-statistic", f"{reset_result.statistic:.4f}")
            st.metric("P-value", f"{reset_result.pvalue:.4f}")
            if reset_result.pvalue > 0.05:
                st.success("✅ No specification error")
            else:
                st.warning("⚠️ Potential specification error")

        with col2:
            # VIF for multicollinearity
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            st.markdown("**Variance Inflation Factors**")
            if X.shape[1] > 1:  # More than just constant
                vif_data = pd.DataFrame()
                vif_data["Variable"] = X.columns
                vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                                   for i in range(X.shape[1])]
                vif_data = vif_data[vif_data["Variable"] != "const"]

                st.dataframe(vif_data.style.format({'VIF': '{:.2f}'}), 
                           use_container_width=True)

                if any(vif_data["VIF"] > 10):
                    st.warning("⚠️ High multicollinearity detected (VIF > 10)")
                else:
                    st.success("✅ No severe multicollinearity")

    def plot_normality_diagnostics(self, series):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Histogram with Normal Curve', 'Q-Q Plot', 
                          'Box Plot', 'Kernel Density')
        )

        # Histogram with normal curve
        fig.add_trace(go.Histogram(x=series, nbinsx=30, name='Data', 
                                  histnorm='probability density'),
                     row=1, col=1)

        # Add normal distribution curve
        x_range = np.linspace(series.min(), series.max(), 100)
        normal_curve = stats.norm.pdf(x_range, series.mean(), series.std())
        fig.add_trace(go.Scatter(x=x_range, y=normal_curve, 
                                mode='lines', name='Normal',
                                line=dict(color='red')),
                     row=1, col=1)

        # Q-Q plot
        qq_data = stats.probplot(series, dist="norm", plot=None)
        fig.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[0][1], 
                                mode='markers', name='Q-Q'),
                     row=1, col=2)
        fig.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[0][0], 
                                mode='lines', line=dict(color='red')),
                     row=1, col=2)

        # Box plot
        fig.add_trace(go.Box(y=series, name='Box Plot'),
                     row=2, col=1)

        # Kernel density
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(series)
        x_kde = np.linspace(series.min(), series.max(), 100)
        y_kde = kde(x_kde)
        fig.add_trace(go.Scatter(x=x_kde, y=y_kde, mode='lines', 
                                name='KDE', line=dict(color='green')),
                     row=2, col=2)

        fig.update_layout(height=600, showlegend=False,
                        title="Normality Diagnostics")
        st.plotly_chart(fig, use_container_width=True)

    def plot_heteroskedasticity_diagnostics(self, model, X):
        residuals = model.resid
        fitted = model.fittedvalues

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residuals vs Fitted', 'Scale-Location', 
                          'Residuals vs Leverage', 'Cook\'s Distance')
        )

        # Residuals vs Fitted
        fig.add_trace(go.Scatter(x=fitted, y=residuals,
                                mode='markers', marker=dict(size=5)),
                     row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

        # Scale-Location plot
        standardized_residuals = residuals / np.std(residuals)
        sqrt_standardized_residuals = np.sqrt(np.abs(standardized_residuals))
        fig.add_trace(go.Scatter(x=fitted, y=sqrt_standardized_residuals,
                                mode='markers', marker=dict(size=5)),
                     row=1, col=2)

        # Residuals vs Leverage
        from statsmodels.stats.outliers_influence import OLSInfluence
        influence = OLSInfluence(model)
        leverage = influence.hat_matrix_diag
        fig.add_trace(go.Scatter(x=leverage, y=standardized_residuals,
                                mode='markers', marker=dict(size=5)),
                     row=2, col=1)

        # Cook's Distance
        cooks_d = influence.cooks_distance[0]
        fig.add_trace(go.Bar(y=cooks_d), row=2, col=2)

        fig.update_layout(height=600, showlegend=False,
                        title="Heteroskedasticity Diagnostics")
        st.plotly_chart(fig, use_container_width=True)

    def plot_acf_pacf(self, series):
        from statsmodels.tsa.stattools import acf, pacf

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Autocorrelation Function', 'Partial Autocorrelation Function')
        )

        # ACF
        acf_values = acf(series, nlags=40)
        fig.add_trace(go.Bar(y=acf_values, marker=dict(color='blue')),
                     row=1, col=1)

        # Add confidence bands
        n = len(series)
        confidence = 1.96 / np.sqrt(n)
        fig.add_hline(y=confidence, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=-confidence, line_dash="dash", line_color="red", row=1, col=1)

        # PACF
        # Calculate appropriate max lag (50% of sample size)
        max_lag = min(40, len(series) // 2 - 1)
        if max_lag <= 0:
            max_lag = 1
        pacf_values = pacf(series, nlags=max_lag)
        fig.add_trace(go.Bar(y=pacf_values, marker=dict(color='green')),
                     row=1, col=2)

        fig.add_hline(y=confidence, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_hline(y=-confidence, line_dash="dash", line_color="red", row=1, col=2)

        fig.update_layout(height=400, showlegend=False,
                        title="ACF and PACF Plots")
        fig.update_xaxes(title_text="Lag", row=1, col=1)
        fig.update_xaxes(title_text="Lag", row=1, col=2)
        fig.update_yaxes(title_text="ACF", row=1, col=1) 
        fig.update_yaxes(title_text="PACF", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)
