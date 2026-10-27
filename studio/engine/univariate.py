import numpy as np
import pandas as pd
from datetime import datetime
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings('ignore')

def safe_float(val, default=0.0):
    try:
        if val is None or np.isnan(val) or np.isinf(val):
            return default
        return round(float(val), 4)
    except:
        return default

def clean_and_prepare_series(df, date_col, target_col, freq='MS'):
    """
    Cleans date column, sorts chronologically, handles duplicates,
    and sets datetime index.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.sort_values(by=date_col)
    df = df.drop_duplicates(subset=[date_col], keep='first')
    
    # Coerce target to numeric
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col])
    
    df.set_index(date_col, inplace=True)
    return df

def analyze_univariate_series(df, date_col, target_col, freq='MS', seasonal_period=12, holdout_size=12, forecast_horizon=12):
    """
    Full pipeline answering Question 1 to Question 7 for Mode A.
    """
    df_clean = clean_and_prepare_series(df, date_col, target_col, freq)
    series = df_clean[target_col].astype(float)
    n = len(series)
    
    if n < 15:
        raise ValueError(f"Dataset has only {n} valid observations. At least 15 observations are required for meaningful time series modeling.")
    
    if holdout_size >= n - 5:
        holdout_size = max(2, n // 5)
    
    # Frequency normalization
    try:
        inferred = pd.infer_freq(series.index)
        if inferred:
            series = series.asfreq(inferred).interpolate(method='linear')
    except:
        pass
    
    dates_str = [d.strftime('%Y-%m-%d') for d in series.index]
    values_list = [safe_float(v) for v in series.values]
    
    # =========================================================================
    # QUESTION 1: What does my data look like?
    # =========================================================================
    n_points = len(series)
    start_date = series.index[0].strftime('%b %Y') if hasattr(series.index[0], 'strftime') else str(series.index[0])
    end_date = series.index[-1].strftime('%b %Y') if hasattr(series.index[-1], 'strftime') else str(series.index[-1])
    
    # Check for gaps
    date_diffs = pd.Series(series.index).diff().dropna()
    mode_diff = date_diffs.mode()[0] if not date_diffs.empty else None
    gaps_count = 0
    if mode_diff:
        gaps_count = int((date_diffs > (mode_diff * 1.5)).sum())
    
    mean_val = safe_float(series.mean())
    std_val = safe_float(series.std())
    min_val = safe_float(series.min())
    q25_val = safe_float(series.quantile(0.25))
    median_val = safe_float(series.median())
    q75_val = safe_float(series.quantile(0.75))
    max_val = safe_float(series.max())
    cv_val = safe_float(std_val / mean_val if mean_val != 0 else 0)
    skew_val = safe_float(series.skew())
    
    q1_data = {
        'n_points': n_points,
        'start_date': start_date,
        'end_date': end_date,
        'gaps_count': gaps_count,
        'summary_verdict': f"{n_points} observations from {start_date} to {end_date}, {gaps_count} irregular gap(s).",
        'stats': {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'q25': q25_val,
            'median': median_val,
            'q75': q75_val,
            'max': max_val,
            'cv': cv_val,
            'skewness': skew_val
        },
        'chart_data': {
            'dates': dates_str,
            'values': values_list
        }
    }
    
    # =========================================================================
    # QUESTION 2: What is it made of? (Decomposition)
    # =========================================================================
    effective_period = min(seasonal_period, max(2, n // 2))
    decomp_success = False
    decomp_dates = []
    trend_vals = []
    seasonal_vals = []
    resid_vals = []
    trend_desc = "Stable level"
    season_desc = "No dominant seasonal pattern"
    strength_trend = 0.0
    strength_seasonal = 0.0
    peak_period_desc = ""
    
    if n >= 2 * effective_period:
        try:
            decomp = seasonal_decompose(series, model='additive', period=effective_period, extrapolate_trend='freq')
            decomp_success = True
            decomp_dates = dates_str
            trend_vals = [safe_float(v) for v in decomp.trend]
            seasonal_vals = [safe_float(v) for v in decomp.seasonal]
            resid_vals = [safe_float(v) for v in decomp.resid]
            
            # Trend strength calculation
            var_resid = np.nanvar(decomp.resid)
            var_trend_resid = np.nanvar(decomp.trend + decomp.resid)
            strength_trend = safe_float(max(0.0, 1.0 - (var_resid / var_trend_resid if var_trend_resid > 0 else 1.0)))
            
            # Seasonal strength calculation
            var_seas_resid = np.nanvar(decomp.seasonal + decomp.resid)
            strength_seasonal = safe_float(max(0.0, 1.0 - (var_resid / var_seas_resid if var_seas_resid > 0 else 1.0)))
            
            # Trend direction
            trend_change = trend_vals[-1] - trend_vals[0]
            if trend_change > (0.05 * std_val * np.sqrt(n)):
                trend_desc = "Prominent rising trend"
            elif trend_change < -(0.05 * std_val * np.sqrt(n)):
                trend_desc = "Prominent downward trend"
            else:
                trend_desc = "Relatively flat trend"
                
            # Seasonal cycle analysis
            seas_pattern = seasonal_vals[:effective_period]
            peak_idx = int(np.argmax(seas_pattern))
            trough_idx = int(np.argmin(seas_pattern))
            
            # Month mapping if monthly
            if effective_period == 12:
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                peak_month = month_names[peak_idx % 12]
                trough_month = month_names[trough_idx % 12]
                season_desc = f"Strong yearly cycle (peaks around {peak_month}, troughs around {trough_month})"
                peak_period_desc = f"Peak in {peak_month}, low in {trough_month}"
            else:
                season_desc = f"Cyclic pattern repeating every {effective_period} steps (peak at phase +{peak_idx+1})"
                peak_period_desc = f"Peak phase at index {peak_idx+1}"
        except Exception as e:
            decomp_success = False

    q2_data = {
        'success': decomp_success,
        'period': effective_period,
        'trend_desc': trend_desc,
        'season_desc': season_desc,
        'strength_trend': strength_trend,
        'strength_seasonal': strength_seasonal,
        'peak_desc': peak_period_desc,
        'chart_data': {
            'dates': decomp_dates,
            'trend': trend_vals,
            'seasonal': seasonal_vals,
            'residual': resid_vals
        }
    }
    
    # =========================================================================
    # QUESTION 3: Is it stable or drifting? (ADF Stationarity & Differencing)
    # =========================================================================
    raw_adf = adfuller(series.dropna())
    raw_stat = safe_float(raw_adf[0])
    raw_pval = safe_float(raw_adf[1])
    raw_crit = {
        'p1': safe_float(raw_adf[4].get('1%')),
        'p5': safe_float(raw_adf[4].get('5%')),
        'p10': safe_float(raw_adf[4].get('10%')),
    }
    raw_is_stationary = bool(raw_pval < 0.05)
    
    # First difference
    diff_series = series.diff().dropna()
    diff_adf = adfuller(diff_series.dropna())
    diff_stat = safe_float(diff_adf[0])
    diff_pval = safe_float(diff_adf[1])
    diff_crit = {
        'p1': safe_float(diff_adf[4].get('1%')),
        'p5': safe_float(diff_adf[4].get('5%')),
        'p10': safe_float(diff_adf[4].get('10%')),
    }
    diff_is_stationary = bool(diff_pval < 0.05)
    
    # Recommended d and D
    recommended_d = 0 if raw_is_stationary else (1 if diff_is_stationary else 2)
    recommended_D = 1 if (strength_seasonal > 0.4 and n > 2 * effective_period) else 0
    
    if raw_is_stationary:
        stationarity_verdict = f"Raw series is stationary (ADF p = {raw_pval:.3f} < 0.05). No differencing strictly required (d=0)."
    elif diff_is_stationary:
        stationarity_verdict = f"Raw series is non-stationary (p = {raw_pval:.3f}). Differencing once achieves stationarity (p = {diff_pval:.4f}), indicating d=1."
        if recommended_D > 0:
            stationarity_verdict += f" Strong seasonal pattern suggests seasonal differencing D={recommended_D} as well."
    else:
        stationarity_verdict = f"Series exhibits strong persistence. May require second differencing (d=2) or seasonal integration (D=1)."

    diff_dates = [d.strftime('%Y-%m-%d') for d in diff_series.index]
    diff_values = [safe_float(v) for v in diff_series.values]
    
    q3_data = {
        'raw': {
            'statistic': raw_stat,
            'p_value': raw_pval,
            'is_stationary': raw_is_stationary,
            'critical_values': raw_crit
        },
        'diff': {
            'statistic': diff_stat,
            'p_value': diff_pval,
            'is_stationary': diff_is_stationary,
            'critical_values': diff_crit
        },
        'recommended_d': recommended_d,
        'recommended_D': recommended_D,
        'verdict': stationarity_verdict,
        'diff_chart': {
            'dates': diff_dates,
            'values': diff_values
        }
    }
    
    # =========================================================================
    # QUESTION 4: How does it remember its past? (ACF / PACF & Reading Assistant)
    # =========================================================================
    nlags = min(36, max(10, n // 3))
    acf_vals = acf(series.dropna(), nlags=nlags)
    pacf_vals = pacf(series.dropna(), nlags=nlags, method='ywm')
    
    # 95% Bartlett confidence threshold
    conf_bound = safe_float(1.96 / np.sqrt(n))
    
    reading_notes = []
    
    # 1. Seasonal check
    if effective_period < nlags:
        spike_season_1 = abs(acf_vals[effective_period]) > conf_bound
        spike_season_2 = (2 * effective_period < nlags) and (abs(acf_vals[2 * effective_period]) > conf_bound)
        if spike_season_1 or spike_season_2:
            reading_notes.append(
                f"Spikes at seasonal lag {effective_period} (and multiple {2*effective_period}) confirm strong annual periodic memory."
            )
            
    # 2. Persistence / Slow decay check
    first_few_acf = [abs(acf_vals[k]) > conf_bound for k in range(1, min(6, nlags))]
    if all(first_few_acf):
        reading_notes.append(
            "ACF shows very slow geometric decay across early lags, classic proof of unit-root trend persistence."
        )
        
    # 3. PACF cutoff check
    sig_pacf_lags = [k for k in range(1, min(6, nlags)) if abs(pacf_vals[k]) > conf_bound]
    if len(sig_pacf_lags) == 1 and sig_pacf_lags[0] == 1:
        reading_notes.append(
            "PACF cuts off cleanly after lag 1, while ACF decays gradually. This strongly suggests an Autoregressive AR(1) process."
        )
    elif len(sig_pacf_lags) > 1 and max(sig_pacf_lags) <= 3:
        reading_notes.append(
            f"PACF shows significant spikes up to lag {max(sig_pacf_lags)} before cutting off, indicating AR({max(sig_pacf_lags)})-type dynamics."
        )
        
    # 4. ACF cutoff check
    sig_acf_lags = [k for k in range(1, min(6, nlags)) if abs(acf_vals[k]) > conf_bound]
    if len(sig_acf_lags) == 1 and not raw_is_stationary:
        pass
    elif len(sig_acf_lags) == 1 and sig_acf_lags[0] == 1 and abs(pacf_vals[1]) > conf_bound and all(abs(pacf_vals[k]) <= conf_bound for k in range(2, 5)):
        pass
    elif len(sig_acf_lags) == 1 and sig_acf_lags[0] == 1:
        reading_notes.append(
            "ACF cuts off abruptly at lag 1 while PACF dampens, suggesting a Moving Average MA(1) component."
        )
        
    if not reading_notes:
        reading_notes.append("Correlations damp out rapidly within confidence bands; simple low-order specifications will suffice.")

    q4_data = {
        'lags': list(range(nlags + 1)),
        'acf': [safe_float(v) for v in acf_vals],
        'pacf': [safe_float(v) for v in pacf_vals],
        'conf_bound': conf_bound,
        'reading_notes': reading_notes
    }
    
    # =========================================================================
    # QUESTION 5: Which model fits best? (Holdout Evaluation & Leaderboard)
    # =========================================================================
    train_series = series.iloc[:-holdout_size]
    test_series = series.iloc[-holdout_size:]
    test_dates = [d.strftime('%Y-%m-%d') for d in test_series.index]
    test_actual = [safe_float(v) for v in test_series.values]
    
    d_use = recommended_d
    D_use = recommended_D
    s_use = effective_period
    
    # Candidate specifications to test
    candidates_configs = [
        # Family, name, order, seasonal_order
        ('Baseline', 'Seasonal Naive' if s_use > 1 else 'Naive Lag-1', (0,0,0), (0,0,0,0)),
        ('AR', f'AR(1)', (1, d_use, 0), (0, 0, 0, 0)),
        ('AR', f'AR(2)', (2, d_use, 0), (0, 0, 0, 0)),
        ('MA', f'MA(1)', (0, d_use, 1), (0, 0, 0, 0)),
        ('ARMA', f'ARMA(1,1)', (1, d_use, 1), (0, 0, 0, 0)),
        ('ARIMA', f'ARIMA(1,{d_use},1)', (1, d_use, 1), (0, 0, 0, 0)),
        ('ARIMA', f'ARIMA(2,{d_use},1)', (2, d_use, 1), (0, 0, 0, 0)),
    ]
    
    if s_use > 1 and n >= 2 * s_use:
        candidates_configs.extend([
            ('SARIMA', f'SARIMA(1,{d_use},0)(1,{D_use},0)[{s_use}]', (1, d_use, 0), (1, D_use, 0, s_use)),
            ('SARIMA', f'SARIMA(0,{d_use},1)(0,{D_use},1)[{s_use}]', (0, d_use, 1), (0, D_use, 1, s_use)),
            ('SARIMA', f'SARIMA(1,{d_use},1)(1,{D_use},1)[{s_use}]', (1, d_use, 1), (1, D_use, 1, s_use)),
            ('SARIMA', f'SARIMA(2,{d_use},1)(1,{D_use},1)[{s_use}]', (2, d_use, 1), (1, D_use, 1, s_use)),
        ])

    candidate_results = []
    best_fitted_model = None
    best_forecast_test = None
    
    for family, name, order, s_order in candidates_configs:
        try:
            if family == 'Baseline':
                if s_use > 1 and len(train_series) >= s_use:
                    # Seasonal naive: repeat last season
                    pred_vals = []
                    for step in range(holdout_size):
                        pred_vals.append(train_series.iloc[-s_use + (step % s_use)])
                    pred = np.array(pred_vals)
                else:
                    pred = np.repeat(train_series.iloc[-1], holdout_size)
                    
                aic_val = None
                bic_val = None
            else:
                model = SARIMAX(train_series, order=order, seasonal_order=s_order, enforce_stationarity=False, enforce_invertibility=False)
                fit_res = model.fit(disp=False, maxiter=150)
                pred = fit_res.forecast(steps=holdout_size).values
                aic_val = safe_float(fit_res.aic)
                bic_val = safe_float(fit_res.bic)
                
            # Error metrics
            errors = test_series.values - pred
            rmse = safe_float(np.sqrt(np.mean(errors ** 2)))
            mae = safe_float(np.mean(np.abs(errors)))
            with np.errstate(divide='ignore', invalid='ignore'):
                mape_arr = np.abs(errors / test_series.values) * 100.0
                mape = safe_float(np.nanmean(mape_arr))
                
            candidate_results.append({
                'model_name': name,
                'model_family': family,
                'order_str': str(order),
                'seasonal_order_str': str(s_order) if s_order != (0,0,0,0) else '',
                'aic': aic_val,
                'bic': bic_val,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'pred_test': [safe_float(v) for v in pred],
                'fit_success': True,
                'notes': f"{family} model with holdout RMSE {rmse:.2f}"
            })
        except Exception as e:
            candidate_results.append({
                'model_name': name,
                'model_family': family,
                'order_str': str(order),
                'seasonal_order_str': str(s_order),
                'aic': None,
                'bic': None,
                'rmse': 999999.0,
                'mae': 999999.0,
                'mape': 999.0,
                'pred_test': [],
                'fit_success': False,
                'notes': f"Convergence issue: {str(e)[:50]}"
            })

    # Sort candidates by RMSE
    valid_candidates = [c for c in candidate_results if c['fit_success']]
    if valid_candidates:
        valid_candidates.sort(key=lambda x: (x['rmse'] if x['rmse'] is not None else 999999))
        for rank_idx, cand in enumerate(valid_candidates):
            cand['rank'] = rank_idx + 1
            cand['is_best'] = (rank_idx == 0)
        best_cand = valid_candidates[0]
    else:
        best_cand = candidate_results[0]
        best_cand['rank'] = 1
        best_cand['is_best'] = True

    q5_data = {
        'best_model_name': best_cand['model_name'],
        'best_rmse': best_cand['rmse'],
        'best_aic': best_cand['aic'],
        'best_bic': best_cand['bic'],
        'best_mae': best_cand['mae'],
        'best_mape': best_cand['mape'],
        'test_dates': test_dates,
        'test_actual': test_actual,
        'test_predicted': best_cand.get('pred_test', []),
        'holdout_size': holdout_size
    }
    
    # =========================================================================
    # QUESTION 6: Can I trust the model? (Residuals & Ljung-Box Test)
    # =========================================================================
    # Refit the winning model architecture on train or full to diagnose residuals
    res_vals = []
    lb_stat = 0.0
    lb_pval = 1.0
    is_white_noise = True
    jb_pval = 1.0
    
    try:
        cand_order = eval(best_cand['order_str']) if best_cand['order_str'] else (1,1,1)
        cand_sorder = eval(best_cand['seasonal_order_str']) if best_cand['seasonal_order_str'] else (0,0,0,0)
        
        full_model = SARIMAX(series, order=cand_order, seasonal_order=cand_sorder, enforce_stationarity=False, enforce_invertibility=False)
        full_res = full_model.fit(disp=False, maxiter=150)
        
        residuals_raw = full_res.resid.dropna()
        # Drop first few difference lags from residuals
        drop_burn = max(2, d_use + D_use * s_use)
        if len(residuals_raw) > drop_burn:
            residuals_clean = residuals_raw.iloc[drop_burn:]
        else:
            residuals_clean = residuals_raw
            
        res_vals = [safe_float(v) for v in residuals_clean.values]
        
        # Ljung-Box test
        lb_lags = min(10, max(2, len(residuals_clean) // 5))
        lb_df = acorr_ljungbox(residuals_clean, lags=[lb_lags], return_df=True)
        lb_stat = safe_float(lb_df['lb_stat'].iloc[0])
        lb_pval = safe_float(lb_df['lb_pvalue'].iloc[0])
        is_white_noise = bool(lb_pval > 0.05)
        
        # Normality (Jarque-Bera)
        jb_res = stats.jarque_bera(residuals_clean)
        jb_pval = safe_float(jb_res.pvalue)
        
        # Residual ACF
        res_acf_vals = [safe_float(v) for v in acf(residuals_clean, nlags=min(20, len(residuals_clean)//3))]
    except Exception as e:
        residuals_clean = series - series.mean()
        res_vals = [safe_float(v) for v in residuals_clean.values]
        res_acf_vals = []

    if is_white_noise:
        residual_verdict = f"Residuals look like white noise (Ljung-Box p = {lb_pval:.3f} > 0.05), so the model successfully captured the underlying structure."
    else:
        residual_verdict = f"Residuals show some remaining autocorrelation (Ljung-Box p = {lb_pval:.3f} < 0.05). The forecast remains valuable, but confidence intervals should be interpreted with extra care."

    q6_data = {
        'ljungbox_stat': lb_stat,
        'ljungbox_pvalue': lb_pval,
        'is_white_noise': is_white_noise,
        'jarque_bera_pvalue': jb_pval,
        'verdict': residual_verdict,
        'residuals': res_vals,
        'res_dates': dates_str[-len(res_vals):],
        'res_acf': res_acf_vals
    }
    
    # =========================================================================
    # QUESTION 7: What happens next? (Out-of-sample Forecast with 95% Band)
    # =========================================================================
    # Generate future dates
    last_dt = series.index[-1]
    if freq == 'MS' or freq == 'M':
        future_dts = pd.date_range(start=last_dt + pd.DateOffset(months=1), periods=forecast_horizon, freq='MS')
    elif freq == 'D':
        future_dts = pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
    elif freq == 'W':
        future_dts = pd.date_range(start=last_dt + pd.Timedelta(weeks=1), periods=forecast_horizon, freq='W')
    elif freq == 'QS' or freq == 'Q':
        future_dts = pd.date_range(start=last_dt + pd.DateOffset(months=3), periods=forecast_horizon, freq='QS')
    else:
        future_dts = pd.date_range(start=last_dt + pd.DateOffset(months=1), periods=forecast_horizon, freq='MS')
        
    future_dates_str = [d.strftime('%Y-%m-%d') for d in future_dts]
    
    forecast_mean = []
    lower_95 = []
    upper_95 = []
    lower_80 = []
    upper_80 = []
    
    try:
        forecast_res = full_res.get_forecast(steps=forecast_horizon)
        forecast_mean = [safe_float(v) for v in forecast_res.predicted_mean]
        ci_95 = forecast_res.conf_int(alpha=0.05)
        lower_95 = [safe_float(v) for v in ci_95.iloc[:, 0]]
        upper_95 = [safe_float(v) for v in ci_95.iloc[:, 1]]
        ci_80 = forecast_res.conf_int(alpha=0.20)
        lower_80 = [safe_float(v) for v in ci_80.iloc[:, 0]]
        upper_80 = [safe_float(v) for v in ci_80.iloc[:, 1]]
    except Exception as e:
        # Fallback extrapolation
        last_val = series.values[-1]
        forecast_mean = [safe_float(last_val) for _ in range(forecast_horizon)]
        lower_95 = [safe_float(last_val - 1.96 * std_val) for _ in range(forecast_horizon)]
        upper_95 = [safe_float(last_val + 1.96 * std_val) for _ in range(forecast_horizon)]
        lower_80 = [safe_float(last_val - 1.28 * std_val) for _ in range(forecast_horizon)]
        upper_80 = [safe_float(last_val + 1.28 * std_val) for _ in range(forecast_horizon)]

    # Format table of future forecast steps
    table_rows = []
    for f_date, mean_v, l95, u95 in zip(future_dates_str, forecast_mean, lower_95, upper_95):
        table_rows.append({
            'date': f_date,
            'forecast': mean_v,
            'lower_95': l95,
            'upper_95': u95,
            'range': safe_float(u95 - l95)
        })

    q7_data = {
        'horizon': forecast_horizon,
        'future_dates': future_dates_str,
        'mean': forecast_mean,
        'lower_95': lower_95,
        'upper_95': upper_95,
        'lower_80': lower_80,
        'upper_80': upper_80,
        'table_rows': table_rows
    }
    
    # =========================================================================
    # PLAIN-LANGUAGE SUMMARY BOX
    # =========================================================================
    next_month_name = future_dts[0].strftime('%B %Y') if len(future_dts) > 0 else 'next period'
    next_val = forecast_mean[0] if forecast_mean else 0.0
    next_l95 = lower_95[0] if lower_95 else 0.0
    next_u95 = upper_95[0] if upper_95 else 0.0
    
    summary_sentences = [
        f"Your series shows a {trend_desc.lower()} and a {season_desc.lower()}.",
        f"It required {recommended_d} regular differencing order (d={recommended_d}) and {recommended_D} seasonal order (D={recommended_D}) to reach stationarity.",
        f"The best performing model is **{best_cand['model_name']}** (holdout RMSE {best_cand['rmse']:.2f} on the last {holdout_size} evaluation points).",
        f"Expected value for {next_month_name} is **{next_val:.2f}**, highly likely to land between **{next_l95:.2f}** and **{next_u95:.2f}** (95% confidence interval)."
    ]
    summary_text = " ".join(summary_sentences)
    summary_headline = f"{best_cand['model_name']} fit for {target_col} | Next period forecast: {next_val:.2f}"
    
    # =========================================================================
    # WARNINGS & GUARDRAILS (What the app will NOT tell)
    # =========================================================================
    warnings_list = [
        {
            'title': 'Patterns, Not Causes',
            'desc': 'Forecasts extrapolate statistical patterns from history; they do not prove why changes occur or what economic levers drive them.'
        },
        {
            'title': 'Sudden Shocks & Regime Shifts',
            'desc': 'Models trained on calm periods fail through black swan events or policy shifts (e.g. lockdowns, sudden price shocks). The past assumption may break.'
        },
        {
            'title': 'Certainty is an Illusion',
            'desc': 'Single line point forecasts hide true uncertainty. Always plan against the 95% uncertainty band, never just the centerline.'
        }
    ]
    
    if cv_val > 0.8:
        warnings_list.append({
            'title': 'High Volatility Warning',
            'desc': f'Series coefficient of variation is high ({cv_val:.2f}), leading to inherently wide confidence intervals.'
        })

    return {
        'q1_data_quality': q1_data,
        'q2_decomposition': q2_data,
        'q3_stationarity': q3_data,
        'q4_acf_pacf': q4_data,
        'q5_best_model': q5_data,
        'candidate_results': valid_candidates if valid_candidates else candidate_results,
        'q6_residuals': q6_data,
        'q7_forecast': q7_data,
        'summary_headline': summary_headline,
        'summary_text': summary_text,
        'warnings': warnings_list
    }
