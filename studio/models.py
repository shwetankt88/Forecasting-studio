from django.db import models
import json

class Dataset(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True, default='')
    file = models.FileField(upload_to='datasets/', blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_type = models.CharField(max_length=20, default='csv')
    rows_count = models.IntegerField(default=0)
    columns_json = models.JSONField(default=list, help_text="List of column names and preview types")
    is_sample = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.name} ({self.rows_count} rows)"

    @property
    def columns(self):
        return self.columns_json if isinstance(self.columns_json, list) else []


class AnalysisRun(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='runs')
    run_name = models.CharField(max_length=200, default='Univariate Analysis')
    date_column = models.CharField(max_length=100)
    target_column = models.CharField(max_length=100)
    frequency = models.CharField(max_length=20, default='MS')
    seasonal_period = models.IntegerField(default=12)
    holdout_size = models.IntegerField(default=12)
    forecast_horizon = models.IntegerField(default=12)
    created_at = models.DateTimeField(auto_now_add=True)

    # Plain Language Summary
    summary_headline = models.CharField(max_length=300, blank=True, default='')
    summary_text = models.TextField(blank=True, default='')

    # Q1: Data Inspection & Quality
    q1_data_quality = models.JSONField(default=dict)

    # Q2: Decomposition (Trend, Seasonal, Residual)
    q2_decomposition = models.JSONField(default=dict)

    # Q3: Stationarity & Differencing (ADF test, d & D advice)
    q3_stationarity = models.JSONField(default=dict)

    # Q4: ACF & PACF + Reading Assistant
    q4_acf_pacf = models.JSONField(default=dict)

    # Q5: Best Model fit
    q5_best_model = models.JSONField(default=dict)

    # Q6: Residual Diagnostics & Ljung-Box test
    q6_residuals = models.JSONField(default=dict)

    # Q7: Out-of-sample Forecast & 95% interval
    q7_forecast = models.JSONField(default=dict)

    # Caveats & Disclaimers
    warnings = models.JSONField(default=list)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.run_name} - {self.target_column} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"

    @property
    def best_candidate(self):
        return self.candidates.filter(is_best=True).first() or self.candidates.first()


class ModelCandidate(models.Model):
    run = models.ForeignKey(AnalysisRun, on_delete=models.CASCADE, related_name='candidates')
    model_name = models.CharField(max_length=100)
    model_family = models.CharField(max_length=50) # AR, MA, ARMA, ARIMA, SARIMA, Baseline
    order_str = models.CharField(max_length=50, blank=True, default='')
    seasonal_order_str = models.CharField(max_length=50, blank=True, default='')
    aic = models.FloatField(null=True, blank=True)
    bic = models.FloatField(null=True, blank=True)
    rmse = models.FloatField(null=True, blank=True)
    mae = models.FloatField(null=True, blank=True)
    mape = models.FloatField(null=True, blank=True)
    rank = models.IntegerField(default=99)
    is_best = models.BooleanField(default=False)
    fit_success = models.BooleanField(default=True)
    notes = models.TextField(blank=True, default='')

    class Meta:
        ordering = ['rank', 'rmse', 'aic']

    def __str__(self):
        return f"{self.model_name} (RMSE: {self.rmse:.3f if self.rmse else 'N/A'})"


class MultivariateRun(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='multivariate_runs')
    run_name = models.CharField(max_length=200, default='Multivariate VAR/VARMA Analysis')
    date_column = models.CharField(max_length=100)
    target_columns = models.JSONField(default=list)
    frequency = models.CharField(max_length=20, default='MS')
    model_type = models.CharField(max_length=20, default='VAR') # VAR or VARMA
    lag_order = models.IntegerField(default=2)
    ma_order = models.IntegerField(default=0)
    forecast_horizon = models.IntegerField(default=12)
    created_at = models.DateTimeField(auto_now_add=True)

    summary_text = models.TextField(blank=True, default='')
    correlation_matrix = models.JSONField(default=dict)
    granger_causality = models.JSONField(default=dict)
    irf_data = models.JSONField(default=dict)
    fevd_data = models.JSONField(default=dict)
    joint_forecast = models.JSONField(default=dict)
    series_history = models.JSONField(default=dict)
    metrics = models.JSONField(default=dict)
    warnings = models.JSONField(default=list)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.run_name} ({', '.join(self.target_columns)}) - {self.model_type}"
