import pandas as pd
from django.core.management.base import BaseCommand
from studio.models import Dataset, AnalysisRun, ModelCandidate, MultivariateRun
from studio.engine.sample_data import generate_sample_datasets
from studio.engine.univariate import analyze_univariate_series
from studio.engine.multivariate import analyze_multivariate_series

class Command(BaseCommand):
    help = "Initializes sample benchmark datasets and runs Mode A (Univariate) & Mode B (VAR) analytical pipelines."

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE("1. Generating canonical benchmark datasets..."))
        created = generate_sample_datasets()
        self.stdout.write(self.style.SUCCESS(f"Datasets ready ({len(created)} created/verified)."))

        # 2. Run Mode A Benchmark: Monthly India Electricity Consumption
        ds_elec = Dataset.objects.filter(name__icontains="Electricity").first()
        if ds_elec and not ds_elec.runs.exists():
            self.stdout.write(self.style.NOTICE("2. Executing Mode A Univariate analysis on India Electricity..."))
            df_elec = pd.read_csv(ds_elec.file.path)
            res_a = analyze_univariate_series(
                df=df_elec,
                date_col='Date',
                target_col='Electricity_Consumption_BU',
                freq='MS',
                seasonal_period=12,
                holdout_size=12,
                forecast_horizon=12
            )

            run_a = AnalysisRun.objects.create(
                dataset=ds_elec,
                run_name="India Monthly Electricity Demand (2012-2023)",
                date_column='Date',
                target_column='Electricity_Consumption_BU',
                frequency='MS',
                seasonal_period=12,
                holdout_size=12,
                forecast_horizon=12,
                summary_headline=res_a['summary_headline'],
                summary_text=res_a['summary_text'],
                q1_data_quality=res_a['q1_data_quality'],
                q2_decomposition=res_a['q2_decomposition'],
                q3_stationarity=res_a['q3_stationarity'],
                q4_acf_pacf=res_a['q4_acf_pacf'],
                q5_best_model=res_a['q5_best_model'],
                q6_residuals=res_a['q6_residuals'],
                q7_forecast=res_a['q7_forecast'],
                warnings=res_a['warnings']
            )

            for cand in res_a['candidate_results']:
                ModelCandidate.objects.create(
                    run=run_a,
                    model_name=cand['model_name'],
                    model_family=cand['model_family'],
                    order_str=cand['order_str'],
                    seasonal_order_str=cand['seasonal_order_str'],
                    aic=cand['aic'],
                    bic=cand['bic'],
                    rmse=cand['rmse'],
                    mae=cand['mae'],
                    mape=cand['mape'],
                    rank=cand.get('rank', 99),
                    is_best=cand.get('is_best', False),
                    fit_success=cand['fit_success'],
                    notes=cand.get('notes', '')
                )
            self.stdout.write(self.style.SUCCESS(f"Mode A completed successfully! Best model: {run_a.best_candidate.model_name} (Holdout RMSE: {run_a.best_candidate.rmse:.2f})"))
        else:
            self.stdout.write(self.style.SUCCESS("Mode A run already exists."))

        # 3. Run Mode B Benchmark: Multivariate Energy & Macro
        ds_macro = Dataset.objects.filter(name__icontains="Energy, Weather").first()
        if ds_macro and not ds_macro.multivariate_runs.exists():
            self.stdout.write(self.style.NOTICE("3. Executing Mode B Multivariate VAR analysis..."))
            df_macro = pd.read_csv(ds_macro.file.path)
            target_cols = ['Electricity_Demand_BU', 'Avg_Temperature_C', 'Industrial_Index_IIP', 'Renewable_Generation_BU']
            
            res_b = analyze_multivariate_series(
                df=df_macro,
                date_col='Date',
                target_cols=target_cols,
                freq='MS',
                model_type='VAR',
                max_lags=4,
                forecast_horizon=12
            )

            run_b = MultivariateRun.objects.create(
                dataset=ds_macro,
                run_name="India Power System & Weather Interaction (VAR)",
                date_column='Date',
                target_columns=target_cols,
                frequency='MS',
                model_type='VAR',
                lag_order=res_b['metrics']['lag_order'],
                forecast_horizon=12,
                summary_text=res_b['summary_text'],
                correlation_matrix=res_b['correlation_matrix'],
                granger_causality=res_b['granger_causality'],
                irf_data=res_b['irf_data'],
                fevd_data=res_b['fevd_data'],
                joint_forecast=res_b['joint_forecast'],
                series_history=res_b['series_history'],
                metrics=res_b['metrics'],
                warnings=res_b['warnings']
            )
            self.stdout.write(self.style.SUCCESS(f"Mode B completed successfully! Estimated VAR(lag={run_b.lag_order})."))
        else:
            self.stdout.write(self.style.SUCCESS("Mode B run already exists."))

        self.stdout.write(self.style.SUCCESS("\nAll benchmarks initialized! Ready to run web server."))
