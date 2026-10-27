import os
import csv
import json
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.views import View
from django.views.generic import ListView, DetailView, CreateView, DeleteView
from django.contrib import messages
from django.urls import reverse, reverse_lazy

from .models import Dataset, AnalysisRun, ModelCandidate, MultivariateRun
from .forms import DatasetUploadForm, UnivariateConfigForm, MultivariateConfigForm
from .engine.sample_data import generate_sample_datasets
from .engine.univariate import analyze_univariate_series
from .engine.multivariate import analyze_multivariate_series

class DashboardView(View):
    def get(self, request):
        datasets = Dataset.objects.all().order_by('-uploaded_at')
        univariate_runs = AnalysisRun.objects.select_related('dataset').all()[:6]
        multivariate_runs = MultivariateRun.objects.select_related('dataset').all()[:6]
        
        # Check if sample datasets exist
        sample_count = Dataset.objects.filter(is_sample=True).count()
        
        context = {
            'datasets': datasets,
            'univariate_runs': univariate_runs,
            'multivariate_runs': multivariate_runs,
            'sample_count': sample_count,
            'total_datasets': datasets.count(),
            'total_runs': AnalysisRun.objects.count() + MultivariateRun.objects.count()
        }
        return render(request, 'studio/dashboard.html', context)


class LoadSampleDatasetsView(View):
    def post(self, request):
        created = generate_sample_datasets()
        messages.success(request, f"Successfully loaded canonical sample datasets!")
        return redirect('dashboard')


class DatasetListView(ListView):
    model = Dataset
    template_name = 'studio/dataset_list.html'
    context_object_name = 'datasets'
    ordering = ['-uploaded_at']


class DatasetUploadView(View):
    def get(self, request):
        form = DatasetUploadForm()
        return render(request, 'studio/dataset_upload.html', {'form': form})
        
    def post(self, request):
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save(commit=False)
            uploaded_file = request.FILES['file']
            ext = uploaded_file.name.split('.')[-1].lower()
            dataset.file_type = ext
            dataset.save()
            
            # Inspect with pandas
            try:
                file_path = dataset.file.path
                if ext == 'csv':
                    df = pd.read_csv(file_path, nrows=500)
                else:
                    df = pd.read_excel(file_path, nrows=500)
                    
                cols_meta = []
                for c in df.columns:
                    col_type = 'string'
                    if pd.api.types.is_numeric_dtype(df[c]):
                        col_type = 'numeric'
                    elif pd.api.types.is_datetime64_any_dtype(df[c]):
                        col_type = 'date'
                    elif 'date' in str(c).lower() or 'time' in str(c).lower() or 'year' in str(c).lower():
                        col_type = 'date'
                    
                    sample_val = str(df[c].dropna().iloc[0]) if not df[c].dropna().empty else ''
                    cols_meta.append({
                        'name': str(c),
                        'type': col_type,
                        'sample': sample_val[:30]
                    })
                    
                dataset.columns_json = cols_meta
                dataset.rows_count = len(df)
                dataset.save()
                messages.success(request, f"Dataset '{dataset.name}' uploaded successfully with {len(cols_meta)} detected columns!")
                return redirect('dataset_detail', pk=dataset.pk)
            except Exception as e:
                messages.error(request, f"Error processing file columns: {str(e)}")
                return redirect('dashboard')
        return render(request, 'studio/dataset_upload.html', {'form': form})


class DatasetDetailView(View):
    def get(self, request, pk):
        dataset = get_object_or_404(Dataset, pk=pk)
        preview_rows = []
        preview_cols = []
        try:
            file_path = dataset.file.path
            if dataset.file_type == 'csv':
                df = pd.read_csv(file_path, nrows=10)
            else:
                df = pd.read_excel(file_path, nrows=10)
            preview_cols = list(df.columns)
            preview_rows = df.fillna('').to_dict(orient='records')
        except Exception as e:
            preview_rows = []
            
        univariate_runs = dataset.runs.all()[:5]
        multivariate_runs = dataset.multivariate_runs.all()[:5]
        
        context = {
            'dataset': dataset,
            'preview_cols': preview_cols,
            'preview_rows': preview_rows,
            'univariate_runs': univariate_runs,
            'multivariate_runs': multivariate_runs,
        }
        return render(request, 'studio/dataset_detail.html', context)


class DatasetDeleteView(DeleteView):
    model = Dataset
    success_url = reverse_lazy('dashboard')
    
    def post(self, request, *args, **kwargs):
        messages.success(request, "Dataset and associated analysis runs removed.")
        return super().post(request, *args, **kwargs)


class UnivariateRunCreateView(View):
    def get(self, request, dataset_id):
        dataset = get_object_or_404(Dataset, pk=dataset_id)
        form = UnivariateConfigForm(dataset=dataset)
        return render(request, 'studio/univariate_create.html', {'dataset': dataset, 'form': form})
        
    def post(self, request, dataset_id):
        dataset = get_object_or_404(Dataset, pk=dataset_id)
        form = UnivariateConfigForm(request.POST, dataset=dataset)
        
        if form.is_valid():
            data_cleaned = form.cleaned_data
            date_col = data_cleaned['date_column']
            target_col = data_cleaned['target_column']
            freq = data_cleaned['frequency']
            seasonal_period = data_cleaned['seasonal_period']
            holdout_size = data_cleaned['holdout_size']
            forecast_horizon = data_cleaned['forecast_horizon']
            run_name = data_cleaned['run_name'] or f"{target_col} Analysis"
            
            try:
                # Load dataframe
                file_path = dataset.file.path
                if dataset.file_type == 'csv':
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                    
                # Run full analytical pipeline (Q1 to Q7 + plain language summary)
                results = analyze_univariate_series(
                    df=df,
                    date_col=date_col,
                    target_col=target_col,
                    freq=freq,
                    seasonal_period=seasonal_period,
                    holdout_size=holdout_size,
                    forecast_horizon=forecast_horizon
                )
                
                # Persist AnalysisRun model
                run = AnalysisRun.objects.create(
                    dataset=dataset,
                    run_name=run_name,
                    date_column=date_col,
                    target_column=target_col,
                    frequency=freq,
                    seasonal_period=seasonal_period,
                    holdout_size=holdout_size,
                    forecast_horizon=forecast_horizon,
                    summary_headline=results['summary_headline'],
                    summary_text=results['summary_text'],
                    q1_data_quality=results['q1_data_quality'],
                    q2_decomposition=results['q2_decomposition'],
                    q3_stationarity=results['q3_stationarity'],
                    q4_acf_pacf=results['q4_acf_pacf'],
                    q5_best_model=results['q5_best_model'],
                    q6_residuals=results['q6_residuals'],
                    q7_forecast=results['q7_forecast'],
                    warnings=results['warnings']
                )
                
                # Persist Model Candidates (Leaderboard)
                for cand in results['candidate_results']:
                    ModelCandidate.objects.create(
                        run=run,
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
                    
                messages.success(request, f"Univariate analysis complete! Best model identified: {run.best_candidate.model_name}")
                return redirect('univariate_detail', pk=run.pk)
            except Exception as e:
                messages.error(request, f"Failed to run analysis: {str(e)}")
                return render(request, 'studio/univariate_create.html', {'dataset': dataset, 'form': form})
                
        return render(request, 'studio/univariate_create.html', {'dataset': dataset, 'form': form})


class UnivariateRunDetailView(View):
    def get(self, request, pk):
        run = get_object_or_404(AnalysisRun.objects.select_related('dataset').prefetch_related('candidates'), pk=pk)
        candidates = run.candidates.all()
        
        # Prepare JSON safe payloads for charts
        chart_payload = {
            'q1': run.q1_data_quality.get('chart_data', {}),
            'q2': run.q2_decomposition.get('chart_data', {}),
            'q3': run.q3_stationarity.get('diff_chart', {}),
            'q4': run.q4_acf_pacf,
            'q5_actual': run.q5_best_model.get('test_actual', []),
            'q5_predicted': run.q5_best_model.get('test_predicted', []),
            'q5_dates': run.q5_best_model.get('test_dates', []),
            'q6_res': run.q6_residuals.get('residuals', []),
            'q6_dates': run.q6_residuals.get('res_dates', []),
            'q6_acf': run.q6_residuals.get('res_acf', []),
            'q7_dates': run.q7_forecast.get('future_dates', []),
            'q7_mean': run.q7_forecast.get('mean', []),
            'q7_lower95': run.q7_forecast.get('lower_95', []),
            'q7_upper95': run.q7_forecast.get('upper_95', []),
        }
        
        # Related runs on the same dataset for quick comparison
        peer_runs = AnalysisRun.objects.filter(dataset=run.dataset).exclude(pk=run.pk)
        
        context = {
            'run': run,
            'candidates': candidates,
            'chart_json': json.dumps(chart_payload),
            'peer_runs': peer_runs
        }
        return render(request, 'studio/univariate_detail.html', context)


class UnivariateExportCSVView(View):
    def get(self, request, pk):
        run = get_object_or_404(AnalysisRun, pk=pk)
        q7 = run.q7_forecast
        table_rows = q7.get('table_rows', [])
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{run.target_column}_forecast_{run.id}.csv"'
        
        writer = csv.writer(response)
        writer.writerow(['Forecast_Date', 'Point_Forecast', 'Lower_95_CI', 'Upper_95_CI', 'Uncertainty_Range_95'])
        
        for row in table_rows:
            writer.writerow([
                row.get('date'),
                row.get('forecast'),
                row.get('lower_95'),
                row.get('upper_95'),
                row.get('range')
            ])
            
        return response


class MultivariateRunCreateView(View):
    def get(self, request, dataset_id):
        dataset = get_object_or_404(Dataset, pk=dataset_id)
        form = MultivariateConfigForm(dataset=dataset)
        return render(request, 'studio/multivariate_create.html', {'dataset': dataset, 'form': form})
        
    def post(self, request, dataset_id):
        dataset = get_object_or_404(Dataset, pk=dataset_id)
        form = MultivariateConfigForm(request.POST, dataset=dataset)
        
        if form.is_valid():
            data_cleaned = form.cleaned_data
            date_col = data_cleaned['date_column']
            target_cols = data_cleaned['target_columns']
            freq = data_cleaned['frequency']
            model_type = data_cleaned['model_type']
            max_lags = data_cleaned['max_lags']
            forecast_horizon = data_cleaned['forecast_horizon']
            run_name = data_cleaned['run_name'] or f"VAR System ({len(target_cols)} series)"
            
            try:
                file_path = dataset.file.path
                if dataset.file_type == 'csv':
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                    
                results = analyze_multivariate_series(
                    df=df,
                    date_col=date_col,
                    target_cols=target_cols,
                    freq=freq,
                    model_type=model_type,
                    max_lags=max_lags,
                    forecast_horizon=forecast_horizon
                )
                
                run = MultivariateRun.objects.create(
                    dataset=dataset,
                    run_name=run_name,
                    date_column=date_col,
                    target_columns=target_cols,
                    frequency=freq,
                    model_type=model_type,
                    lag_order=results['metrics']['lag_order'],
                    forecast_horizon=forecast_horizon,
                    summary_text=results['summary_text'],
                    correlation_matrix=results['correlation_matrix'],
                    granger_causality=results['granger_causality'],
                    irf_data=results['irf_data'],
                    fevd_data=results['fevd_data'],
                    joint_forecast=results['joint_forecast'],
                    series_history=results['series_history'],
                    metrics=results['metrics'],
                    warnings=results['warnings']
                )
                
                messages.success(request, f"Multivariate analysis successfully estimated VAR(lag={run.lag_order})!")
                return redirect('multivariate_detail', pk=run.pk)
            except Exception as e:
                messages.error(request, f"Failed multivariate estimation: {str(e)}")
                return render(request, 'studio/multivariate_create.html', {'dataset': dataset, 'form': form})
                
        return render(request, 'studio/multivariate_create.html', {'dataset': dataset, 'form': form})


class MultivariateRunDetailView(View):
    def get(self, request, pk):
        run = get_object_or_404(MultivariateRun.objects.select_related('dataset'), pk=pk)
        
        multi_payload = {
            'variables': run.target_columns,
            'history': run.series_history,
            'corr': run.correlation_matrix,
            'granger': run.granger_causality,
            'irf': run.irf_data,
            'fevd': run.fevd_data,
            'forecast': run.joint_forecast
        }
        
        context = {
            'run': run,
            'multi_json': json.dumps(multi_payload)
        }
        return render(request, 'studio/multivariate_detail.html', context)


class MultivariateExportCSVView(View):
    def get(self, request, pk):
        run = get_object_or_404(MultivariateRun, pk=pk)
        joint_fc = run.joint_forecast
        rows = joint_fc.get('table_rows', [])
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="joint_var_forecast_{run.id}.csv"'
        
        if rows:
            headers = list(rows[0].keys())
            writer = csv.DictWriter(response, fieldnames=headers)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        return response


class RunCompareView(View):
    def get(self, request):
        dataset_id = request.GET.get('dataset')
        run_ids = request.GET.getlist('runs')
        
        datasets = Dataset.objects.filter(runs__isnull=False).distinct()
        selected_dataset = None
        runs_to_compare = []
        
        if dataset_id:
            selected_dataset = get_object_or_404(Dataset, pk=dataset_id)
            available_runs = selected_dataset.runs.all()
            
            if run_ids:
                runs_to_compare = AnalysisRun.objects.filter(id__in=run_ids, dataset=selected_dataset).prefetch_related('candidates')
            else:
                runs_to_compare = available_runs[:3]
        else:
            if datasets.exists():
                selected_dataset = datasets.first()
                runs_to_compare = selected_dataset.runs.all()[:3]
                
        compare_chart_data = {
            'runs': []
        }
        for r in runs_to_compare:
            q7 = r.q7_forecast
            compare_chart_data['runs'].append({
                'run_name': r.run_name,
                'target': r.target_column,
                'best_model': r.best_candidate.model_name if r.best_candidate else 'N/A',
                'rmse': r.best_candidate.rmse if r.best_candidate else None,
                'dates': q7.get('future_dates', []),
                'mean': q7.get('mean', [])
            })
            
        context = {
            'datasets': datasets,
            'selected_dataset': selected_dataset,
            'runs_to_compare': runs_to_compare,
            'compare_json': json.dumps(compare_chart_data)
        }
        return render(request, 'studio/run_compare.html', context)


class RunDeleteView(DeleteView):
    model = AnalysisRun
    
    def get_success_url(self):
        return reverse('dataset_detail', kwargs={'pk': self.object.dataset.pk})
        
    def post(self, request, *args, **kwargs):
        messages.success(request, "Run record deleted successfully.")
        return super().post(request, *args, **kwargs)
