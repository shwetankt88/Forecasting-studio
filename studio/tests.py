from django.test import TestCase, Client
from django.urls import reverse
from studio.models import Dataset, AnalysisRun, MultivariateRun

class ForecastingStudioTests(TestCase):
    def setUp(self):
        self.client = Client()
        # Ensure benchmark sample exists
        from studio.engine.sample_data import generate_sample_datasets
        generate_sample_datasets()

    def test_dashboard_view(self):
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Forecasting Studio")
        self.assertContains(response, "Univariate Report Studio")
        self.assertContains(response, "Multivariate VAR")

    def test_datasets_list_view(self):
        response = self.client.get(reverse('dataset_list'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Time Series Datasets")

    def test_dataset_detail_view(self):
        ds = Dataset.objects.first()
        self.assertIsNotNone(ds)
        response = self.client.get(reverse('dataset_detail', kwargs={'pk': ds.pk}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, ds.name)

    def test_univariate_run_and_detail(self):
        ds = Dataset.objects.filter(name__icontains="Electricity").first()
        self.assertIsNotNone(ds)
        
        # Test creating a univariate run via POST
        post_data = {
            'run_name': 'Test Elec Run',
            'date_column': 'Date',
            'target_column': 'Electricity_Consumption_BU',
            'frequency': 'MS',
            'seasonal_period': 12,
            'holdout_size': 12,
            'forecast_horizon': 12
        }
        response = self.client.post(reverse('univariate_create', kwargs={'dataset_id': ds.pk}), post_data, follow=True)
        self.assertEqual(response.status_code, 200)
        
        # Check that run was created
        run = AnalysisRun.objects.filter(run_name='Test Elec Run').first()
        self.assertIsNotNone(run)
        
        # Check detail page answers the 7 questions
        detail_res = self.client.get(reverse('univariate_detail', kwargs={'pk': run.pk}))
        self.assertEqual(detail_res.status_code, 200)
        self.assertContains(detail_res, "What does my data look like?")
        self.assertContains(detail_res, "What is it made of?")
        self.assertContains(detail_res, "Is it stable or drifting?")
        self.assertContains(detail_res, "How does it remember its past?")
        self.assertContains(detail_res, "Which model fits best?")
        self.assertContains(detail_res, "Can I trust the model?")
        self.assertContains(detail_res, "What happens next?")
        self.assertContains(detail_res, "Analyst Executive Summary")
        self.assertContains(detail_res, "What This App Will NOT Tell You")

        # Test CSV export
        export_res = self.client.get(reverse('univariate_export', kwargs={'pk': run.pk}))
        self.assertEqual(export_res.status_code, 200)
        self.assertEqual(export_res['Content-Type'], 'text/csv')
        self.assertIn(b'Forecast_Date', export_res.content)

    def test_multivariate_run_and_detail(self):
        ds = Dataset.objects.filter(name__icontains="Energy, Weather").first()
        self.assertIsNotNone(ds)
        
        post_data = {
            'run_name': 'Test VAR Macro Run',
            'date_column': 'Date',
            'target_columns': ['Electricity_Demand_BU', 'Avg_Temperature_C'],
            'frequency': 'MS',
            'model_type': 'VAR',
            'max_lags': 3,
            'forecast_horizon': 12
        }
        response = self.client.post(reverse('multivariate_create', kwargs={'dataset_id': ds.pk}), post_data, follow=True)
        self.assertEqual(response.status_code, 200)
        
        multi_run = MultivariateRun.objects.filter(run_name='Test VAR Macro Run').first()
        self.assertIsNotNone(multi_run)
        
        detail_res = self.client.get(reverse('multivariate_detail', kwargs={'pk': multi_run.pk}))
        self.assertEqual(detail_res.status_code, 200)
        self.assertContains(detail_res, "Do these series move together?")
        self.assertContains(detail_res, "Does A help predict B?")
        self.assertContains(detail_res, "How long does a shock last?")
        self.assertContains(detail_res, "What drives the uncertainty?")
        self.assertContains(detail_res, "What are the joint forecasts?")

        # Test CSV export
        export_res = self.client.get(reverse('multivariate_export', kwargs={'pk': multi_run.pk}))
        self.assertEqual(export_res.status_code, 200)
        self.assertEqual(export_res['Content-Type'], 'text/csv')

    def test_run_compare_view(self):
        response = self.client.get(reverse('run_compare'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Run Comparison Studio")
