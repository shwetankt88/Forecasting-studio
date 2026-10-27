from django.urls import path
from . import views

urlpatterns = [
    path('', views.DashboardView.as_view(), name='dashboard'),
    path('sample/load/', views.LoadSampleDatasetsView.as_view(), name='load_sample_datasets'),
    
    # Datasets
    path('datasets/', views.DatasetListView.as_view(), name='dataset_list'),
    path('datasets/upload/', views.DatasetUploadView.as_view(), name='dataset_upload'),
    path('datasets/<int:pk>/', views.DatasetDetailView.as_view(), name='dataset_detail'),
    path('datasets/<int:pk>/delete/', views.DatasetDeleteView.as_view(), name='dataset_delete'),
    
    # Mode A: Univariate
    path('datasets/<int:dataset_id>/univariate/new/', views.UnivariateRunCreateView.as_view(), name='univariate_create'),
    path('univariate/<int:pk>/', views.UnivariateRunDetailView.as_view(), name='univariate_detail'),
    path('univariate/<int:pk>/export/', views.UnivariateExportCSVView.as_view(), name='univariate_export'),
    path('univariate/<int:pk>/delete/', views.RunDeleteView.as_view(), name='univariate_delete'),
    
    # Mode B: Multivariate (VAR / VARMA)
    path('datasets/<int:dataset_id>/multivariate/new/', views.MultivariateRunCreateView.as_view(), name='multivariate_create'),
    path('multivariate/<int:pk>/', views.MultivariateRunDetailView.as_view(), name='multivariate_detail'),
    path('multivariate/<int:pk>/export/', views.MultivariateExportCSVView.as_view(), name='multivariate_export'),
    
    # Run Comparison
    path('compare/', views.RunCompareView.as_view(), name='run_compare'),
]
