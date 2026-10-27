from django import forms
from .models import Dataset, AnalysisRun, MultivariateRun

class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['name', 'description', 'file']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none',
                'placeholder': 'e.g. India Electricity Consumption 2012-2023'
            }),
            'description': forms.Textarea(attrs={
                'class': 'w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none',
                'rows': 3,
                'placeholder': 'Brief description of frequency, sources, and units...'
            }),
            'file': forms.FileInput(attrs={
                'class': 'w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer'
            })
        }

    def clean_file(self):
        f = self.cleaned_data.get('file')
        if f:
            ext = f.name.split('.')[-1].lower()
            if ext not in ['csv', 'xlsx', 'xls']:
                raise forms.ValidationError("Please upload a valid CSV or Excel file (.csv, .xlsx, .xls).")
        return f


class UnivariateConfigForm(forms.Form):
    run_name = forms.CharField(
        max_length=200,
        required=False,
        initial="Univariate Analysis",
        widget=forms.TextInput(attrs={
            'class': 'w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm'
        })
    )
    date_column = forms.ChoiceField(
        widget=forms.Select(attrs={
            'class': 'w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm bg-white'
        })
    )
    target_column = forms.ChoiceField(
        widget=forms.Select(attrs={
            'class': 'w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm bg-white'
        })
    )
    frequency = forms.ChoiceField(
        choices=[
            ('MS', 'Monthly (Month Start - e.g. 2020-01-01)'),
            ('M', 'Monthly (Month End - e.g. 2020-01-31)'),
            ('D', 'Daily (7 days/week)'),
            ('B', 'Business Daily (Mon-Fri)'),
            ('W', 'Weekly'),
            ('QS', 'Quarterly (Quarter Start)'),
            ('YS', 'Yearly / Annual')
        ],
        initial='MS',
        widget=forms.Select(attrs={
            'class': 'w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm bg-white'
        })
    )
    seasonal_period = forms.IntegerField(
        initial=12,
        min_value=1,
        help_text="Expected periodicity: 12 for monthly, 4 for quarterly, 7 for daily with weekly cycles.",
        widget=forms.NumberInput(attrs={
            'class': 'w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm'
        })
    )
    holdout_size = forms.IntegerField(
        initial=12,
        min_value=2,
        help_text="Number of most recent observations reserved for holdout validation leaderboard.",
        widget=forms.NumberInput(attrs={
            'class': 'w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm'
        })
    )
    forecast_horizon = forms.IntegerField(
        initial=12,
        min_value=1,
        max_value=60,
        help_text="Number of future steps to forecast beyond the end of the series.",
        widget=forms.NumberInput(attrs={
            'class': 'w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm'
        })
    )

    def __init__(self, *args, **kwargs):
        dataset = kwargs.pop('dataset', None)
        super().__init__(*args, **kwargs)
        if dataset and dataset.columns:
            cols = [c['name'] for c in dataset.columns] if isinstance(dataset.columns[0], dict) else dataset.columns
            choices = [(c, c) for c in cols]
            self.fields['date_column'].choices = choices
            self.fields['target_column'].choices = choices
            
            # Smart defaults
            date_matches = [c for c in cols if 'date' in c.lower() or 'time' in c.lower() or 'year' in c.lower()]
            if date_matches:
                self.fields['date_column'].initial = date_matches[0]
                
            numeric_cols = [c for c in cols if c not in date_matches]
            if numeric_cols:
                self.fields['target_column'].initial = numeric_cols[0]


class MultivariateConfigForm(forms.Form):
    run_name = forms.CharField(
        max_length=200,
        required=False,
        initial="Multivariate VAR Analysis",
        widget=forms.TextInput(attrs={
            'class': 'w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm'
        })
    )
    date_column = forms.ChoiceField(
        widget=forms.Select(attrs={
            'class': 'w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm bg-white'
        })
    )
    target_columns = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple(attrs={
            'class': 'h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-slate-300 rounded mr-2'
        }),
        help_text="Select at least 2 time series to estimate joint dynamic feedback."
    )
    frequency = forms.ChoiceField(
        choices=[
            ('MS', 'Monthly (Month Start)'),
            ('M', 'Monthly (Month End)'),
            ('D', 'Daily'),
            ('W', 'Weekly'),
            ('QS', 'Quarterly')
        ],
        initial='MS',
        widget=forms.Select(attrs={
            'class': 'w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm bg-white'
        })
    )
    model_type = forms.ChoiceField(
        choices=[
            ('VAR', 'Vector Autoregression (VAR)'),
            ('VARMA', 'Vector Autoregressive Moving Average (VARMA - Advanced)')
        ],
        initial='VAR',
        widget=forms.Select(attrs={
            'class': 'w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm bg-white'
        })
    )
    max_lags = forms.IntegerField(
        initial=4,
        min_value=1,
        max_value=12,
        help_text="Maximum lag order p for VAR/VARMA selection.",
        widget=forms.NumberInput(attrs={
            'class': 'w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm'
        })
    )
    forecast_horizon = forms.IntegerField(
        initial=12,
        min_value=1,
        max_value=36,
        widget=forms.NumberInput(attrs={
            'class': 'w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:outline-none text-sm'
        })
    )

    def __init__(self, *args, **kwargs):
        dataset = kwargs.pop('dataset', None)
        super().__init__(*args, **kwargs)
        if dataset and dataset.columns:
            cols = [c['name'] for c in dataset.columns] if isinstance(dataset.columns[0], dict) else dataset.columns
            choices = [(c, c) for c in cols]
            self.fields['date_column'].choices = choices
            self.fields['target_columns'].choices = choices
            
            date_matches = [c for c in cols if 'date' in c.lower() or 'time' in c.lower()]
            if date_matches:
                self.fields['date_column'].initial = date_matches[0]
                
            non_date = [c for c in cols if c not in date_matches]
            if len(non_date) >= 2:
                self.fields['target_columns'].initial = non_date[:4]

    def clean_target_columns(self):
        targets = self.cleaned_data.get('target_columns')
        if not targets or len(targets) < 2:
            raise forms.ValidationError("Please select at least 2 series for multivariate VAR modeling.")
        return targets
