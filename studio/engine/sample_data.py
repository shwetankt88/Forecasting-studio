import os
import pandas as pd
import numpy as np
from datetime import datetime
from django.conf import settings
from studio.models import Dataset

def generate_sample_datasets():
    """
    Creates the two canonical datasets if they do not already exist:
    1. Monthly India Electricity Consumption (2012-2023, 144 points)
    2. India Energy & Macro Multivariate Series (2012-2023)
    """
    datasets_dir = os.path.join(settings.MEDIA_ROOT, 'datasets')
    os.makedirs(datasets_dir, exist_ok=True)
    
    created_datasets = []

    # 1. Monthly India Electricity Consumption (Univariate Mode A Canonical Example)
    univ_qs = Dataset.objects.filter(name="Monthly India Electricity Consumption (2012-2023)", is_sample=True)
    if not univ_qs.exists():
        # 144 months: Jan 2012 to Dec 2023
        dates = pd.date_range(start='2012-01-01', end='2023-12-01', freq='MS')
        n = len(dates)
        
        # Base demand ~ 70 Billion Units (BU / TWh) in 2012, growing to ~ 130 BU by 2023
        # Upward trend
        trend = np.linspace(70.0, 130.0, n)
        
        # Seasonal cycle in India:
        # Months: Jan=0, Feb=1, Mar=2, Apr=3, May=4, Jun=5, Jul=6, Aug=7, Sep=8, Oct=9, Nov=10, Dec=11
        # Summer peak in April-June (+10 to +16 BU) due to extreme cooling / heatwaves
        # Monsoon dip in July-August (-2 to -4 BU) due to cooling rains and agricultural hydropower
        # Festive season / post-monsoon pick-up in Sept-Oct (+5 to +8 BU)
        # Winter baseline in Dec-Jan
        monthly_seasonality = np.array([
            -4.5,  # Jan
            -3.0,  # Feb
             3.2,  # Mar
            11.5,  # Apr (Summer peak begins)
            16.8,  # May (Peak summer heat)
            13.4,  # Jun (Late summer / pre-monsoon)
             3.0,  # Jul (Monsoon sets in)
            -1.5,  # Aug (Monsoon lull)
             4.0,  # Sep (Post-monsoon)
             7.2,  # Oct (Festive Diwali surge)
             0.5,  # Nov
            -3.8   # Dec (Winter)
        ])
        
        seasonal_component = np.tile(monthly_seasonality, int(np.ceil(n / 12)))[:n]
        
        # Random noise (seed fixed for reproducible elegance)
        np.random.seed(42)
        noise = np.random.normal(0, 2.2, n)
        
        # COVID-19 shock in March - May 2020 (Noticeable sudden shock for educational realism)
        # Apr 2020 was month index 99
        shock = np.zeros(n)
        shock[99] = -14.0  # Apr 2020 strict lockdown
        shock[100] = -8.0  # May 2020 gradual unlock
        
        electricity_bu = trend + seasonal_component + noise + shock
        electricity_bu = np.round(electricity_bu, 2)
        
        df_univ = pd.DataFrame({
            'Date': dates.strftime('%Y-%m-%d'),
            'Electricity_Consumption_BU': electricity_bu,
            'Year': dates.year,
            'Month': dates.strftime('%b')
        })
        
        file_path = os.path.join(datasets_dir, 'india_electricity_monthly.csv')
        df_univ.to_csv(file_path, index=False)
        
        ds1 = Dataset.objects.create(
            name="Monthly India Electricity Consumption (2012-2023)",
            description="Official-style monthly national electricity demand in Billion Units (BU). Features rising macroeconomic trend, strong summer cooling seasonality (peaks in May-June), and the 2020 lockdown shock.",
            file='datasets/india_electricity_monthly.csv',
            file_type='csv',
            rows_count=len(df_univ),
            columns_json=[
                {'name': 'Date', 'type': 'date', 'sample': '2012-01-01'},
                {'name': 'Electricity_Consumption_BU', 'type': 'float', 'sample': '68.42'},
                {'name': 'Year', 'type': 'int', 'sample': '2012'},
                {'name': 'Month', 'type': 'string', 'sample': 'Jan'}
            ],
            is_sample=True
        )
        created_datasets.append(ds1)

    # 2. Multivariate Energy, Weather & Industrial Macro (Mode B Canonical Example)
    multi_qs = Dataset.objects.filter(name="India Energy, Weather & Macro Dynamics (2012-2023)", is_sample=True)
    if not multi_qs.exists():
        dates = pd.date_range(start='2012-01-01', end='2023-12-01', freq='MS')
        n = len(dates)
        np.random.seed(101)
        
        # 1. Temperature in Celsius (Mean national ambient temp: 18C in Jan to 34C in May)
        temp_cycle = np.array([19.2, 22.4, 27.5, 32.1, 34.6, 33.2, 30.1, 29.3, 29.0, 27.1, 23.4, 20.0])
        temp_season = np.tile(temp_cycle, int(np.ceil(n / 12)))[:n]
        temp_series = np.round(temp_season + np.random.normal(0, 0.8, n), 1)
        
        # 2. Industrial Production Index (IIP ~ 90 in 2012 to 145 in 2023, seasonal dips in monsoon)
        iip_trend = np.linspace(92.0, 142.0, n)
        iip_cycle = np.array([-2.0, 0.5, 8.0, 2.0, 4.0, 1.0, -3.5, -4.0, -1.0, 5.0, 3.0, 6.0])
        iip_season = np.tile(iip_cycle, int(np.ceil(n / 12)))[:n]
        iip_series = np.round(iip_trend + iip_season + np.random.normal(0, 1.8, n), 1)
        
        # 3. Electricity Demand BU - driven by Temperature (lagged and current cooling) and IIP
        elec_trend = np.linspace(72.0, 132.0, n)
        elec_weather_impact = (temp_series - 25.0) * 1.8
        elec_iip_impact = (iip_series - 100.0) * 0.35
        elec_series = np.round(elec_trend + elec_weather_impact + elec_iip_impact + np.random.normal(0, 1.5, n), 2)
        
        # 4. Renewable Generation BU (Solar + Wind: huge ramp-up from 2017 onwards, peak windy monsoon)
        ramp_part1 = np.linspace(3.0, 7.0, 60)
        ramp_part2 = np.linspace(7.0, 28.0, n - 60)
        renew_ramp = np.concatenate([ramp_part1, ramp_part2])
        wind_season = np.array([-1.0, -1.2, -0.5, 0.2, 1.8, 3.5, 4.2, 3.8, 1.5, -0.5, -1.2, -1.0])
        renew_season = np.tile(wind_season, int(np.ceil(n / 12)))[:n]
        renew_series = np.round(np.maximum(1.5, renew_ramp + renew_season + np.random.normal(0, 0.9, n)), 2)
        
        df_multi = pd.DataFrame({
            'Date': dates.strftime('%Y-%m-%d'),
            'Electricity_Demand_BU': elec_series,
            'Avg_Temperature_C': temp_series,
            'Industrial_Index_IIP': iip_series,
            'Renewable_Generation_BU': renew_series
        })
        
        multi_file_path = os.path.join(datasets_dir, 'india_energy_weather_multivariate.csv')
        df_multi.to_csv(multi_file_path, index=False)
        
        ds2 = Dataset.objects.create(
            name="India Energy, Weather & Macro Dynamics (2012-2023)",
            description="Multivariate monthly system tracking Electricity Demand (BU), Average Temperature (°C), Industrial Production Index (IIP), and Renewable Generation (BU). Perfect for Granger causality, Impulse Response (IRF), and VAR/VARMA joint forecasting.",
            file='datasets/india_energy_weather_multivariate.csv',
            file_type='csv',
            rows_count=len(df_multi),
            columns_json=[
                {'name': 'Date', 'type': 'date', 'sample': '2012-01-01'},
                {'name': 'Electricity_Demand_BU', 'type': 'float', 'sample': '71.5'},
                {'name': 'Avg_Temperature_C', 'type': 'float', 'sample': '19.2'},
                {'name': 'Industrial_Index_IIP', 'type': 'float', 'sample': '91.8'},
                {'name': 'Renewable_Generation_BU', 'type': 'float', 'sample': '3.2'}
            ],
            is_sample=True
        )
        created_datasets.append(ds2)

    return created_datasets
