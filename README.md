# Forecasting Studio: Automated Time Series Analyst

An analyst-grade fullstack Django web application designed to answer the fixed set of questions that matter for any time series dataset. Built using Django's **MVT architecture**, **ORM**, **Django Template Language (DTL)**, **Django Forms**, **Statsmodels**, **Pandas**, and **Plotly.js**.

---

## The Concept

Think of the application as an automated time series analyst who answers a fixed set of critical questions about any univariate or multivariate dataset you provide:

### Mode A (One Series): The Univariate Report
| # | Question Answered | How the App Displays It | India Electricity Benchmark Example |
|---|---|---|---|
| **1** | **What does my data look like?** | Interactive timeline, descriptive stats, data-quality audit | *"144 monthly points, Jan 2012 to Dec 2023, 0 gaps"* |
| **2** | **What is it made of?** | Classical & STL decomposition into Trend, Seasonal cycle, and Residual | *"Rising trend (strength 0.98), strong yearly cycle with peaks in May-June"* |
| **3** | **Is it stable or drifting?** | Augmented Dickey-Fuller (ADF) test table, stationarity badge, recommended $d$ and $D$ | *"Raw series: non-stationary (p=0.84). After differencing: stationary (p < 0.001), so d=1 and D=1"* |
| **4** | **How does it remember its past?** | Dual ACF/PACF correlograms with $\pm 1.96/\sqrt{N}$ Bartlett bounds + **Reading Assistant** | *"Spikes at lag 12 and 24 indicate annual memory; PACF cuts off cleanly, suggesting AR(1)-like behaviour"* |
| **5** | **Which model fits best?** | Out-of-sample holdout validation chart & Candidate Leaderboard (AR, MA, ARMA, ARIMA, SARIMA) ranked by RMSE | *"SARIMA(0,1,1)(0,1,1)[12] achieved lowest holdout error on the last 12 points (RMSE: 2.61)"* |
| **6** | **Can I trust the model?** | Residual series plot, Residual ACF correlogram, and Ljung-Box white noise test | *"Residuals look like white noise (p = 0.48 > 0.05), so the model successfully captured the underlying structure"* |
| **7** | **What happens next?** | Future forecast projection with 95% and 80% confidence ribbons + 1-click CSV download | *"Next 12 months: expected values, each with a 95% range"* |

### Plain-Language Summary Box
The top card of every report converts the statistical metrics into plain-language sentences (rule-based templates, no external LLM required):
> *"Your series shows a prominent rising trend and a strong yearly cycle (peaks around May, troughs around Dec). It required 1 regular differencing order (d=1) and 1 seasonal order (D=1) to reach stationarity. The best performing model is SARIMA(0,1,1)(0,1,1)[12] (holdout RMSE 2.61 on the last 12 evaluation points). Expected value for next period is 129.45 BU, highly likely to land between 123.80 BU and 135.10 BU (95% confidence interval)."*

### What the App Will NOT Tell You (Guardrails in the UI)
1. **Patterns, Not Causes:** Forecasts extrapolate historical patterns; they do not prove why changes occur or what economic levers drive them.
2. **Sudden Shocks & Regime Shifts:** Models trained on calm periods fail through black swans (e.g., COVID lockdowns, unexpected policy shocks).
3. **Certainty is an Illusion:** Single line point forecasts hide true risk. Always plan against the 95% confidence interval, never just the centerline.
4. **Weak Structure Warning:** If a series is essentially a random walk with high noise, honest wide bands and warnings are presented.

---

### Mode B (Several Series): Multivariate VAR / VARMA Studio
Evaluates interdependent systems (e.g. Electricity Demand, Temperature, Industrial Output, and Renewable Generation):
- **Do these series move together?** Pearson correlation matrix heatmap and standardized ($Z$-score) co-movement overlay.
- **Does A help predict B?** Pairwise Granger causality test matrix and plain-language interpretation cards (*"Temperature -> Electricity Demand: p = 0.008, so temperature carries strong predictive power"*).
- **How long does a shock last?** Orthogonalized Impulse Response Functions (IRF) tracing dynamic feedback over horizons.
- **What drives the uncertainty?** Forecast Error Variance Decomposition (FEVD) stacked percentage shares.
- **What are the joint forecasts?** Coordinated multi-series forecasts with individual 95% bands and downloadable joint CSV.

---

## Fullstack Django Architecture

- **M (Models / ORM):**
  - `Dataset`: Uploaded or benchmark CSV/Excel metadata, row count, detected schema.
  - `AnalysisRun`: Stores all metrics, decomposed components, ADF stats, ACF/PACF arrays, residuals, and forecasts in structured JSON fields for fast retrieval and auditing.
  - `ModelCandidate`: Leaderboard items linked to each run storing model family, order, AIC, BIC, holdout RMSE, MAE, MAPE, rank, and selection flag.
  - `MultivariateRun`: Mode B system configuration, Granger matrix, IRF curves, FEVD shares, and joint forecasts.
- **V (Views):**
  - Class-based views (`DashboardView`, `DatasetUploadView`, `DatasetDetailView`, `UnivariateRunCreateView`, `UnivariateRunDetailView`, `MultivariateRunCreateView`, `MultivariateRunDetailView`, `RunCompareView`).
  - Streaming CSV export views (`UnivariateExportCSVView`, `MultivariateExportCSVView`).
- **T (Templates & DTL):**
  - Responsive layouts using Tailwind CSS CDN, Lucide Icons, and Plotly.js for zoomable, interactive charts with tooltips.
  - Reusable components: Stationarity badges, Leaderboard tables, Reading Assistant callouts, and Side-by-side run comparison.
- **Django Forms:**
  - `DatasetUploadForm`: Handles file uploads with validation (.csv, .xlsx).
  - `UnivariateConfigForm`: Dynamic choice fields bound to the dataset's detected columns, seasonal periods, holdout splits, and forecast horizons.
  - `MultivariateConfigForm`: Multi-column selection checkboxes, lag order constraints, and model type selector (VAR vs VARMA).

---

## Quickstart & Running Locally

### 1. Activate Environment
```bash
source .venv/bin/activate
```

### 2. Apply Migrations & Initialize Benchmarks
```bash
python manage.py migrate
python manage.py init_studio
```

### 3. Run the Development Server
```bash
python manage.py runserver 0.0.0.0:8000
```
Open **`http://127.0.0.1:8000`** in your browser.

### 4. Run Test Suite
```bash
python manage.py test
```
All unit and integration tests run in under 1 second.
