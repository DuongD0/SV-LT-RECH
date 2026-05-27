# Data Sources

Every external dataset is pulled through `+data/+fetch/cacheFetch.m`, which checksums the response against `manifest.yaml`. Re-runs hit local cache under `data/raw/`. The fetcher fails loudly if a remote source has changed shape rather than silently corrupting downstream results.

## Daily closing prices

| Source | Coverage | Endpoint | Notes |
|---|---|---|---|
| Yahoo Finance CSV | SPX, N225, DJI, DAX, FTSE, AORD, CAC40, BVSP, AEX, BFX | `https://query1.finance.yahoo.com/v7/finance/download/<ticker>?period1=...&period2=...&interval=1d&events=history` | No API key. Tickers per `config/markets.yaml`. |
| investing.com (5-market panel) | VN100, N225, CAC40, ASX, BVSP — same series as RECH-X §4.2 | manual snapshot once, checked into `data/raw/investing/` then loaded | Site blocks programmatic access. We checksum the user-supplied CSV against the manifest. |
| vn.investing.com / vnstock | VN-Index, HNX-Index | `+data/+fetch/fetchVN.m` calls `vnstock` via MATLAB-Python interop | Snapshot date pinned in manifest. |

## Realized volatility

The Oxford-Man Realized Library was decommissioned in July 2022. Two fallbacks are wired in priority order:

1. **Liu, Wang, Tran & Kohn (2025) extended panel** — 31 indices through 2023, used as the primary RV source. Files mirrored at `https://github.com/VBayesLab/RealRECH/...` (data folder).
2. **OMI Wayback snapshot** — `https://web.archive.org/web/20220601000000*/oxford-man.ox.ac.uk/...`. Cached locally on first run.

If neither is available, `+data/+fetch/fetchRealizedLibrary.m` aborts with a clear error. Squared returns are NOT a silent fallback — that would corrupt §10 metrics.

## Macro covariates

| Source | Variable | Endpoint |
|---|---|---|
| FRED | Industrial Production (INDPRO), Unemployment (UNRATE), CPI (CPIAUCSL), M2 (M2SL), Term Spread (T10Y2Y), Effective FFR | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<series>` |
| FRED | WTI Crude Oil (DCOILWTICO), Gold (GOLDAMGBD228NLBM) | same |
| FRED | USD/VND, USD/JPY, etc. | `DEXVNUS`, `DEXJPUS`, ... |
| CBOE | VIX | `https://cdn.cboe.com/data/us/indices/daily_price_history/VIX_History.csv` |
| Yahoo / FRED | Bitcoin USD | yfinance `BTC-USD` |

## Frequency alignment

- Daily covariates: forward-filled across non-trading days.
- Monthly / quarterly macros (industrial production, unemployment, CPI, M2, GDP): aggregated via MIDAS Beta-polynomial weighting (Ghysels et al. 2006) in `+data/+preprocess/`. Used only for the optional Stage 5 MIDAS-LASSO extension.

## Citation policy

Every dataset's license / terms-of-use is restated in `manifest.yaml`. Yahoo Finance and investing.com are personal/research use only — not redistributable. We commit fetchers, not data.
