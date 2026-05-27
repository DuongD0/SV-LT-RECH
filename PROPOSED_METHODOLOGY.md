# Proposed Methodology — Stochastic-Volatility + Deep-Learning Hybrid for Daily Stock-Index Volatility Forecasting

**Project**: Replicate the methodology of Nguyen et al. (2022) *RECH* and Nguyen, Nguyen & Tran (2024) *RECH-X* (Finance Research Letters, [DOI 10.1016/j.frl.2024.106145](https://doi.org/10.1016/j.frl.2024.106145)), but replace the **GARCH backbone** with a **Stochastic Volatility (SV)** backbone and combine it with a modern deep-learning recurrent layer.

**Date** (current revision): 2026-05-16
**Initial draft**: 2026-05-15

### Revision Log

| Date | Change | Why |
|------|--------|-----|
| 2026-05-15 | Initial draft (§§1–16) | Prior session output. |
| 2026-05-16 | Re-verified both reference papers end-to-end; ran fresh 2025–2026 SOTA searches; elevated **LSTM from "robustness check" to co-primary architecture alongside SRN**; added §8.6 (novelty vs prior art matrix); added concrete SMC hyperparameters (§9.7); appended very recent references (Liu-Wang-Tran-Kohn 2025 *Econ. Modelling*, Perekhodko & Ślepaczuk 2025 arXiv, Yin et al. 2025 *Comp. Econ.*, Unified GARCH-RNN 2025 arXiv). | The most direct precedents (SR-SV 2023 *JBES*, RealRECH 2025 *Econ. Modelling*, Perekhodko 2025) all use **LSTM**, not SRN, and successfully estimate it with SMC at the same sample sizes we target. Paper B §3.4.2 simulations also show LSTM > GRU > SRN at variable selection. Restricting to SRN would understate the model class. |

### Reference papers fully read (verified end-to-end this session)

- **Paper A** — Nguyen, H. T., Nguyen, H. & Tran, M.-N. (2024). *Deep learning enhanced volatility modeling with covariates* (RECH-X). *Finance Research Letters* 70, 106145. [DOI](https://doi.org/10.1016/j.frl.2024.106145). Read §§1–5 + Appendices A–C; tables and equations cross-checked against PaddleOCR transcript.
- **Paper B** — *Building Statistical Machine Learning Models for Volatility Forecasting: Applications in the Stock Market*. NEU Hanoi thesis, 2023; published outputs: Paper A above + JFAR Issue 285 (April 2025) + Economic Modelling submission. Read §§I (introduction & objectives) + §3.1 (literature) + §3.2 (theoretical framework, GARCH/RealGARCH/GARCH-X) + §3.4 (RECH-X + LSTM-GARCH-MIDAS-LASSO simulation and 5-market application: AORD, BVSP, DJI, DAX, N225) + §3.5 (Vietnam application: VN-Index and HNX-Index, 2,262 obs Jan 2016–Jan 2025, kurtosis 7.73 and 7.91). Confirmed all numerical anchors used in this methodology.

---

## 0. Executive Summary

We will build **SV-LT-RECH** — a Stochastic Volatility model with **L**everage + Student's-**t** innovations whose latent log-variance state is augmented by a **R**ecurrent neural network that ingests **e**xogenous covariates and historical state information (the SV analog of RECH-X). When high-frequency intraday data is available, we extend it to **RSV-LT-RECH** by adding a Realized-Volatility measurement equation (the SV analog of RealRECH).

### 0.1 One-page summary (read this first)

- **Question.** *Can we improve daily stock-index volatility forecasts by replacing the GARCH backbone of RECH-X with a Stochastic Volatility backbone, while still being able to inject exogenous covariates through an RNN-augmented log-variance state?*
- **Model class.** SV-LT-RECH = SV (Taylor 1982 / Kim-Shephard-Chib 1998) **+** Leverage (Yu 2005; Omori et al. 2007) **+** Student-t innovations (ν > 2) **+** RNN-augmented log-variance state (RECH-style additive ωₜ). Two co-primary RNN cells: **SRN** (interpretable, fewest weights) and **LSTM** (most expressive; same cell SR-SV/RealRECH use successfully).
- **Pre-fitting pipeline** (your "ARIMA-something" intuition, made precise): (1) log returns ×100; (2) ADF + PP + KPSS stationarity tests; (3) mean equation — μₜ = 0 default, ARMA(p,q) fallback if Ljung-Box rejects (note: returns are I(0) so it is ARMA, not ARIMA — the "I" was the price→return differencing already done in step 1); (4) residual diagnostics (Ljung-Box, McLeod-Li, ARCH-LM, JB, BDS) to confirm volatility clustering, heavy tails, nonlinearity; (5) covariate standardization with **training-only** statistics (no leakage).
- **Estimation.** Bayesian Sequential Monte Carlo with likelihood annealing (in-sample) and data annealing (rolling OOS). Same scheme as RECH-X, SR-SV, and RealRECH. N = 5,000 parameter particles × M = 200 latent-state particles; adaptive temperature schedule; 10 random-walk Metropolis moves per anneal step.
- **Evaluation.** Five predictive scores (PPS, QS, MSE, MAE, R²LOG) per Paper A; QLIKE robust loss as primary; Diebold-Mariano and Model Confidence Set for inference; log marginal likelihood as in-sample Bayes-factor metric.
- **Empirical plan.** Stage 1 simulation study (30 replicates). Stage 2 ten major markets with RV (Oxford-Man). Stage 3 five markets with VIX/OIL/GOLD/EXR covariates (matches Paper A §4.2). Stage 4 Vietnamese frontier market (VN-Index + HNX-Index, 2,262 obs, four covariates per Paper B §3.5.3). Stage 5 (optional) MIDAS extension with monthly macro covariates.
- **Net novelty.** First model combining SV + leverage + Student-t + RNN-with-covariates + (optional) realized-vol measurement + joint Bayesian SMC + emerging-market application. See §8.6 for the explicit prior-art matrix.
- **Deliverables.** MATLAB toolbox (extending VBayesLab), reproducible YAML-configured experiment runner, Tables T1–T7 + Figures F1–F4 (§12), and a paper draft targeting *Finance Research Letters* or *Journal of Forecasting*.

### 0.2 Critical naming clarification (your "ARIMA something" question)

The user prompt mentioned "Mean equations + Volatility/Variance equations ARIMA something". To be precise:

| Component | What it is | What model name |
|-----------|-----------|-----------------|
| **Mean equation** for daily returns | Often just μₜ = 0; if not, an **ARMA(p,q)** on stationary returns | ARMA (not ARIMA — the I = integrated step was differencing prices into returns, which we already did at step 1) |
| **Variance / Volatility equation** | The conditional variance dynamic | **GARCH-family** OR **SV-family**. We use SV-family (specifically SV-LT plus an RNN augmentation). |

ARIMA is the wrong label for the variance equation; it would apply if we modeled the volatility level itself as an integrated AR-MA process (e.g., HAR-style log-RV dynamics), which is *not* the SV framework. We use ARMA on the **mean** if needed, and **SV** on the **variance**.

The full pipeline is:

```
Raw prices Pₜ
   │
   ▼   §1  log returns, ×100 scaling
rₜ = 100·ln(Pₜ/Pₜ₋₁)
   │
   ▼   §2  stationarity tests (ADF, PP, KPSS)
   ▼   §3  Mean equation: μₜ = 0 (default) OR ARMA(p,q)
ε̂ₜ = rₜ − μ̂ₜ
   │
   ▼   §4  Residual diagnostics: Ljung-Box, McLeod-Li, ARCH-LM, Jarque-Bera, BDS
   │       (confirm ARCH effects + heavy tails + nonlinearity)
   ▼   §5  Exogenous covariate prep: standardize (in-sample stats only),
   │       sign handling, frequency alignment
   ▼   §6  Train/test split 75/25 chronological; expanding window for OOS
   │
   ▼   §7  Baseline volatility models: SV, SV-L, SV-t, SV-LT, GARCH(1,1), GJR, EGARCH
   ▼   §8  PROPOSED: SV-LT-RECH (primary)  and  RSV-LT-RECH (when RV available)
   ▼   §9  Estimation: likelihood-annealing SMC (in-sample) + data-annealing SMC (OOS)
   ▼  §10  Forecast evaluation: PPS, QS, MSE, MAE, R²LOG  +  Diebold-Mariano, MCS
   ▼  §11  Robustness: GRU variant, jumps, regime breaks, alt priors
```

Each section below gives the test, the decision rule, the equations, and the rationale.

---

## 1. Data

### 1.1 Universe

Daily closing prices of stock indices. Default panel (matches the reference papers so results are directly comparable):

| Symbol  | Index                          | Source            |
|---------|--------------------------------|-------------------|
| SPX     | S&P 500                        | Oxford-Man / Yahoo|
| N225    | Nikkei 225                     | Oxford-Man / Yahoo|
| DJI     | Dow Jones Industrial Average   | Oxford-Man / Yahoo|
| DAX     | Frankfurt DAX                  | Oxford-Man / Yahoo|
| FTSE    | UK FTSE 100                    | Oxford-Man / Yahoo|
| AORD    | Australia All Ordinaries       | Oxford-Man / Yahoo|
| CAC40   | Euronext Paris CAC 40          | Oxford-Man / Yahoo|
| BVSP    | São Paulo Bovespa              | Oxford-Man / Yahoo|
| AEX     | Amsterdam                      | Oxford-Man        |
| BFX     | Brussels BEL-20                | Oxford-Man        |
| VN-Index| Vietnam HOSE                   | investing.com     |
| HNX-Index| Vietnam HNX                   | investing.com     |

Each series: ~2,000–2,500 daily closing prices. Period: typically Jan 2015 – Jan 2025 (covers COVID-19 shock).

### 1.2 Exogenous covariates

Same-frequency (daily) covariates that the literature supports as drivers of equity volatility:

| Covariate | Symbol | Sign expectation on volatility |
|-----------|--------|-------------------------------|
| Realized Volatility (5-min)| RV    | + (when available)           |
| CBOE Volatility Index      | VIX   | +                             |
| WTI Crude Oil log-return   | OIL   | mixed (market-specific)       |
| Gold spot log-return       | GOLD  | mixed                         |
| USD exchange rate (vs local)| EXR  | mixed                         |
| Bitcoin log-return         | BTC   | + (Vietnam application)       |

Optional low-frequency macro covariates (monthly / quarterly) — handled via MIDAS aggregation (Engle, Ghysels & Sohn 2013): industrial production, unemployment, CPI inflation, M2, term spread, GDP.

### 1.3 Storage & engineering

- Wide CSV per series with the schema: `date, open, high, low, close, volume, realized_volatility`.
- Side CSV per covariate, same date index.
- Persist all preprocessed tensors to disk (`.npz` or `.parquet`) keyed by `(market, freq, train|test)` so estimation is fully reproducible.

---

## 2. Stationarity testing (raw and transformed series)

Volatility models require **I(0)** input. Prices Pₜ are I(1); returns rₜ must be I(0).

### 2.1 Tests

Apply to both raw prices and to returns:

| Test                        | H₀                          | Decision @ α = 5%                |
|-----------------------------|-----------------------------|----------------------------------|
| Augmented Dickey–Fuller (ADF)| unit root present          | reject (p < 0.05) ⇒ stationary  |
| Phillips–Perron (PP)        | unit root present           | reject (p < 0.05) ⇒ stationary  |
| KPSS                        | (level/trend) stationarity  | fail-to-reject (p > 0.05) ⇒ stationary |

### 2.2 Decision rule

- **ADF + PP both reject** AND **KPSS does not reject** ⇒ series is stationary, proceed.
- **Conflict** (e.g. ADF rejects but KPSS also rejects) ⇒ borderline / trend-stationary; either detrend or use first-difference. For log-returns this rarely happens.

Expected outcome: prices Pₜ fail all three tests (I(1)); log-returns rₜ pass all three (I(0)).

Implementation: `statsmodels.tsa.stattools.adfuller`, `kpss`, `phillips_perron` (or `arch` package).

---

## 3. Mean Equation (μₜ)

For daily equity returns the conditional mean is small and often economically negligible. We adopt the **two-option** protocol:

### 3.1 Default: μₜ = 0

```
rₜ = εₜ           (zero-mean assumption)
```

This is what RECH and RECH-X use (Nguyen et al. 2022, 2024). Returns are treated as residuals and feed directly into the variance equation. Justification: efficient-market weak form; for daily index returns the mean-to-volatility ratio is typically << 1.

### 3.2 Fallback: ARMA(p,q) — Box-Jenkins

Used only if Ljung-Box on rₜ at lag 10 or 20 rejects (§4). The mean equation becomes:

```
rₜ = c + Σᵢ φᵢ rₜ₋ᵢ + Σⱼ θⱼ εₜ₋ⱼ + εₜ
```

**Order selection protocol:**
1. Plot ACF and PACF of rₜ; inspect cut-offs.
2. Fit ARMA(p,q) for p, q ∈ {0,1,2,3,4,5}.
3. Pick (p*,q*) by minimum **BIC** (BIC preferred over AIC for parsimony — Tsay 2010).
4. Verify residuals via Ljung-Box at multiple lags (10, 20, 30).

> **Terminology note.** What the problem statement called "ARIMA" is actually the **mean equation**, and on returns it is almost always ARMA (returns are already I(0); the "I" in ARIMA is for the price-to-return differencing, which we already did in §1). The variance/volatility equation is GARCH/SV, **not** ARIMA.

### 3.3 Implementation

- `statsmodels.tsa.arima.model.ARIMA(rₜ, order=(p,0,q))`
- Final mean-equation residuals ε̂ₜ become the input series for §7–8.
- When μₜ = 0, ε̂ₜ ≡ rₜ.

---

## 4. Residual Diagnostics (must pass before fitting variance model)

Five diagnostics on ε̂ₜ that motivate each design choice downstream:

| Test | Detects | H₀ | Lag/Param | Action if rejected |
|------|---------|----|-----------|--------------------|
| **Ljung-Box Q(ℓ)** on ε̂ₜ | residual serial correlation | white noise | ℓ = 10, 20 | re-specify mean equation |
| **Ljung-Box Q(ℓ)** on ε̂ₜ² (McLeod-Li) | ARCH effects | no ARCH | ℓ = 10, 20 | proceed — volatility clustering exists |
| **Engle's ARCH-LM(q)** | conditional heteroskedasticity | homoscedastic | q = 5, 10 | **mandatory rejection** to justify SV/GARCH |
| **Jarque-Bera** | non-normality | normal | — | use Student's t innovations (motivates SV-t) |
| **BDS** (Brock-Dechert-Scheinkman) | nonlinear/iid departures | iid | m = 2,3,4,5; ε = 0.5σ | motivates the DL component |

### 4.1 Why each test matters

- **McLeod-Li / ARCH-LM rejection** is the *prerequisite* for any SV or GARCH model. Without it the variance model is gratuitous.
- **JB rejection** (heavy tails, kurtosis usually 5–10 for daily index returns) ⇒ use **Student's t innovations** (df ν ∈ [4, 12]). The reference papers report kurtosis ≈ 7.7 for VN-Index, 7.9 for HNX-Index — heavy tails are normal in this data.
- **BDS rejection** is the empirical justification that linear GARCH/SV is insufficient and a **neural network correction** is warranted.

### 4.2 Implementation

- `statsmodels.stats.diagnostic.acorr_ljungbox`
- `statsmodels.stats.diagnostic.het_arch` (ARCH-LM)
- `scipy.stats.jarque_bera`
- `nolds` or custom BDS routine

All diagnostic results are reported in the empirical Table 1 of the final paper.

---

## 5. Exogenous covariate preprocessing

### 5.1 Stationarity transformations

For each covariate xₜ apply the same ADF / KPSS protocol from §2.

- **VIX**: level is usually I(0) (mean-reverting). Use level. If borderline, use Δlog VIX.
- **Oil, Gold, FX, BTC**: use log-returns (price changes are I(1), returns are I(0)).
- **RV**: use **log(RV)** to stabilize variance and approximate normality (standard Andersen-Bollerslev-Diebold-Labys 2003).

### 5.2 Standardization (CRITICAL — no leakage)

For each covariate, compute mean μ_x and std σ_x using **training data only**:

```
x_t^{std} = (x_t − μ_x^{train}) / σ_x^{train}      (applied to BOTH train and test)
```

> ⚠ **Never** compute μ_x, σ_x over the full sample — that leaks future information into the training set and inflates the apparent out-of-sample accuracy.

### 5.3 Sign handling for GARCH-X (relevant for benchmark only)

Francq & Thieu (2019) require nonnegative inputs into the GARCH-X variance recursion. For signed covariates, use:

```
x_t → x_t^2     or    x_t → |x_t|
```

This is **not needed** for the SV-LT-RECH proposed model, because the RNN inside the latent log-variance state can absorb signed inputs naturally (ReLU activation handles sign).

### 5.4 Frequency alignment

- **Same frequency as returns (daily)**: feed directly. This is the primary mode (matches RECH-X).
- **Lower frequency (monthly macro)**: aggregate via the **MIDAS Beta-polynomial weighting** of Ghysels et al. (2006), or simple forward-fill for naive alignment. MIDAS treatment is the principled choice (used in the LSTM-GARCH-MIDAS-LASSO extension of Paper B).

### 5.5 Missing-data handling

- Forward-fill across non-trading days (weekends, market holidays).
- Drop any date where the **target return rₜ is missing**.
- Document the calendar-alignment policy explicitly in the data appendix.

---

## 6. Train / Test split

| Setup | Use case |
|-------|----------|
| Fixed 75/25 chronological | In-sample fitting & one-shot OOS evaluation (RECH paper convention: first 1,500 of 2,000 obs train) |
| Fixed 80/20 chronological | Sensitivity check |
| **Expanding window** rolling re-estimation | True one-step-ahead OOS evaluation; matches RECH-X protocol |

**No shuffling.** Returns are temporally dependent; shuffling destroys the volatility-clustering signal that the model exists to capture.

For Vietnamese application (2,262 obs): 1,500 train / 762 test (matches Paper B §3.5.3).

---

## 7. Baseline & Benchmark Models

We fit five benchmarks so the contribution of the proposed SV-LT-RECH is clearly isolated.

### 7.1 GARCH(1,1) with Student's t innovations

```
rₜ = σₜ εₜ,        εₜ ~ t(ν), iid
σₜ² = ω + α rₜ₋₁² + β σₜ₋₁²
```
Constraints: α, β > 0; α·ν/(ν−2) + β < 1; ν > 2.

### 7.2 GJR-GARCH (leverage GARCH benchmark)

```
σₜ² = ω + α rₜ₋₁² + γ rₜ₋₁² · 1{rₜ₋₁ < 0} + β σₜ₋₁²
```

### 7.3 GARCH-X (the RECH-X benchmark with exogenous covariates)

```
σₜ² = ω + α rₜ₋₁² + β σₜ₋₁² + πᵀ xₜ₋₁
```

### 7.4 RealGARCH (Hansen et al. 2012, when RV is available)

```
σₜ² = ω + β σₜ₋₁² + γ RVₜ₋₁
log RVₜ = ξ + φ σₜ² + τ₁ εₜ + τ₂ [(ν−2)/ν · εₜ² − 1] + uₜ
```

### 7.5 Plain SV — three increasingly rich versions

**(a) Log-SV (Taylor 1982 / Kim-Shephard-Chib 1998):**

```
rₜ = exp(hₜ/2) · εₜ,       εₜ ~ N(0,1)
hₜ = μ + φ (hₜ₋₁ − μ) + σ_η ηₜ,   ηₜ ~ N(0,1),  |φ| < 1
```

**(b) SV-t** — adds Student's t innovations on the return equation, ν ∈ [4, 30]:

```
εₜ ~ t(ν)
```

**(c) SV-L-t** — adds leverage correlation between εₜ and ηₜ₊₁:

```
corr(εₜ, ηₜ₊₁) = ρ           (typical ρ ∈ [-0.7, 0])
```

These are the **SV-side baselines** we beat in the empirical section.

---

## 8. PROPOSED MODELS

### 8.1 Why Stochastic Volatility over GARCH

| Property | GARCH(1,1) | SV |
|----------|-----------|----|
| Variance equation deterministic given past? | Yes | **No — has its own innovation** |
| Vol-of-vol behaviour | Mechanical | **Genuine, smoothly distributed** |
| Leverage modeling | Hard threshold (GJR) | **Continuous correlation ρ** |
| Latent state amenable to NN injection | Awkward (σ²ₜ is observable from past) | **Natural — hₜ is latent, the RNN augments it directly** |
| Fit to realized variance distribution | Under-disperses upper tail | Better empirical fit |
| Estimation complexity | OLS / QMLE, closed form | **Particle filter / SMC required** |

The third row is decisive. SV's latent log-variance hₜ is exactly the right injection point for a neural network — the RNN models *residual nonlinear deviations from the AR(1) log-variance recursion*, which is structurally the same role the RNN played for σₜ² in RECH.

### 8.2 Primary model — **SV-LT-RECH** (SV with Leverage + Student's t + Recurrent Conditional component)

Full specification:

**Observation equation**
```
rₜ = exp(hₜ/2) · εₜ,            εₜ ~ t(ν),   ν > 2
```

**Latent log-variance equation (RNN-augmented)**
```
hₜ = μ + φ (hₜ₋₁ − μ) + ωₜ + σ_η ηₜ,    ηₜ ~ N(0,1)
                                        |φ| < 1
                                        σ_η > 0
                                        corr(εₜ, ηₜ₊₁) = ρ   (leverage)
```

**Recurrent component (the deep-learning layer)**
```
ωₜ = β₀ + β₁ · sₜ
sₜ = Ψ( vᵀ xₜ + w_h sₜ₋₁ + b ),   s₁ ≡ 0
Ψ(u) = max{u, 0}      (ReLU)
xₜ = (hₜ₋₁, rₜ₋₁, ωₜ₋₁,  zₜ₋₁)ᵀ
```

where `zₜ₋₁` is the vector of exogenous covariates (VIX, OIL, GOLD, EXR, BTC, ...), each standardized as in §5.2.

**Identifiability constraints**:
- ν > 2 (finite variance of εₜ)
- |φ| < 1 (stationarity)
- σ_η > 0
- ρ ∈ (−1, 1)
- β₀ ≥ 0, β₁ ≥ 0 (long-term volatility level non-negative — analog of RECH)

**Parameter set:**

θ = (μ, φ, σ_η, ρ, ν, β₀, β₁, v_h, v_r, v_ω, v_z, w_h, b)

For one exogenous covariate the input vector is 4-dimensional, so |θ| = 13. For four covariates (oil/gold/VIX/FX as in RECH-X §4.2) it grows to 16. Hidden size is **1** (scalar sₜ) by default — same as RECH — to keep the SMC posterior tractable.

**Why this architecture (justification from the research briefing, updated 2026-05-16)**:

- **Two co-primary RNN cells: SRN *and* LSTM** — fit both, let log marginal likelihood and the Model Confidence Set decide per market. Direct precedents at our sample size all support both being tractable under SMC:
  - **SR-SV** (Nguyen, Tran, Gunawan & Kohn, 2023 *JBES* 41(2): 414–428) augments the SV log-variance with an LSTM and estimates it with PMMH/SMC on five major indices (DAX, HSI, CAC, SPX, TSX) at 2,000–3,000 obs — direct existence proof that **LSTM-inside-SV is estimable**.
  - **RealRECH** (Liu, Wang, Tran & Kohn, 2025, *Economic Modelling* 142, 106922; arXiv 2302.08002) replaces the SRN in RECH with **LSTM** and adds a realized-measurement equation; SMC with annealing converges on 31 indices including the COVID period.
  - **Paper B §3.4.2.4 simulation** finds **LSTM > GRU > RNN** at LASSO variable selection accuracy (100% / 60% / 80% on θ₁; LSTM dominates everywhere).
  - **Perekhodko & Ślepaczuk** (Dec 2025, arXiv 2512.12250) stack SV outputs as LSTM features on SPX 1998–2024 and beat both standalone SV and LSTM; their hybrid is *feature-stack*, ours will be *joint Bayesian* — a strict improvement.
- **Additive integration**, not multiplicative: keeps hₜ Gaussian-conditional given sₜ, which means the particle filter's importance weights remain numerically stable. This is the same trick RECH-X uses; the joint posterior remains amenable to likelihood-annealing SMC.
- **ReLU (SRN) or default LSTM gates**: ReLU is standard in RECH and preserves sign asymmetry of exogenous shocks; LSTM gates (sigmoid / tanh) handle the persistence/forget trade-off natively.
- **Hidden state sₜ scalar (or up to 5 dims)**: keep the posterior tractable. We start with hidden size = 1 (matches RECH-X / SR-SV / RealRECH) and **only** scale up to 3 or 5 if simulation §11.1 confirms the parameters are identifiable.
- **Why NOT a deeper / wider net**: at T ≈ 2,000 daily obs the bias-variance frontier is set by data, not capacity. Increasing capacity raises posterior multimodality (label-switching of RNN-weight signs) and inflates SMC compute by O(N · M · |θ|) per anneal step. Hidden size 1 is the proven sweet spot in all four direct precedents above.

### 8.3 Robustness model — **RSV-LT-RECH** (Realized SV variant for when high-freq RV is available)

Adds a measurement equation linking observed log realized volatility to the latent state:

```
log RVₜ = ξ + φ_RV · hₜ + τ₁ εₜ + τ₂ [(ν−2)/ν · εₜ² − 1] + uₜ
uₜ ~ N(0, σ_u²)
```

The latent log-variance hₜ then has **two** pieces of information per period (rₜ AND log RVₜ), which sharpens identification dramatically. This is the SV analog of RECH-X with realized measures (RECH-X §4.1, Paper A Tables 1-5).

### 8.4 Co-primary model — **SV-LT-LSTM-RECH** (LSTM cell, same SV backbone)

Identical to §8.2 except sₜ is computed by an LSTM cell. Conditioning on sₜ, the SV log-variance equation remains Gaussian, so the particle filter stays bootstrap-feasible.

```
fₜ = σ(W_f · [xₜ, sₜ₋₁] + b_f)            # forget gate
iₜ = σ(W_i · [xₜ, sₜ₋₁] + b_i)            # input gate
oₜ = σ(W_o · [xₜ, sₜ₋₁] + b_o)            # output gate
c̃ₜ = tanh(W_c · [xₜ, sₜ₋₁] + b_c)         # candidate cell state
cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ                  # cell state update
sₜ = oₜ ⊙ tanh(cₜ)                         # hidden state
ωₜ = β₀ + β₁ · sₜ                          # additive injection (same as SRN variant)
```

with `xₜ = (hₜ₋₁, rₜ₋₁, ωₜ₋₁, zₜ₋₁)` as before. Hidden size 1 ⇒ |θ| ≈ 25–28 (vs 13–16 for SRN). All weights N(0, 0.1) prior. Empirically supported as the **strongest** RNN-in-SV instantiation by SR-SV (2023) and RealRECH (2025) on this exact data scale.

### 8.5 Sensitivity model — **SV-LT-GRU-RECH** (GRU substitution)

Identical to §8.2 except sₜ is computed by a GRU cell:

```
zₜ = σ(Wz · [xₜ, sₜ₋₁] + bz)              # update gate
r_gateₜ = σ(Wr · [xₜ, sₜ₋₁] + br)         # reset gate
s̃ₜ = tanh(Wh · [xₜ, r_gateₜ ⊙ sₜ₋₁] + bh)
sₜ = (1 − zₜ) ⊙ sₜ₋₁ + zₜ ⊙ s̃ₜ
```

GRU has ~3× the parameters of SRN but ~25% fewer than LSTM. Reported as a sensitivity check between the two co-primary cells.

### 8.6 Novelty matrix — exactly what we contribute that prior art does not

| Feature | RECH (2022) | RECH-X (2024) | SR-SV (2023) | RealRECH (2025) | Perekhodko & Ślepaczuk (2025) | **SV-LT-RECH (this work)** |
|---------|:----------:|:-------------:|:------------:|:----------------:|:------------------------------:|:--------------------------:|
| Backbone | GARCH | GARCH | **SV** | RealGARCH | SV | **SV** |
| Recurrent cell | SRN | SRN | LSTM | LSTM | LSTM | **SRN + LSTM (co-primary)** |
| Student-t innovations | ✓ | ✓ | ✗ | ✓ | ✗ | **✓** |
| Leverage (corr εₜ, ηₜ₊₁) | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Exogenous covariates in cell | ✗ | ✓ | ✗ | ✗ | ✗ | **✓** |
| Realized-vol measurement eq | ✗ | ✗ | ✗ | ✓ | ✗ | **✓ (RSV-LT-RECH variant)** |
| Joint Bayesian estimation | SMC | SMC | SMC | SMC | **No** (two-stage stack) | **SMC (annealed)** |
| Mixed-frequency MIDAS | ✗ | ✗ | ✗ | ✗ | ✗ | **Optional extension §5.4 / §11.5** |
| Emerging-market focus | ✗ | partial (VN100) | ✗ | ✗ (China indices in follow-on) | ✗ (SPX only) | **✓ (VN-Index + HNX-Index)** |

**Net contribution**: We are the first to combine all five features in one model — *(i)* SV (not GARCH) backbone, *(ii)* leverage correlation, *(iii)* Student-t innovations, *(iv)* exogenous covariates inside an RNN-augmented log-variance state, *(v)* optional realized-vol measurement, *(vi)* joint Bayesian SMC inference, and *(vii)* application to a Vietnamese frontier-market dataset. Each component individually exists; the joint specification does not.

### 8.7 What we do NOT use, and why

| Architecture | Reason for rejection |
|--------------|---------------------|
| Transformer / TFT | Needs ≥ 10⁴ samples or cross-sectional pooling; will overfit on 2,000-obs single series and SMC will not mix. |
| Temporal Convolutional Network (TCN) | Dilated-causal-conv parameter count breaks Bayesian inference at this sample size. Reasonable as a pure-DL benchmark only. |
| DCRNN (Diffusion-Convolutional RNN) | Requires a multi-asset adjacency graph; we model one index at a time. |
| BiLSTM | Uses **future** information — disqualified for genuine 1-step-ahead forecasting. |
| N-BEATS / NHITS | Designed as standalone forecasters with their own basis expansion; cannot slot into a log-variance equation without abandoning the econometric backbone. |

---

## 9. Estimation — Likelihood-Annealing Sequential Monte Carlo (SMC)

This matches the inference scheme used in RECH and RECH-X (Nguyen et al. 2022, 2024) and is the current best practice for SV+NN hybrids (Nguyen, Tran, Gunawan & Kohn 2022 SR-SV in *JBES*; VBayesLab MATLAB toolbox).

### 9.1 Priors

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| ν (degrees of freedom) | Gamma(1, 0.1) | Allows ν ∈ [3, 30]; weakly informative |
| μ (unconditional log-vol) | N(0, 10) | Diffuse |
| φ (persistence) | (1 + φ)/2 ~ Beta(20, 1.5) | Concentrated near 0.95, allows mean reversion |
| σ_η | half-Cauchy(0, 1) | Standard SV prior |
| ρ (leverage) | Uniform(−1, 1) | Equivalent to Fisher-z on full real line |
| β₀, β₁ | Uniform(0, 0.5) | Same as RECH-X, ensures positivity |
| v (RNN input weights) | N(0, 0.1) | Small weights — same as RECH-X |
| w_h, b | N(0, 0.1) | Same |
| v_z (covariate coefs) | N(0, 0.5) | Larger variance allows significant covariates |

### 9.2 Likelihood-annealing SMC for in-sample fitting

Sequentially samples from
```
πₖ(θ) ∝ p(θ) · p(y|θ)^aₖ,    0 = a₁ < a₂ < ... < aK = 1
```

Implementation steps:
1. Draw N = 5,000 particles from prior p(θ).
2. At each annealing step k, reweight by L(θ)^(aₖ − aₖ₋₁) where L is the marginal likelihood from the particle filter on hₜ.
3. Resample (systematic resampling) when ESS drops below 0.5·N.
4. Move via MCMC kernel (random-walk Metropolis on the tempered posterior) to combat particle impoverishment.
5. At aK = 1, final particles are samples from p(θ|y).

The annealing schedule is **adaptive**: choose aₖ so the conditional ESS = 0.5·N.

By-product: **log marginal likelihood** estimate (for model comparison via Bayes factors).

### 9.3 Particle filter for latent hₜ

Bootstrap particle filter (Doucet, de Freitas & Gordon 2001) with M = 200 latent-state particles per parameter particle. At each t:
1. Propagate hₜ from hₜ₋₁ using the SV transition equation (given current ωₜ from the RNN).
2. Compute importance weight w_t ∝ p(rₜ | hₜ, ν).
3. Normalize, resample.
4. Aggregate to get p(rₜ | y₁:ₜ₋₁, θ) — feeds the likelihood for SMC over θ.

### 9.4 Data-annealing SMC for expanding-window OOS forecasts

After the first OOS observation arrives, update posterior to
```
πₜ(θ) ∝ p(θ) · p(y₁:T+t | θ),     t = 1, 2, ...
```

This avoids re-running the full likelihood-annealing for every new observation; reweight existing particles by the new observation's likelihood and rejuvenate via MCMC only when ESS drops.

### 9.5 Optional faster alternative — Variational Bayes

For prototyping or sensitivity, replace SMC with **mean-field VB** (factored Gaussian over θ). VB underestimates posterior variance and biases the marginal likelihood, so do NOT use it for the headline results — only for development iteration speed.

### 9.6 Implementation stack

- **MATLAB (primary)**: `VBayesLab/Stochastic-Volatility` toolbox (which already implements LSTM-SV, the closest published precedent) + custom SV-LT-RECH classes. `VBayesLab/RealRECH` provides a working LSTM-in-volatility SMC pipeline that we adapt by swapping the GARCH transition for an SV transition. Matches Paper A and Paper B's implementation language.
- **R alternative for SV baselines**: `stochvol` v3.2.4+ (Kastner & Hosszejni) provides `svsample`, `svtsample`, `svlsample`, `svtlsample` — directly fits SV, SV-t, SV-l, and **SV-l-t** (our exact §7.5(c) baseline) via ASIS-MCMC. Use for sanity-checking baseline log marginal likelihoods.
- **Python alternative**: `numpyro` / `pyro` (NUTS + autodiff) for HMC sampling on the SV part, with the RNN unrolled as a `numpyro` `scan` primitive. Hand-rolled SMC in JAX is feasible but only worth it once the model is locked.

### 9.7 Concrete SMC hyperparameters (recommended defaults)

| Setting | Value | Rationale |
|---------|------|-----------|
| Parameter particles N | 5,000 | Same as RECH-X / RealRECH; gives stable ESS at every anneal step with |θ| ≈ 13–28. |
| Latent-state particles M (bootstrap PF) | 200 per parameter particle | Matches Liu et al. 2025; enough for ν > 4 obs likelihoods. |
| ESS threshold for resampling | 0.5·N = 2,500 | Standard SMC heuristic (Doucet et al. 2001). |
| Annealing schedule | Adaptive — pick aₖ so cond. ESS = N/2 | Self-tunes to the geometry of the tempered posterior. |
| MCMC rejuvenation moves per anneal step | 10 random-walk Metropolis sweeps | Same as Nguyen et al. 2022; balance mixing vs cost. |
| Proposal scale (RWM) | 0.5·√diag(empirical cov of particles) | Adaptive; recompute after each anneal step. |
| Random seeds | Log 10 independent seeds per market×model | Report Monte Carlo SE of log marginal likelihood (as Paper A Tables 1, 4). |
| OOS data-annealing | Reweight + 5 MCMC moves when ESS < 0.5·N; full re-run every 100 new obs | Computational sanity. |
| Convergence diagnostic | Compare log marg. lik. across 10 runs; require MCSE < 0.5 nats | Tables 1, 4, 6 of Paper A report MCSEs ≈ 0.1–0.4 nats. |
| Compute budget | ~6 h per market on a single 16-core CPU for SRN; ~12 h for LSTM | Empirically observed in RECH-X / RealRECH code; parallelizes across particles and markets. |

---

## 10. Forecast Evaluation

The reference papers use **five predictive scores**. We keep all five and add three formal model-comparison tests.

### 10.1 Point and density scores (same as RECH-X)

Let T_test be the number of OOS observations.

**Partial Predictive Score (PPS)** — log predictive density:
```
PPS = −(1 / T_test) · Σ log p(rₜ | r₁:ₜ₋₁)
```
Lower is better. The principal **density** forecast metric.

**Quantile Score (QS, Taylor 2019)** — at α-VaR level:
```
QS = (1 / T_test) · Σ (α − 1{rₜ ≤ qₜ,α}) · (rₜ − qₜ,α)
```
Lower is better. Direct Value-at-Risk score; report at α = 1% and α = 5%.

**Mean Squared Forecast Error (MSE)**:
```
MSE = (1 / T_test) · Σ (√RVₜ − v̂ₜ)²
```

**Mean Absolute Error (MAE)**:
```
MAE = (1 / T_test) · Σ |√RVₜ − v̂ₜ|
```

**R² LOG** (Hansen-Lunde 2005):
```
R²LOG = (1 / T_test) · Σ [ log(RVₜ / v̂ₜ²) ]²
```

with v̂ₜ² = ν · σ̂ₜ² / (ν − 2) when innovations are Student-t.

When **RV** is unavailable (e.g. VN-Index emerging market), proxy realized volatility by **squared returns** rₜ² and acknowledge this in the limitations.

### 10.2 Model-comparison tests

- **Diebold-Mariano test** for pairwise loss equality (under QLIKE and MSE loss). Two-sided, HAC-robust, Newey-West variance estimator with bandwidth ⌊T^(1/3)⌋.
- **Model Confidence Set (Hansen, Lunde, Nason 2011)** to identify the set of statistically indistinguishable best models at the 75% confidence level.
- **Log marginal likelihood** comparison (Bayes factors) — by-product of SMC.

### 10.3 Loss function choice

Use **QLIKE** (quasi-likelihood loss) as the primary loss:
```
QLIKE(σ̂², σ²) = log σ̂² + σ² / σ̂²
```

QLIKE is **robust** to noise in the volatility proxy (Patton 2011 *J. Econometrics*) and is asymmetric — preferring under-prediction to over-prediction, which mirrors the asymmetric risk preferences of risk managers.

### 10.4 Diagnostic checks on residuals from each model

Standardized residuals from each fitted model:
```
ε̂ₜ = rₜ / σ̂ₜ
```

Apply Φ⁻¹(F_ν(ε̂ₜ)) under Student-t to get normal-residuals; expect:
- Mean ≈ 0, std ≈ 1.
- Skewness ≈ 0, kurtosis ≈ 3.
- Pass Ljung-Box on standardized squared residuals (no leftover ARCH).
- QQ-plot against N(0,1) approximately on the diagonal.

The reference papers report these in their Tables 2, etc.

---

## 11. Empirical Plan

### 11.1 Stage 1 — Simulation study

Generate 30 synthetic series of length 2,000 from the SV-LT-RECH data-generating process with known parameter vector θ. Estimate via SMC and report:

- Parameter MAE between θ̂ and θ_true.
- Coverage of 95% credible intervals.
- Recovery of the RNN coefficients v_z when zₜ has zero / nonzero true effect.

This validates the estimator (parameter identifiability + SMC convergence) before applying to real data. Mirrors Paper B §3.4.2.4.

### 11.2 Stage 2 — Application 1: Ten major stock markets with RV

Use the Oxford-Man Institute dataset (same as Paper A §4.1 and Paper B §3.4.1.3). Series: SPX, N225, DJI, DAX, FTSE, AORD, CAC40, BVSP, AEX, BFX. ~2,000 obs each, 1,500 train / 500 test.

Compare the proposed **RSV-LT-RECH** (since RV is available) against:
- GARCH(1,1)-t
- GJR-GARCH-t
- RealGARCH-t
- SV-t
- SV-LT
- RECH (original, GARCH backbone)
- RECH-X (Paper A model)
- RSV-LT-RECH (proposed)

Hypothesis: RSV-LT-RECH wins on PPS, QS, MSE, MAE, R²LOG for the majority of markets, and is in the Model Confidence Set everywhere.

### 11.3 Stage 3 — Application 2: Five markets with financial-economic covariates

Use the investing.com dataset from Paper A §4.2 and Paper B: Vietnam (VN100), Japan (N225), France (CAC40), Australia (ASX), Brazil (BVSP). ~2,000 obs, 1,000 train / 1,000 test. Covers Covid-19. Covariates: USD exchange rate, VIX, gold, WTI oil.

Compare the proposed **SV-LT-RECH** against:
- GARCH-t
- GARCH-X-t
- RECH-X
- SV-LT
- SV-LT-RECH (proposed)

Hypothesis: SV-LT-RECH dominates RECH-X because the SV backbone captures vol-of-vol that the GARCH backbone of RECH-X cannot.

### 11.4 Stage 4 — Application 3: Vietnamese stock market

Direct application to **VN-Index and HNX-Index** with the four covariates from Paper B §3.5.3 (crude oil, USD/VND, Bitcoin, gold). 2,262 obs, 1,500 train / 762 test. Covers Jan 2016 – Jan 2025. Confirmed descriptive statistics from Paper B Table 4.14: VN-Index mean 0.0149, std 0.498, skew −0.987, kurt **7.73**; HNX-Index mean 0.0197, std 0.557, skew −0.985, kurt **7.91** — strongly heavy-tailed, justifying Student-t innovations.

Headline empirical contribution: a **rigorous SV-DL-hybrid framework for an emerging market** with both domestic-macro (USD/VND) and global-financial (VIX, gold) drivers — answers Paper B's Research Question 3.

### 11.5 Stage 5 (optional, ambitious) — **SV-LT-MIDAS-RECH-LASSO**

Mirroring Paper B §3.4.2's second contribution but in the SV setting. The latent log-variance state is decomposed into short-term and long-term components:

```
hₜ = τₘ + h_ST,ₜ
h_ST,ₜ = μ + φ (h_ST,ₜ₋₁ − μ) + ωₜ + σ_η ηₜ
τₘ = m + θᵀ Φ(W; ω_w) · Xₘ                      (Beta MIDAS weighting on monthly covariates)
```

where `m` is the unconditional log-variance, `Xₘ` is the vector of low-frequency macro covariates (industrial production, unemployment, CPI, M2, GDP), Φ(W; ω_w) is the Beta-polynomial weighting kernel of Ghysels et al. (2006) with `W` lags, and `θ` are the MIDAS slopes — penalized by LASSO with the soft-thresholding operator. Estimation: SMC with the proximal-gradient LASSO step inside the MCMC rejuvenation kernel (proximal-MCMC). Direct precedent: Bildirici & Ersin (2023) for the LSTM-augmented GARCH-MIDAS form.

This is filed as **Stage 5** rather than core because (a) it extends scope and (b) it requires careful identification work to ensure the MIDAS slopes are separately identifiable from the SV `μ`. It can be deferred to a follow-up paper.

---

## 12. Reporting Tables (deliverable structure)

| Table | Content |
|-------|---------|
| T1 | Descriptive statistics + diagnostics (ADF, KPSS, JB, ARCH-LM, BDS) per series |
| T2 | Posterior means and std deviations of all parameters for each model |
| T3 | Log marginal likelihoods across models |
| T4 | Five predictive scores (PPS, QS, MSE, MAE, R²LOG) per model per market |
| T5 | Diebold-Mariano pairwise test p-values |
| T6 | Model Confidence Set membership |
| T7 | Residual diagnostics (mean, std, skew, kurt, LB-Q²) |
| F1 | 95% one-step-ahead forecast intervals overlaid on OOS returns |
| F2 | QQ-plots of standardized residuals |
| F3 | Posterior densities for v_z (covariate coefficients) |
| F4 | Time series of the recurrent state ωₜ — interpretability plot |

---

## 13. Reproducibility & Software

- All code in a single GitHub-ready repo: `sv-lt-rech/`
- Languages: MATLAB (primary, matches reference papers) + Python (validation).
- Required packages: `arch`, `statsmodels`, `numpyro` / `pyro`, `numpy`, `pandas`, `scipy`. MATLAB toolbox: `Econometrics`, `Statistics and ML`, plus `VBayesLab/Stochastic-Volatility`.
- Data: stored as Parquet; data-loading utilities under `src/data/`.
- Configuration: each experiment driven by a single YAML / JSON config file (market, dates, model class, priors, SMC settings, evaluation window).
- Random seeds fixed and logged.
- Run scripts: `run_simulation.m`, `run_application_1.m`, `run_application_2.m`, `run_application_3.m`.
- Output: each script writes `results/<exp>/<model>/posterior.npz`, `forecasts.csv`, `scores.csv`, `figures/*.png`.

---

## 14. Risk Register

| Risk | Mitigation |
|------|-----------|
| SMC mixes poorly when RNN dimension grows | Start with scalar sₜ (hidden size 1); only increase after diagnostic confirms identifiability. |
| Posterior multimodality (RNN sign ambiguity) | Use weakly informative N(0, 0.1) priors on RNN weights; check label-switching diagnostics. |
| Vol proxy bias when RV unavailable | Use squared returns as proxy and **bold-flag** in tables; report QLIKE which is proxy-robust. |
| Look-ahead leakage via covariate standardization | Standardize using IN-SAMPLE statistics only; document in code review checklist. |
| Computational cost (SMC on 8+ markets × multiple models) | Parallelize across markets via batch scripts; SMC is embarrassingly parallel across particles. |
| Reviewer pushback on "yet another LSTM+GARCH" | The novelty is the **SV backbone with NN-augmented latent state + leverage + Student-t + exogenous covariates** — none of the cited papers do all five simultaneously. |

---

## 15. Sources & Key References

### Reference papers (read in full)
- Nguyen, Nguyen & Tran (2024) — *Deep learning enhanced volatility modeling with covariates* (RECH-X). *Finance Research Letters*. [DOI: 10.1016/j.frl.2024.106145](https://doi.org/10.1016/j.frl.2024.106145).
- Building Statistical Machine Learning Models for Volatility Forecasting (NEU Hanoi thesis, 2023).

### SV + DL prior art (read in summary, updated 2026-05-16)
- Nguyen, T.-N., Tran, M.-N., Gunawan, D. & Kohn, R. (2023). *A Statistical Recurrent Stochastic Volatility Model for Stock Markets* (SR-SV). *Journal of Business & Economic Statistics* 41(2): 414–428. [arXiv 1906.02884](https://arxiv.org/abs/1906.02884). LSTM inside latent log-variance, PMMH/SMC inference — direct existence proof for our LSTM variant.
- Nguyen, T.-N., Tran, M.-N. & Kohn, R. — *A long short-term memory stochastic volatility model* (LSTM-SV working paper, VBayesLab MATLAB toolbox at github.com/VBayesLab/Stochastic-Volatility).
- **Liu, C., Wang, C., Tran, M.-N. & Kohn, R. (2025)**. *A long short-term memory enhanced realized conditional heteroskedasticity model* (RealRECH). *Economic Modelling* 142, 106922. DOI: 10.1016/j.econmod.2024.106922. [arXiv 2302.08002](https://arxiv.org/abs/2302.08002). RealGARCH backbone + LSTM + measurement equation + SMC; 31 indices including COVID. Code: github.com/VBayesLab/RealRECH.
- Zhang, G., Zhao, H. & Fan, R. (2024). *Predicting the volatility of Chinese stock indices based on realized recurrent conditional heteroskedasticity*. *PLOS ONE* 19(10), e0308967. SSE50/CSI300/CSI500/CSI1000 application of RealRECH.
- **Perekhodko, A. & Ślepaczuk, R. (2025)**. *Stochastic Volatility Modelling with LSTM Networks: A Hybrid Approach for S&P 500 Index Volatility Forecasting*. [arXiv 2512.12250](https://arxiv.org/abs/2512.12250), submitted 13 Dec 2025. Most recent direct precedent — but their hybrid is a feature-stack (SV → LSTM input), not joint Bayesian. Code: github.com/aperekhodko/sv_lstm_hybrid_model.
- Luo, R., Zhang, W., Pan, X. & Zhu, S.-C. (2018). *A Neural Stochastic Volatility Model*. AAAI-18 — variational deep state-space SV.
- **Tian, B., Yan, T. & Yin, H. (2025)**. *Forecasting the Volatility of CSI 300 Index with a Hybrid Model of LSTM and Multiple GARCH Models*. *Computational Economics* 66: 1969–1999. Compares LSTM with symmetric + 3 asymmetric GARCH for CSI 300; "GEJT-LSTM-1" winner.
- **Anonymous (2025)**. *Unified GARCH-Recurrent Neural Networks in Financial Volatility Forecasting*. [arXiv 2504.09380](https://arxiv.org/abs/2504.09380), April 2025. Embeds GARCH(1,1) dynamics directly into GRU/LSTM cells; closest competitor to RECH-X.
- Bildirici, M. & Ersin, Ö. (2023). *Financial Volatility Modeling with the GARCH-MIDAS-LSTM Approach*. *Mathematics* 11(8), 1785. Direct precedent for the MIDAS extension of our model (§5.4 / §11.5).

### Foundational SV with leverage and Student-t
- Yu, J. (2005). *On leverage in a stochastic volatility model*. *Journal of Econometrics* 127(2): 165–178.
- Omori, Y., Chib, S., Shephard, N. & Nakajima, J. (2007). *Stochastic volatility with leverage: Fast and efficient likelihood inference*. *Journal of Econometrics* 140(2): 425–449.
- Nakajima, J. & Omori, Y. (2012). *Stochastic volatility model with leverage and asymmetrically heavy-tailed error using GH skew Student's t-distribution*. *Computational Statistics & Data Analysis* 56(11): 3690–3704.
- Hosszejni, D. & Kastner, G. (2021). *Modeling univariate and multivariate stochastic volatility in R with* stochvol *and* factorstochvol. *Journal of Statistical Software* 100(12). Implements SV, SV-t, SV-l, **SV-tl** via efficient MCMC; CRAN package `stochvol` v3.2.4+.
- Mao, X., Czellar, V., Ruiz, E. & Veiga, H. (2020). *Asymmetric stochastic volatility models: Properties and particle filter-based simulated maximum likelihood estimation*. *Econometrics and Statistics* 13: 84–105.
- Hosszejni, D. & Kastner, G. (2019). *Approaches Toward the Bayesian Estimation of the Stochastic Volatility Model with Leverage*. arXiv 1901.11491 — ASIS sampler for SV-l-t.

### Foundational econometrics
- Engle (1982) — *ARCH and the LM test*, *Econometrica*.
- Bollerslev (1986) — *Generalized ARCH*, *J. Econometrics*.
- Taylor (1982, 1994) — original SV model.
- Kim, Shephard & Chib (1998) — *Stochastic Volatility: Likelihood Inference*, *RES*.
- Hansen, Huang & Shek (2012) — *RealGARCH*, *J. Applied Econometrics*.
- Engle, Ghysels & Sohn (2013) — *GARCH-MIDAS*, *Review of Economics and Statistics*.

### Methodology / inference
- Doucet, de Freitas & Gordon (2001) — *Sequential Monte Carlo Methods in Practice*.
- Kastner & Frühwirth-Schnatter (2014) — ASIS for SV, *CSDA*.
- Kingma & Ba (2014) — *Adam* optimizer.
- Patton (2011) — *Volatility Forecast Comparison Using Imperfect Volatility Proxies*, *J. Econometrics*.
- Hansen, Lunde & Nason (2011) — *Model Confidence Set*, *Econometrica*.

### Reference textbook
- Tsay (2010) — *Analysis of Financial Time Series* (3rd ed.), Wiley.

---

## 16. Next Steps (action plan)

1. **Week 1** — Pull data (Oxford-Man + investing.com + Macrobond) and run §1–§5 preprocessing for all 12 markets. Produce Table 1 (descriptives + diagnostics).
2. **Week 2** — Implement baseline GARCH-t and SV-LT in MATLAB (via `VBayesLab` and `Econometrics Toolbox`). Reproduce RECH-X paper numbers on the SPX series as a calibration check.
3. **Week 3** — Implement the proposed **SV-LT-RECH** with likelihood-annealing SMC. Run §11.1 simulation study (30 replicates).
4. **Week 4** — Run Applications 1, 2, 3 (§11.2–§11.4). Generate Tables T2–T7 and Figures F1–F4.
5. **Week 5** — Robustness checks: GRU variant, jumps, alternative priors, alternative covariate sets.
6. **Week 6** — Write up paper draft. Target journal: *Finance Research Letters* or *Journal of Forecasting*.

---

*End of methodology document.*
