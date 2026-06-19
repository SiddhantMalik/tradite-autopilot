# Tradite — Portfolio & Risk Management Grounding Master Sheet

**Purpose.** Companion to `MARKET_GROUNDING.md`. That sheet grounds *idea origination* (news → direction/horizon/confidence). This one grounds the **portfolio & risk layer** (PRD §8.3 Risk & Portfolio Optimization, §17.3 the hard clamp, §11 safety controls): how much to hold, diversification, correlation, drawdown control, the cost/tax reality — and how the sentiment LLM should become **portfolio-aware** so its conviction respects the book.

**Architectural truth (PRD).** The sentiment LLM **does not size positions.** It originates ideas; a risk/sizing layer computes size and a hard pre-trade clamp has final say and can only *reduce* exposure, never raise it. So this sheet has two consumers: (a) the sizing/risk layer + operator, and (b) the sentiment LLM's prompt, which now carries live portfolio state to temper conviction/horizon.

**How it is used.**
- The **Cardinal Portfolio Rules** (§1) are injected into the system prompt as `PORTFOLIO_PRINCIPLES` (always-on).
- Live **portfolio state** (drawdown, exposure, concentration, held names) is passed per call and rendered into the prompt by `portfolio_grounding.portfolio_aware_lines()` via `rag.py` — so the LLM sees the book as an explicit constraint.
- `portfolio_grounding.py` is the source of truth for what's injected; this `.md` is the human reference + rationale + sizing methodology for the risk layer.

**⚠️ Point-in-time safety.** The *methodology* here (sizing math, correlation facts, caps) is timeless → leakage-safe. **Portfolio state is live/as-of**: for backtests the caller MUST pass the point-in-time portfolio snapshot at the item's date, never the current book. Tax/margin/cost *levels* are 2025–26 values — re-verify at runtime (they change with budgets).

**Reliability legend.** High = replicated peer-reviewed and/or official rule. Medium = single study / reputable practitioner. Low = rule-of-thumb / folklore. Treat Low as weak priors.

---

## 1. Cardinal Portfolio Rules (always-on — injected as `PORTFOLIO_PRINCIPLES`)

1. **You originate, you do not size.** A separate risk layer + hard clamp decide quantity and can only *reduce* it. Never imply a quantity, "add", "scale up", "buy more", or leverage. Output only direction/horizon/confidence.
2. **Your confidence is not a calibrated probability.** Treat your own "high" as "more textual evidence than usual," not P≈0.8. The risk layer discounts it (a blanket ~0.5× until ≥100 live trades calibrate it). Default modest. *(High — LLMs are systematically overconfident)*
3. **Respect portfolio state.** When the book is in drawdown, near an exposure cap, at max positions, or already concentrated/correlated in this name or sector → **lower conviction, shorten horizon, bias toward reducing not adding.** If a hard limit is hit, prefer `direction=neutral`.
4. **Concentration guides (defaults):** single name ≤ 10–15% of NAV, single sector ≤ 30%, ~5–15 names total. **Correlated names in the same sector act as ONE larger bet** (hidden concentration) — two long private-banks ≈ one double-size position. *(High)*
5. **Diversification fails in stress.** Calm-market correlations of ~0.3 spike to 0.8–0.9 in selloffs, asymmetrically on the downside. Don't assume a multi-name long book is safe in a shock. *(High)*
6. **Cost + tax floor.** Round-trip delivery friction ≈ 0.25–0.3% (STT dominates) and STCG is **20%** (≤12m) vs LTCG **12.5%** (>12m). If the expected *net* move is below ~0.5%, it's not actionable → `neutral`. Sub-~₹25k delivery positions are uneconomic (fixed ₹15.34 DP fee). *(High)*
7. **Tax cliff.** For an EXIT call on a *profitable* position held ~11–12 months, flag the 12-month LTCG cliff (20% → 12.5%) — the tax saving can dominate a small alpha. *(High)*
8. **Sizing philosophy = survive first.** Risk ≤ 1% of equity per trade; use ¼-Kelly at most (full Kelly ⇒ roughly a 50% chance of a ~50% drawdown, and is fatal when edge is mis-estimated). Risk-of-ruin rises *exponentially* with per-trade risk. *(High)*

---

## 2. Position & Bet Sizing (for the risk layer)

| Rule / method | Value | Reliability | Source | Caveat |
|---|---|---|---|---|
| Risk per trade (fixed-fractional) | **0.5–1%** of equity (max 2%); loss = entry→stop distance × shares | High | ThetaEdge; ChartMill | At 1%/trade you survive 69 straight losses to −50%; at 5%, only 14. |
| ATR stop multiple | **1.5–2.5× ATR(14)** for swing entries (3–4× for mean-reversion) | Medium | AlphaExCapital; AvaTrade | ATR is backward-looking; India gap risk can blow through stops. |
| Vol targeting | size ∝ **target_vol / σ_stock**; target portfolio vol **10–15%** ann. | Medium | QuantPedia; Moreira-Muir 2017 | **OOS evidence is MIXED (Cederburg 2020): helps momentum/profitability, not the market factor or most strategies → use as a damper that REDUCES exposure in high vol, not a return booster.** |
| Kelly | full Kelly = optimal only with *known* edge → **don't use raw**; use **¼–½ Kelly** | High (consensus) | Kelly (Wikipedia); Frontiers 2020 | A ±5pp win-rate error swings Kelly ~3×; full Kelly ⇒ ~50% chance of a ~50% drawdown. |
| Confidence → size | dead-zone below ~0.65; map to ≤1× base; never exceed clamp | High | arXiv 2505.16690 / 2601.07852 | **Raw LLM confidence ≠ P(correct)** — recalibrate (isotonic) after ≥100 trades; until then blanket ~0.5×. |
| Meta-labeling (LdP) | secondary model predicts "act?" → bet-size scalar ∈[0,1] | Medium | Hudson&Thames; LdP 2018 | Needs ~300–500 labels/regime or it overfits; "not a silver bullet". |
| Min economic size | **≥ ₹25k/position** (DP ₹15.34 → <7bps); hard floor ~₹10k | High | Zerodha charges | DP is per-scrip-per-sell-day; staggering sells multiplies it. |
| Max position count | **~5–7 concurrent** at 1% risk before correlated drawdown breaches a ~15–20% DD cap | Medium | derived | More names only help if genuinely uncorrelated. |
| Scaling in | all-at-once at paper stage; pyramiding only with a *raised* stop | Medium | LuxAlgo | Most retail "pyramiding" is disguised over-leverage. |

**Sizing pipeline (3 gates, fail-closed):** (1) ATR fixed-fractional core `shares = equity×risk% / (n×ATR)`; (2) confidence/meta-label scalar (≤1×, default 0.5× early); (3) hard clamp `min(15% NAV, gate-2 size)`. The clamp can only shrink. Output of the layer is the *maximum allowed*.

---

## 3. Diversification, Concentration & Correlation (for the risk layer + LLM)

| Rule / effect | Value | Reliability | Source | Caveat |
|---|---|---|---|---|
| Idiosyncratic risk vs N names | bulk gone by **~20–30** names; modern markets (higher idiosyncratic vol) may need **30–50** | High | MDPI review; Evans&Archer 1968; Campbell 2001 | Evans-Archer (1968) said 8–10; threshold is regime-dependent. Moot for a 5–15 name book → rely on caps+stops, not name count. India intra-sector corr weaker in calm markets. |
| 1/N vs optimizers | **Equal-weight beats 14 mean-variance variants OOS** (needs ~3,000 mo data to beat 1/N) | High | DeMiguel-Garlappi-Uppal 2009 | Optimizers "error-maximize"; means are 11× costlier to mis-estimate than variances (Chopra-Ziemba). |
| Single-name cap | **≤10–15%** of NAV | Medium | Britannica; derived | At 15%, a total loss = 15% DD. Concentrated books need *tighter* stops, not looser. |
| Sector cap | **≤30%** of book | Medium | derived | Nifty is ~38% financials — the engine will over-originate financials; the cap is the gate. |
| Correlated-pair rule | pairwise ρ>0.7 (60-day) → treat as **one combined position** ≤1.5× single-name cap | Medium | FAJ 2018; derived | Without return data, use *same-sector count* as the proxy. |
| Stress correlation | calm ρ≈0.3 → **0.8–0.9 in crashes**, downside-asymmetric | High | FAJ "When Diversification Fails"; PMC | In stress, add +0.2 to assumed correlations and re-check caps. |
| Cash buffer | keep **10–20% cash**; don't run 100% invested | Medium | derived | Buffers correlated drawdowns + funds opportunistic adds. |

---

## 4. Risk, Drawdown & the Hard Clamp (§11/§17.3)

**Layered fail-closed clamp** (each layer only reduces): per-trade max loss (≤1%) → per-strategy budget → **daily max loss → kill-switch** (≈ −2% of equity: cancel orders, flatten, lock out for the session, *non-overridable*) → drawdown-responsive scaling → gross/net exposure cap (F&O counted at delta-equivalent notional) → portfolio max-drawdown circuit-breaker (~15–20% → halt new entries).

| Control / effect | Value | Reliability | Source | Caveat |
|---|---|---|---|---|
| Drawdown-recovery asymmetry | −10%→+11%, −25%→+33%, **−50%→+100%**, −75%→+300% | High (identity) | IntellectualFinance | Preserving capital when losses are small is geometrically efficient. |
| Risk of ruin | exponential in per-trade risk: 2%→~7% RoR, **4%→~30%** (same edge) | High | QuantifiedStrategies | Doubling risk ~4×'s ruin, not 2×. Losing streaks cluster in regime shifts. |
| Drawdown-tiered scaling | 5% DD→80% size, 10%→50–75%, 15%→25–50%, **20%+→~0** | Medium | Quantfish | Thresholds are system-specific; the point is *monotonic* reduction. |
| Vol targeting | damper only (reduce in high vol) | Medium | Moreira-Muir 2017; **Cederburg 2020 (OOS mixed/largely negative)** | Helps momentum/profitability OOS, not the market factor; don't treat as a return booster; lagged protection. |
| Stops — momentum/trend | **help** (e.g., Faber: Sharpe 0.41→0.75) | Medium-High | Kaminski&Lo 2013; Faber | Positive serial correlation makes stops add value. |
| Stops — mean-reversion | **hurt** if tight; use wide (3–4× ATR) or **time-based** exits | Medium | QuantifiedStrategies | Branch stop logic by signal type — common, costly error. |
| VaR / CVaR | dashboard only; actual GFC losses ran **5–10× VaR** | High | QuantDecoded | Fat tails on NSE (kurtosis>3); never a capital-adequacy bound. |
| Kill-switch | daily loss ~**2–3%** of equity; technically non-overridable in-session | Medium | CrossTrade; KillSwitch.in | Loss-aversion ⇒ humans override manually; enforce in code. |

---

## 5. India Costs, Taxes, Margins, Liquidity & Turnover

**The tax regime changed (Finance Act 2024, eff. 23 Jul 2024) — old 15%/10% are wrong now.**

| Item | Value (2025–26) | Reliability | Source | Caveat |
|---|---|---|---|---|
| STCG (equity, ≤12m) | **20%** (was 15%) | High | Business Standard; ClearTax | On transfers after 23 Jul 2024. |
| LTCG (equity, >12m) | **12.5%** (was 10%), no indexation | High | ClearTax | + ~4% cess; the 7.5pp wedge drives the 12-month cliff. |
| LTCG annual exemption | **₹1.25 lakh/yr** | High | ClearTax | If total annual gains < this, LTCG ≈ 0. |
| F&O income | **slab business income** (up to 30%); STT not deductible | High | ClearTax; 5paisa | A 30%-slab F&O trader needs a much higher gross hurdle. |
| Round-trip — delivery swing | **~25–30 bps** (STT 0.2% dominates) | High | Zerodha charges | DP fee disproportionate on small trades. |
| Round-trip — futures | **~10–12 bps** of notional | High | Zerodha charges | Tax is slab on profit, not notional. |
| DP fee | **₹15.34/scrip/sell** (fixed) | High | Zerodha charges | Floor under economic trade size (~₹25k). |
| SPAN+exposure — index F&O | **~10–12%** of notional | Medium | Zerodha/NCL | Dynamic; spikes with VIX → peak-margin penalties. |
| SPAN+exposure — stock F&O | **~15–25%** of notional | Medium | Zerodha/NCL | Higher for volatile names. |
| MTF interest | **0.04%/day ≈ 14.6%/yr** | High | Zerodha | 4-week hold ≈ 112 bps carry — beats only high-conviction moves. |
| Liquidity / ADV cap | size ≤ **1–5% of ADV** | Medium | NSE; quant convention | Mid/small-caps are ADV-limited, not capital-limited; impact rises non-linearly >1% ADV. |
| Turnover vs cost (project truth) | edge dies >**~50 bps** round-trip; weekly dies ~**15 bps** | High (internal) | ml_lab xsec | Lower turnover usually wins net; favor monthly+ rebalancing. |

---

## 6. Making the Sentiment LLM Portfolio-Aware

The LLM's prompt now receives a **live portfolio-state block** (PIT in backtest). It does not size — it adjusts conviction/horizon and flags interactions. State → action rules (implemented in `portfolio_grounding.portfolio_aware_lines()`):

| Portfolio state | What the LLM is told / does |
|---|---|
| **Drawdown 5–10%** | "Book in drawdown — lower conviction on new longs, don't add positions, shorten horizon ~50%, bias to reducing." |
| **Drawdown 10–15%** | "Significant drawdown — highest-conviction single-name signals only; no new F&O; reduce, don't add." |
| **Drawdown ≥ ~15–20%** | "Drawdown halt — new entries blocked; emit only exit / stop-adjust reads." → effectively `neutral` on new longs. |
| **Near daily loss limit** (≤ −1.6%) | "Near daily loss limit — no new long ideas today; only exits/stop adjustments." |
| **Candidate already held** | "Already hold X% of this name (cap ~15%) — this is an ADD-ON; headroom Y%. If at/over cap → neutral." |
| **Candidate sector near cap** | "Sector Z at A% of the 30% cap — limited headroom; if at cap, suppress regardless of sentiment." |
| **≥2 same-sector names held** | "Correlated/hidden concentration — treat as one larger bet; down-grade conviction one notch." |
| **Near gross/exposure cap or max positions** | "Near exposure ceiling / max names — a new idea requires trimming an existing position first (capital is zero-sum here)." |
| **Elevated vol regime** | "High-vol regime — prefer shorter horizon; don't read high vol as a buying opportunity unless the signal is very clear." |

Always-on (in `PORTFOLIO_PRINCIPLES`): never imply upsizing/quantity; confidence isn't calibrated probability; cost+tax floor; the 12-month tax cliff on exits. The output contract (`SentimentSignal`) is unchanged — portfolio-awareness modulates `direction`/`horizon`/`confidence` and may add tags like `portfolio_concentration`, `add_on`, `drawdown_derisk`, `ltcg_cliff`.

---

## 7. Hard Caveats — "Do NOT Assume"

1. **Don't trust raw LLM confidence as probability** — it's overconfident; the risk layer applies a blanket ~0.5× until ≥100 trades calibrate it.
2. **Don't use full Kelly or computed Kelly directly** — a small win-rate error makes it ruinous; ¼-Kelly / fixed 0.5–1% risk per trade.
3. **Don't assume mean-variance optimization helps** — with realistic data it underperforms 1/N OOS; use equal-weight / inverse-vol.
4. **Don't assume name count = diversification** — 15 names in 3 correlated sectors ≈ 3 bets. Correlation, not count, is what diversifies.
5. **Don't assume historical correlations hold in a selloff** — they jump toward 1 exactly when you need diversification.
6. **Don't apply momentum-style tight stops to mean-reversion** — it converts noise into realized losses; branch stop logic by signal type.
7. **Don't treat VaR/CVaR as a safety bound** — tails run 5–10× VaR; hard per-trade and exposure caps are the real protection.
8. **Don't use pre-Jul-2024 tax rates** — STCG is 20%, LTCG 12.5%; F&O is slab business income, not capital gains.
9. **Don't size below the economic floor** — sub-₹25k delivery trades are eaten by the fixed DP fee regardless of signal quality.
10. **Don't let the LLM imply sizing or upsizing** — sizing and the clamp are downstream and authoritative; the LLM only tilts conviction.
11. **Don't count F&O at premium/margin for exposure** — use delta-equivalent notional; short gamma is unbounded on gaps.
12. **Don't make the kill-switch human-overridable in-session** — loss-aversion guarantees it gets overridden when it matters.

---

## 8. Sources (consolidated)

**Sizing:** Kelly (Wikipedia); "Optimal Betting Under Parameter Uncertainty" (Academia); Frontiers Applied Math 2020 (fractional Kelly); Hudson & Thames + López de Prado 2018 (meta-labeling); arXiv 2505.16690 / 2601.07852 (LLM confidence miscalibration); QuantPedia/StoffelWealth/DayTrading (vol targeting); AlphaExCapital/AvaTrade (ATR); Zerodha charges (DP/min size).
**Diversification & correlation:** DeMiguel-Garlappi-Uppal 2009 (RFS, 1/N); Evans & Archer 1968; MDPI diversification review; Chopra-Ziemba (MOSEK cookbook); FAJ "When Diversification Fails" 2018; PMC stress-correlation; Nifty/Bank Nifty weights (IPO Central, Enrich Money); NSE cross-correlation (arXiv 0704.2115).
**Risk & drawdown:** Moreira & Muir 2017 (JoF, vol-managed); **Cederburg et al. 2020 (JFE, OOS failure)**; Kaminski & Lo 2013 (stops); Faber 2006; QuantifiedStrategies (risk-of-ruin, mean-reversion stops); Quantfish (DD scaling); QuantDecoded (VaR/CVaR); CrossTrade / KillSwitch.in; SEBI peak-margin (Zerodha Z-Connect, Bajaj Broking).
**India costs/tax/margins:** Zerodha charges (live); Business Standard / ClearTax / Finnovate (STCG 20%, LTCG 12.5%, ₹1.25L exemption); ClearTax / 5paisa / Sahi (F&O slab); Zerodha/NCL (SPAN+exposure); Zerodha MTF; NSE impact-cost & lot-size circulars.

*Per-claim URLs and reliability grades live in the four agent briefs that generated this sheet.*

---

## 9. Verification log

Built by a 4-agent web-grounded research fleet, then a separate adversarial fact-checker re-verified the load-bearing numbers. Outcome: **3 confirmed, 4 refined, 0 wrong.**
- ✅ Confirmed against primary sources: STCG **20%** / LTCG **12.5%** (>12m) / **₹1.25L** exemption, effective 23 Jul 2024 (CBDT); F&O = non-speculative **slab** business income; DeMiguel-Garlappi-Uppal 2009 (1/N beats the optimizers OOS); Zerodha MTF **0.04%/day**.
- 🔧 Refined: Cederburg 2020 on vol-targeting is **mixed**, not a blanket OOS failure (it helps momentum/profitability, not the market factor) — wording softened; full-Kelly drawdown stated **probabilistically** (~50% chance of a ~50% drawdown); diversification threshold widened to **~20–50 names** (modern markets need more than the classic 20–30; moot for a 5–15 name book → rely on caps+stops).

Smoke tests: portfolio-awareness **20/20**, market-grounding regression **29/29**. Re-verify tax/margin/cost *levels* at runtime — they change with budgets.

