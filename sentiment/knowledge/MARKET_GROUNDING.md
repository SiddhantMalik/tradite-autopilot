# Tradite — Market Grounding Master Sheet

**Purpose.** This is the factual knowledge base that grounds Tradite's sentiment LLM (PRD §8.1 / §19.2). The LLM reads ONE news/filing/social item about one stock and emits a structured `SentimentSignal` (sentiment, direction, horizon, event_tags, thesis, confidence). Without grounding it guesses from generic priors. This sheet replaces guesses with **evidence-graded base rates** for how the Indian market (NSE/BSE, Zerodha) actually behaves.

**How it is used.**
- The **Cardinal Rules** (§1) are injected verbatim into the LLM system prompt on every call (always-on grounding).
- The rest of the sheet is the **retrievable master reference**: `market_grounding.py` distills it into a machine-readable index, and `rag.py` injects only the *relevant slices* (matched on detected event tags, channel, and price context) into each call. Edit this sheet → update `market_grounding.py` to match (the `.py` is the source of truth for retrieval; this `.md` is the human reference and rationale).

**⚠️ Point-in-time safety (PRD §19.6).** Everything here is *general, timeless market knowledge* (mechanics + statistical base rates) — never instrument-specific facts about the future. That makes it leakage-safe to inject when scoring historical items. **Time-varying macro levels are deliberately excluded from the scored grounding** (e.g. the current repo rate, current crude price, current FII stance). Only *structural relationships* (rate-cut → banks positive; crude↑ → OMC negative) are used, because those are stable across time. If you ever add live macro values, gate them behind the item's as-of date.

**Reliability legend.**
- **High** — replicated peer-reviewed evidence and/or official exchange/SEBI rule.
- **Medium** — single study, reputable practitioner source, or India-direction confirmed but magnitude uncertain.
- **Low** — folklore, anecdote, or US-only evidence with no India confirmation. Treat as a weak prior; never the basis for high confidence.

---

## 1. Cardinal Rules (always-on grounding — injected into every prompt)

These ten rules + the confidence rubric are the compact core. They encode the most robust, decision-critical facts so the LLM never reasons from assumption.

1. **React to SURPRISE vs. consensus — not the headline's tone or the absolute level.** Drift accrues to *unexpected* news (PEAD is driven by the earnings *surprise*, not the EPS level). If an event was expected, pre-announced, or is being re-reported, its incremental impact ≈ 0 → sentiment near 0, low confidence. *(High)*
2. **Default to weak.** Most single news items produce no durable, tradeable move; in liquid names the alpha is arbitraged in days, in illiquid names it is mostly noise. Start from confidence ≈ 0.30 and raise only on strong evidence. **Hard ceiling 0.85.** *(High)*
3. **Sentiment alpha decays fast.** Liquid large-caps: predictive window ~1–2 trading days → horizon `hours`/`days`. Genuine earnings/estimate surprises drift longer (PEAD ~weeks) and thin mid/small-caps can run a few days–2 weeks. Do **not** assign `weeks` without a structural catalyst (earnings surprise, estimate revision, M&A, buyback, index change). *(High; Heston-Sinha 2017)*
4. **Net out market & sector beta.** A move in line with index × beta is not stock-specific. Macro/flow news (RBI rates, crude, INR, FII/DII, global cues / GIFT Nifty) is **sector-level, not a single-stock signal** — score the sector via the §7 map, not the company.
5. **Costs set a floor.** Round-trip equity delivery on Zerodha ≈ **0.25–0.5%** (statutory ~0.25%: STT 0.2% + exchange/stamp/GST; plus a **fixed ₹15.34 DP fee per scrip on sell** that makes small trades proportionally dearer); intraday ≈ 0.05–0.08%. If the expected net move is below ~0.5% (delivery), output `direction=neutral` / low confidence — there is no edge below the cost floor. *(High)*
6. **Source credibility scales confidence:** exchange filing > company release / earnings call > tier-1 media (Reuters/Bloomberg/ET/BS/Moneycontrol) > unverified media > social/rumor. **Social-only on a small/illiquid name = manipulation risk** (documented Telegram pump-and-dump in India) → very low confidence. *(High)*
7. **Check the price context you are given (`ret_1d/5d/20d`).** If the stock has already moved far in the news direction, much is likely priced in → reversal risk → lower confidence / shorter horizon. News from a quiet base is a cleaner, higher-confidence signal. *(High)*
8. **Mechanical ≠ informational.** Ex-dividend / bonus / split / rights ex-dates drop the price *mechanically* by the adjustment ratio — this is **NOT bearish**. Never emit a short on an ex-date adjustment. *(High — certain mechanics)*
9. **Acquirer ≠ target (M&A).** The takeover *target* rises (often +15–30%, floored by the SEBI open-offer price); the *acquirer* is typically flat-to-negative. Identify which company the item is about. *(High)*
10. **Respect market structure.** Non-F&O scrips have daily price bands (2/5/10/20%) that cap/halt the move; F&O scrips have no *static* band — only a dynamic ~10% operating range that is relaxed intraday, so they can still move far. ASM/GSM/illiquid names behave erratically (100% margins, T2T, weekly-only trading) → lower confidence. Signals near F&O expiry (last Tuesday, NSE) and on event days (RBI policy, Union Budget, US FOMC) are noisier. *(High)*

**Confidence rubric.** Raise toward **0.6–0.8** only when *all* hold: official/credible source **and** genuine novelty or large surprise vs consensus **and** liquid large-cap **and** a single clean driver **and** corroborated by ≥2 independent sources. Lower to **0.1–0.3** for social/rumor, stale or expected news, illiquid/small-cap, mixed/ambiguous signals, or moves that are mostly market/sector beta. Baseline **0.25–0.35**. Never exceed **0.85**. LLM sentiment is verbally overconfident by default — discount it mechanically with this rubric, regardless of how certain the text sounds.

---

## 2. Event Playbook (the core deliverable)

Per event type: default direction, default horizon, the empirical base rate, what raises/lowers confidence, and the trap to avoid. Tags marked **(new)** extend the original 8-tag vocabulary in `llm_client.py` and are now detected by `market_grounding.detect_event_tags()`.

| Event (→ event_tag) | Default direction | Default horizon | Base rate (magnitude / drift) | Reliability | Confidence raisers / lowers | Key caveat |
|---|---|---|---|---|---|---|
| Earnings beat (`earnings_beat`) | long | weeks | +1–3% on day; PEAD drift continues in the surprise direction for weeks. India-confirmed *significance* (NSE 2002–2017); the ~60-day window is the US benchmark. | High | ↑ large SUE, small/mid-cap, beat **and** guidance raise, estimate upgrades follow. ↓ pre-result run-up, one-off/non-operating beat, mega-cap (fast arbitrage). | Needs a real *surprise*. In-line-with-guidance beat → ~0 drift. |
| Earnings miss (`earnings_miss`) | short | weeks | −3–7% on day, **asymmetrically larger than a beat**; downward drift ~60 days. | High | ↑ miss **and** guidance cut, miss in a crowded momentum name, downgrades follow. ↓ one-off/tax miss, "kitchen-sink" clean-up quarter, already beaten down. | Misses are punished harder than beats are rewarded. |
| Guidance raise (`guidance_raise`) | long | days–weeks | +2–5% (US eq.; India commentary is informal). "Beat **and** raise" is the strongest bullish combo. | Medium | ↑ credible mgmt, both revenue **and** margin raised. ↓ India firms rarely give formal numeric guidance — soft commentary = lower confidence. | Map qualitative concall tone cautiously. |
| Guidance cut (`guidance_cut`) | short | days–weeks | −3–8% (often > the raise reaction); persists if it triggers estimate cuts. | Medium | ↑ multiple demand-weakness signals, capex deferral. ↓ macro-driven (already known), already priced. | Downward guidance can dominate even a current-quarter beat. |
| Analyst upgrade / target raise (`rating_change`) | long | days | +3% on day; modest ~+2.4% drift over ~30 days (short-lived). | Medium-High (Womack 1996) | ↑ first/initiation of coverage, high-conviction broker, big target hike. ↓ "upgrade after the run", consensus already bullish. | Upgrade drift is weaker & shorter than downgrade drift. |
| Analyst downgrade / target cut (`rating_change`) | short | weeks–months | −4.7% on day; **−9.1% drift over ~6 months** — stronger and longer than upgrades. | Medium-High (Womack 1996) | ↑ first sell on a consensus-buy name, target slashed >20%. ↓ already down 20%+, mild "buy→hold". | Hard to monetise short in India (short-sale limits). |
| Order / deal win (`deal_win`) | long | days | +2–5% (inferred from India corp-announcement studies). Scales with deal value / market cap. | Low | ↑ deal >5% of revenue, new client/geography, competitive win. ↓ routine repeat order, already guided, execution-risk history. | Recurring/expected orders → muted. |
| M&A — target / open offer (`mna` **new**) | long | days (most realised immediately) | Target +15–30%; in India the SEBI open-offer floor (52-wk VWAP formula) sets a hard price floor → limited post-announcement drift as the arb spread closes. | High | ↑ all-cash, competing bidders. ↓ spread already closed to offer price. | **Acquirer ≠ target** — see next row. |
| M&A — acquirer (`mna` **new**) | neutral→short | days–weeks | Acquirer ≈ flat-to-slightly-negative on day; mild negative drift; worse if stock-funded. | Medium | ↑ cash deal, clear synergies. ↓ stock-funded, large/dilutive, serial acquirer. | Pre-deal run-up then "sell the news" common. |
| Buyback — tender (`buyback` **new**) | long | days–2 weeks | +2.1–2.8% on day; CAR ~5–6% over ±10–20d. India-confirmed. Tender > open-market. | High (India) | ↑ tender vs OMR, offer at >10% premium, undervalued. ↓ open-market (no firm timeline), cash-rich sector with no growth options. | OMR is weaker & slower than tender. |
| Dividend (`dividend_action`) | long (weak) | days | Regular increase +~1.3% on day; **initiation +2.7–3.8%** (signals confidence). | Medium | ↑ initiation or surprise special dividend > consensus. ↓ routine/expected, ex-date imminent. | **Ex-date drop is mechanical, not bearish — suppress shorts.** |
| Bonus issue (`bonus_issue` **new**) | long (weak) | days | +~1.8% on announcement (liquidity/retail signalling; no fundamental change). | Low-Medium | ↑ first-ever bonus, strong retail following. ↓ tiny abnormal return; pure optics. | Ex-date halving is mechanical — never short it. |
| Stock split (`stock_split` **new**) | long (very weak) | days | +~0.8% on announcement; weaker than bonus. | Low-Medium | ↓ no fundamental change; mostly retail enthusiasm. | Ex-date division is mechanical — never short it. |
| Index inclusion (`index_change` **new**) | long → then reverse | announcement→effective date | +2–5% on announcement, pre-effective run-up, then **~4–7% reversal within 60 days** post-effective. | Medium (India) | ↑ stock newly added, low float, large passive demand. ↓ already anticipated, effective date passed. | Time-boxed trade: long on news, exit by effective date. |
| Index exclusion (`index_change` **new**) | short → then recover | days, then 60–240d | −0.9–1.7% on announcement (sharper than inclusion gain); negative ~10 days, then +4–7% recovery over 60–240d. | Medium (India) | ↑ forced passive selling. ↓ contrarian long after the flush. | Permanent volume decline post-exclusion. |
| Block / bulk deal — buy (`block_deal` **new**) | long | 1–3 days | Positive; but pre-deal run-up (~9.6% over 10d in small-caps) shows heavy front-running. | Low-Medium | ↑ marquee institutional buyer at premium. ↓ leak likely already priced. | Front-running contaminates the signal. |
| Promoter / institutional sell (`promoter_sell` **new**) | short | days–weeks | Negative; severity scales with discount to market; large promoter sale at a discount is a strong negative. | Low-Medium | ↑ promoter selling at steep discount, large % of holding. ↓ pre-planned/▪OFS already known. | Distinguish OFS/strategic exit from distress. |
| Promoter pledging (`pledging` **new**) | short (risk modifier) | ongoing | No immediate shock typically, but **raises crash-tail risk** (forced-sale cascade) and correlates with weaker performance. | Medium (India) | Use as a **confidence-reducer on longs**, not a standalone short. | Tail risk, not linear drift. |
| Legal / regulatory / fraud (`legal_regulatory`) | short | weeks–months | −5% to −20%+ on disclosure (can hit lower circuit); negative drift while a probe runs. | Medium | ↑ SEBI interim order with trading curbs, ED/criminal probe, revenue-inflation/fraud allegation. ↓ sector-wide rule (not company-specific), small fine, early-stage show-cause notice. | SEBI actions stage: show-cause < interim order < final order. |

**Cross-checks across the research fleet.** PEAD is the single most robust, India-confirmed effect (use it for the earnings tags). The surprise-vs-level rule, fast decay, and the ex-date trap are corroborated by ≥2 briefs each. Magnitudes for `deal_win`, `bonus_issue`, `stock_split`, `block_deal` rest on thinner evidence (graded Low/Low-Medium) — keep confidence modest.

---

## 3. Sentiment Dynamics & Confidence Calibration

The empirical record shows **both** under-reaction (→ drift) and over-reaction (→ reversal); which one dominates depends on news type, horizon, source, and stock size. The LLM's job is to pick the right regime and size confidence honestly.

| Effect | Tendency / half-life | Drift vs reversal | Horizon | Reliability | Source |
|---|---|---|---|---|---|
| Daily news sentiment | Predicts ~1–2 days (large-cap); weekly-aggregated ~13 weeks | Drift in sentiment direction | 1–2 days | High | Heston & Sinha 2017 (FAJ) |
| Firm-specific negative words | Brief under-reaction, return predictability in first few days | Drift (short) | days | High | Tetlock et al. 2008 (JF) |
| Post-news drift | Drift after **bad** headlines; reversal after **good** headlines (asymmetry) | Mixed by sign | 1–12 mo | High | Chan 2003 (JFE) |
| Stale / re-reported news | Attenuated reaction; **reverses** next week | Reversal | ~1 week | High | Tetlock 2011 (RFS) |
| Attention spike (high volume / extreme move) | Retail net-buys attention stocks → transient pressure | Reversal after spike | days–weeks | High | Barber & Odean 2008 (RFS) |
| Social-media pump (small-cap) | Abnormal social volume → same-day spike → reversal | Reversal | days–1 wk | High | Renault (AMF); Kogan et al. |
| "Buy the rumor, sell the news" | Price can fall on good-but-expected news | Reversal at event | event ± days | Medium | BRSN review |
| Long-run overreaction | Losers beat winners over 3–5 yrs | Reversal | years (irrelevant to swing) | High | De Bondt & Thaler 1985 |

**Operational calibration rubric** (this is the heart of "facts not assumptions"):

- **Baseline** every item at confidence **0.25–0.35** and sentiment magnitude modest. Assume no edge until evidence proves otherwise.
- **Raise to 0.6–0.8** only when *all*: credible/official source **+** genuine novelty or large surprise vs consensus **+** single clean driver (not beta) **+** liquid large-cap (>₹20,000 Cr, high turnover) **+** corroborated by ≥2 independent sources **+** fundamentals-focused content (revenue/earnings/contract/approval, not vague macro).
- **Raise only to 0.4–0.6** when partial: tier-3 media with named sources + one corroboration, or partially-anticipated but materially beats/misses, or liquid mid-cap (₹5,000–20,000 Cr).
- **Lower to 0.1–0.3** for ANY of: social/rumor-only; stale/re-reported; anticipated event at/near consensus; ambiguous or conflicting signals; illiquid/micro-cap (<₹2,000 Cr); predominantly market/sector beta; long speculative narrative chains (each unverified causal link multiplies uncertainty); positive news after a large run-up (attention exhaustion / BRSN reversal risk).
- **Never exceed 0.85.** LLMs are verbally overconfident and shift neutral text toward positive — discount mechanically.

**Channel → credibility mapping** (the `channel` field on each item):
- `filing` → Tier-1 (exchange/SEBI/company disclosure): highest trust, novel, verifiable. But check it is not a *re-report* of an already-public filing.
- `news` → trust by outlet; tier-1 financial media is good, unverified media is medium.
- `social` → lowest trust; treat as manipulation-prone, especially for small/illiquid names. Corroboration with a filing/tier-1 source is required before confidence > 0.3.

---

## 4. Price-Context Interpretation (using `last_close`, `ret_1d/5d/20d`)

The LLM is handed point-in-time trailing returns. Use them to choose between **continuation (PEAD/under-reaction)** and **reversal (priced-in / short-term reversal / exhaustion)** — do **not** treat them as standalone technical signals.

| Context (trailing return vs news direction) | Interpretation | Adjustment |
|---|---|---|
| Flat / mild prior move (`ret_20d` ≈ −5%..+10%) + fresh catalyst | Clean signal from a base | Full confidence; standard horizon (days–weeks) |
| Already +10–20% (`ret_20d`), same catalyst explaining the run | Partly priced in | −15–25% confidence; shorten horizon to days; tag `partial_priced_in` |
| Up >25% (`ret_20d`) or parabolic into the news | Exhaustion / reversal risk (short-term reversal effect) | −25–40% confidence; raise the bar for a long; shorter horizon |
| Down hard (`ret_20d` < −15%) + fresh **bad** news | Drift vs bounce tension | Lean continuation if news is *new* info; lean bounce if it's a *re-report*; reduce confidence either way |
| Quiet, low-vol base + major catalyst (earnings/M&A/regulatory) | Classic clean breakout setup | Highest confidence, longest justified horizon |
| Large same-day gap (`ret_1d`) consistent with the news | Breakaway gap (news-driven gaps mostly do **not** fill) | Confirm direction; extend horizon vs a small/ordinary gap |

**Evidence-graded technicals (what to trust).** Time-series & cross-sectional momentum, the 52-week-high anchoring effect, PEAD, short-term (1-month) reversal, and volatility clustering (GARCH — high vol persists) are **High**-reliability. Moving-average crossovers, RSI, MACD as standalone signals are **Low–Medium** (lagging, weak net of cost). Chart patterns (head-and-shoulders, flags) and "support/resistance = sustained barrier" are **Low** / folklore.

**Do not** let an oscillator (e.g. "RSI>70") override a genuine fundamental event read. RSI/MACD are smoothed transforms of the same `ret_*` you already have — no new information, just noise. Use volatility (ATR/realized vol) only for expected-move sizing, never for direction; expect realized vol to *exceed* trailing ATR on the news event itself.

---

## 5. Market Mechanics, Costs, Circuits & Surveillance (India)

**Sessions (IST).** Pre-open call auction 09:00–09:15 (equilibrium price set ~09:08–09:12 by max-volume matching); continuous 09:15–15:30; closing-price = VWAP of 15:00–15:30. The pre-open already discounts overnight news — a large gap at 09:15 is the market's first answer, **not** fresh alpha. *(High)*

**Settlement.** Mandatory **T+1** for equities; optional **T+0** for the top ~500 stocks. Mid/small-caps remain T+1 — don't assume same-day proceeds. *(High)*

**Round-trip costs (Zerodha).** Equity **delivery ≈ 0.25–0.5%** of value (statutory ~0.25%, higher on small trades due to the fixed DP fee; STT 0.1% each side = 0.2%, exchange txn ~0.006%, stamp 0.015% buy, **DP fee ₹15.34/scrip/sell fixed**, GST on charges, ₹0 brokerage). Equity **intraday ≈ 0.05–0.08%** (STT 0.025% sell-only, brokerage ₹20/order). Futures STT rose to 0.05% sell-side (Oct 2024); options STT 0.15% of premium sell-side. **The DP fixed fee makes small delivery trades proportionally expensive** — a ₹5,000 position pays ~0.3% just in DP. *(High)*

**Circuit bands.**
- *Market-wide* (Nifty/Sensex): 10% / 15% / 20% trigger graduated halts; 20% halts the day. Rare. *(High)*
- *Individual scrip*: daily bands of **2/5/10/20%** (or "No Band") — **F&O stocks have no *static* band**, only a dynamic ~10% operating range relaxed intraday, so they can move far in a session. Before bounding an expected move, check F&O status: cap non-F&O moves at the band; treat F&O moves as effectively uncapped intraday. If a non-F&O scrip has already used ≥80% of its band on the day, remaining headroom is tiny and it may halt mid-signal → lower confidence, tag `circuit_risk`. *(High)*

**Surveillance (reduce confidence / lengthen horizon).** ASM (additional margins up to 100%, no pledging), GSM (trade-for-trade, 5% band, ASD deposits, **Stage III+ trades only once a week**), and T2T (no intraday netting) make a stock erratic or near-untradeable. GSM Stage III+ → effectively `direction=neutral`, confidence <0.15. ASM was extended to F&O stocks in Aug 2024. Status changes without notice — re-check lists at execution. *(High)*

**Disclosure speed.** SEBI LODR Reg 30 requires board outcomes within ~30 min and most material events within 12–24h. Institutions watch the NSE/BSE announcement portals in real time, so by the time a *news article* repeats a filing, smart money has reacted — discount the alpha and prefer a short horizon. *(High)*

---

## 6. Derivatives & Positioning (India F&O)

**Structure (verify — changed in 2024–25).** NSE index/stock F&O expiry moved to **last Tuesday** of the month (BSE: Thursday) from Sep 2025. Weekly options exist **only on Nifty 50 (NSE)** and **Sensex (BSE)** — Bank Nifty/FinNifty/MidcpNifty weeklies and **all single-stock weeklies were discontinued** (Nov 2024). Lot sizes change often (never hardcode). *(High)*

**Horizon → vehicle.** Single stocks have no weekly options → a `hours`/`days` single-stock view maps to cash or futures, not options. Near expiry (≤3 days), a `days` view should roll to the next series (gamma/decay noise). *(High)*

**IV crush.** Stock IV inflates 30–80% in the ~5–10 days before earnings, then collapses 30–60% right after. A *directionally correct* options buy can still lose if the move is smaller than the implied move. So a bullish read ≠ "buy calls" near an event — flag `iv_elevated`; prefer futures/cash. *(Medium)*

**Open Interest four-quadrant read** (changes vs prior session): price↑ OI↑ = long buildup (strong bullish); price↑ OI↓ = short covering (bullish, fading); price↓ OI↑ = short buildup (strong bearish); price↓ OI↓ = long unwinding (bearish, fading). **Interaction with sentiment:** short buildup + bullish news → short-covering squeeze (raise confidence); long buildup + bearish news → fast long-unwind (widen risk). Beware: near expiry, OI falls mechanically from rollover — not a conviction signal. *(Medium)*

**PCR** is a weak, debated contrarian indicator (extremes only: >1.3 mild bullish, <0.7 mild bearish). Never a primary override. *(Low-Medium)*

**F&O ban (MWPL).** A stock crossing 95% of its market-wide position limit enters a ban (only position-reducing trades; exit <80%). A bullish view on a banned stock can't be expressed via new F&O — cash only. Tag `mwpl_ban`. *(High)*

---

## 7. Macro, Sector→Driver Map & Beta (structural — PIT-safe)

**Only structural relationships are used in scoring** (directions are stable across time); current macro *levels* are excluded to avoid leakage/staleness.

**Beta first.** Every stock's move = market/sector beta + idiosyncratic. A move in line with `index × beta` is not news. Bank Nifty ≈ 1.4× Nifty; FMCG/pharma <0.8×. Financials are ~38% of Nifty and Reliance/HDFC Bank/ICICI/Bharti/SBI ~30% — heavyweight news drags the whole index (don't read index-driven moves as company news). *(High direction)*

**Sector → driver → direction cheat-sheet:**

| Macro driver | Positive for | Negative for | Reliability |
|---|---|---|---|
| RBI **rate cut** | Banks, NBFCs, real estate, autos (rate-sensitives) | — (transmission lags 1–2 quarters; banks can dip short-term) | High |
| RBI **rate hike** | — | Banks (NIM squeeze short-term), real estate, autos, NBFCs | High |
| **INR depreciation** (USD/INR↑) | IT services, pharma & specialty-chem exporters | OMCs, airlines, tyres, importers | High |
| **INR appreciation** | Importers | IT/pharma exporters | High |
| **Crude oil ↑** (Brent) | Upstream (ONGC, Oil India) | OMCs, aviation, tyres, paints (India imports ~85% crude) | High dir / Medium mag |
| **Crude oil ↓** | OMCs, aviation, tyres, paints | Upstream E&P | High |
| **FII heavy buying** | Large-caps, index | — (DII SIP inflows now offset FII selling; old "FII sell = crash" rule weakened) | Medium |
| **Global risk-on** (S&P/Nasdaq up, GIFT Nifty up) | Broad market, IT/tech-adjacent | — | High dir |
| **Metals/commodities ↑** (LME/China) | Tata Steel, Hindalco, JSW, Vedanta | Metal *consumers* | High |

**Event-risk windows (noisier signals → lower confidence, wider horizon):** F&O expiry week (last Tue), RBI MPC days, Union Budget (Feb 1) week, US FOMC. GIFT Nifty (replaced SGX Nifty, 2023) is a pre-open beta indicator only — never a single-stock signal. *(High)*

**Caveats:** OMCs depend on government pump-price policy (crude↓ doesn't always help if prices aren't cut); IT INR-sensitivity is muted by 12–18mo hedges; pharma INR gain is offset by Chinese API import costs. *(Medium)*

---

## 8. Hard Caveats — Master "Do NOT Assume" List

1. Don't react to the headline's *tone/level* — react to the **surprise vs consensus**. Expected/pre-announced/re-reported news ≈ no edge.
2. Don't assign `weeks` without a structural catalyst (earnings surprise, estimate revision, M&A, buyback, index change). Most sentiment alpha is gone in 1–5 days.
3. Don't treat an **ex-date** drop (dividend/bonus/split/rights) as bearish — it's mechanical. Never short it.
4. Don't confuse **acquirer and target** in M&A. Target up; acquirer flat/down.
5. Don't trust **social/rumor**, especially on small/illiquid names — India Telegram/finfluencer pump-and-dump is documented; the big initial move is the pump, and it reverses.
6. Don't read a **beta/sector move** as company news. Net out `index × beta`; score macro via the sector map, not the firm.
7. Don't act below the **cost floor** (~0.4–0.5% delivery round-trip). Tiny expected moves → neutral.
8. Don't cap an **F&O stock's** move with price-band logic (no band); don't ignore that **non-F&O** moves can halt at the band.
9. Don't trade **ASM/GSM/T2T/illiquid** names with normal confidence — margins, weekly-only trading, and no netting break the thesis.
10. Don't assume options express a view cleanly near events — **IV crush** can sink a correct direction. Single stocks have no weekly options.
11. Don't let **RSI/MACD/chart patterns** override a fundamental event read — weak/folklore, and redundant with the `ret_*` you already have.
12. Don't extrapolate **long narrative chains** ("approval → revenue → re-rating") — each unverified link multiplies uncertainty and inflates confidence.
13. Don't treat the LLM's own fluent certainty as calibrated probability — **discount with the §3 rubric**.
14. Don't use **stale macro levels** (repo rate, crude price, FII stance, lot sizes, expiry day) — they change; rely on structural relationships and re-verify specifics at runtime.
15. Don't ignore **base rates** — start from "this probably has no durable edge" and update upward only on strong, corroborated, novel, credible-source evidence.

---

## 9. Sources (consolidated, by domain)

**Events & anomalies.** Ball & Brown 1968 / Bernard & Thomas 1989 (PEAD); Harshita et al. 2018 (PEAD India, SCIRP); QuantPedia PEAD review; Womack 1996 (analyst recs, JF); Irvine 2002 (initiations, JFE); Rani et al. 2015 (M&A India, SAGE); India open-offer study (Springer); buyback India (Emerald IRJMS; Business Perspectives); dividend initiation (Bulan/Brandeis); bonus/split India (IJIET 2017); Nifty inclusion/exclusion (ResearchGate CNX Nifty); bulk-deal front-running (Emerald; ScienceDirect); promoter pledging & crash risk (Emerald JAMR 2024); SEBI action impact (Cogent Econ & Finance 2022). NSE corporate-actions adjustment (ex-date mechanics).

**Factors & decay.** Jegadeesh & Titman 1993; Moskowitz-Ooi-Pedersen 2012 (TS momentum, AQR/JFE); George & Hwang 2004 (52-week high, JF); momentum/reversal India (ScienceDirect S0927538X23002640, S0970389617301647); Heston & Sinha 2017 (News vs Sentiment, FAJ); Tetlock 2007/2008/2011 (JF, RFS); Chan 2003 (JFE); De Bondt & Thaler 1985 (JF); Barber & Odean 2008 (RFS); Di Mascio-Lines-Naik 2016 (alpha decay); low-vol India (BacktestIndia, 18-yr NSE); GARCH NSE (PMC).

**Behavioral / manipulation.** Kogan-Moskowitz-Niessner (Fake News); Renault (AMF social-media manipulation); SEBI finfluencer enforcement (Mondaq 2024-25); LLM bias in finance (ACM AI-Finance 2025; arXiv 2602.14233).

**Mechanics / structure (official).** Zerodha charges & Z-Connect (Oct-2024 STT revision; SEBI index-derivatives rules); NSE circuit breakers, price bands, ASM FAQ, GSM, T+0 settlement, pre-open, MWPL; NSE Clearing margins; SEBI LODR Reg 30. Lot-size/expiry changes 2024–26 (Angel One, HDFC Sky, Kotak Neo, ICICI Direct). India VIX / IV (practitioner). GIFT Nifty migration 2023.

**Macro / flows.** RBI MPC; CPI (Trading Economics); USD/INR sector impact (ICICI Direct); crude→sector (HDFC Sky, Upstox, multibagg); FII/DII (Motilal Oswal, Kotak, Wright Research); Nifty weightage (AnalyticsInsight, Smart-Investing); beta (Enrich Money, 5paisa).

*Full URLs with per-claim reliability grades are preserved in the agent research briefs that generated this sheet (see project research log). Every quantitative figure above carries its grade inline in §2–§7; treat Low-grade figures as weak priors only.*

---

## 10. Verification log

This sheet was built by a 7-agent web-grounded research fleet, then a separate adversarial fact-checker re-verified the most decision-critical numbers against primary sources (NSE/SEBI/Zerodha/journals). Outcome: **4 confirmed, 2 adjusted, 1 softened.**
- ✅ Confirmed: NSE F&O expiry = last Tuesday (BSE Thursday), weekly options restricted to Nifty 50 / Sensex, Womack downgrade-drift (−9.1% / 6mo) > upgrade (+2.4%), T+1 settlement (+ optional T+0).
- 🔧 Adjusted: delivery round-trip cost corrected to **~0.25–0.5%** (statutory ~0.25% + fixed DP fee on small trades), not a flat 0.4–0.5%; F&O scrips have **no *static* band but a dynamic ~10% intraday-relaxable range**, not literally "no limit".
- 〰️ Softened: PEAD on NSE is confirmed *significant* but the precise **~60-trading-day** drift window is the US benchmark, not an India-specific measurement.

Re-verify time-sensitive structural facts (expiry day, lot sizes, STT rates, surveillance lists) at runtime — they change.

