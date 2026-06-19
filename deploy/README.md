# Deploy the Tradite autopilot on DigitalOcean (paper mode)

Autonomous, **paper-first** trading worker. Cadence: monitor/auto-exit every 30 min during NSE
hours, decide once daily post-close (15:45 IST), write an EOD report. No secrets needed for paper.

## What runs
`python -m sentiment.scheduler run` — a single long-lived process (the container CMD). It:
- value-ranks the universe daily → sells downgraded holds, buys the best worth-buying names
  (each routed through the hard risk gate: ≤15%/name, ≤30%/sector, stop required, R:R ≥1.5);
- every 30 min in market hours, marks live prices and auto-exits at stop (loss) or target (profit);
- writes `reports/paper_<date>.md` + appends `reports/ledger.jsonl`.

State (the paper book) lives in `data_cache/paper_book.json` — **mount a volume** so it survives
restarts.

## Option A — DigitalOcean Droplet + Docker (recommended; persistent, ~$6/mo)

```bash
# on the droplet (Ubuntu + Docker installed)
git clone <your tradite repo>            # or scp the ml_lab/ folder up
cd tradite/ml_lab
cp deploy/.env.example .env              # edit if you want; paper needs no secrets

docker build -f deploy/Dockerfile -t tradite-autopilot .
docker volume create tradite_data
docker volume create tradite_reports

docker run -d --name tradite --restart unless-stopped \
  --env-file .env \
  -v tradite_data:/app/data_cache \
  -v tradite_reports:/app/reports \
  tradite-autopilot

docker logs -f tradite                   # watch it decide/monitor
```

Container clock is UTC; the scheduler converts to IST internally, so market-hours logic is correct
regardless of droplet timezone.

## Option B — DigitalOcean App Platform (Worker)
Create an App → **Worker** component from your repo, Dockerfile path `deploy/Dockerfile`, run command
`python -m sentiment.scheduler run`. Set the env vars from `.env.example`. ⚠ App Platform's filesystem
is ephemeral — attach a managed DB or persist the book externally, or prefer Option A for a durable book.

## Option C — cron / scheduled jobs (one-shot)
If you'd rather not run a daemon, schedule the one-shot subcommands instead:
```cron
*/30 4-10  * * 1-5  cd /app/ml_lab && python -m sentiment.scheduler monitor   # ~09:15–15:45 IST in UTC
45    10    * * 1-5  cd /app/ml_lab && python -m sentiment.scheduler decide
50    10    * * 1-5  cd /app/ml_lab && python -m sentiment.scheduler report
```
(UTC hours shown; 04:00–10:15 UTC ≈ 09:30–15:45 IST.)

## Verify it's working
```bash
python -m sentiment.scheduler decide      # run one decision now
python -m sentiment.trade_engine status   # NAV, positions, P&L
cat reports/paper_*.md                     # the EOD report
```

## Going live later (only after weeks of good paper results)
Set `KITE_API_KEY` / `KITE_ACCESS_TOKEN` (token refreshes **daily** — the one part that can't be fully
unattended) and `TRADITE_AUTOPILOT_LIVE=I_UNDERSTAND`. Live exits are placed as GTT one-cancels-other
on Zerodha, so they fire even when the container is down. Don't enable this until paper has earned trust.

> Paper fills are assumed at the trigger price (real fills slip). The risk gate caps per-trade loss but
> can't prevent losses. Tooling, not financial advice.
