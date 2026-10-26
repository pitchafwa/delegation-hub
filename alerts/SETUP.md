# Alerts setup (one time, about 10 minutes)

## 1. ntfy on your iPhone
1. Install the **ntfy** app (App Store, free).
2. Tap **+**, choose "Subscribe to topic", keep the default server (ntfy.sh), and enter the topic name Claude gave you (`fantasyhub-...`). Allow notifications.
   The topic name is the only secret: anyone who knows it can read and send to it, so do not share it.
3. In iOS Settings > Notifications > ntfy, turn on Time Sensitive Notifications if you want lineup alerts to break through Focus.

## 2. GitHub secret
Repo > Settings > Secrets and variables > Actions > New repository secret: name `NTFY_TOPIC`, value the topic name. (`ESPN_S2` and `SWID` are already there.)

## 3. Send yourself a test
Repo > Actions > "Phone alerts (ntfy)" > Run workflow > mode `test`. A notification should arrive within a minute.

## 4. Reliable timer: Cloudflare Worker `delegation-timer` (since 2026-10-26; replaces cron-job.org)
`cloudflare/timer/` wakes every 5 minutes and starts every workflow on time (alerts check/daily/weekly/lineup, the morning league refresh, the plan rebuilds);
the schedule is in `cloudflare/timer/index.js`. Deploy: `npx wrangler deploy` from that folder. Token (once): `npx wrangler secret put GH_TOKEN` (the same
fine-grained token: this repo only, Actions read and write). Status: https://delegation-timer.tommy-valtin.workers.dev  (token_set, last run, today's schedule).

## 4 (old). cron-job.org timer (retired once the Worker is confirmed; kept for reference)
1. Create a free account at cron-job.org.
2. Create a GitHub token: GitHub > Settings > Developer settings > Personal access tokens > Fine-grained tokens > Generate. Repository access: only `delegation-hub`.
   Permissions: Repository permissions > Actions: Read and write. Expiration: 1 year. Copy the token.
3. In cron-job.org create three jobs. For each: Method POST, URL
   `https://api.github.com/repos/pitchafwa/delegation-hub/actions/workflows/alerts.yml/dispatches`
   Headers: `Authorization: Bearer <your token>`, `Accept: application/vnd.github+json`, `X-GitHub-Api-Version: 2022-11-28`, `Content-Type: application/json`
   Time zone: America/New_York (it handles daylight saving).
   | Job | Body | Schedule |
   |---|---|---|
   | check | `{"ref":"main","inputs":{"mode":"check"}}` | every 20 minutes from 8:00 to 23:00, every day |
   | daily | `{"ref":"main","inputs":{"mode":"daily"}}` | 8:30 every day |
   | weekly | `{"ref":"main","inputs":{"mode":"weekly"}}` | 9:30 every day (the script only sends on the first day of a matchup, Monday) |
   | lineup | `{"ref":"main","inputs":{"mode":"lineup"}}` | every 5 minutes from 10:00 to 23:55, every day (added 2026-10-04: the LATE-SCRATCH watch; it exits in seconds unless one of your teams tips within 4 hours) |
   Data refreshes (added 2026-10-26: GitHub was starting the scheduled refreshes 3-7 hours late). Same headers and token; body `{"ref":"main"}`:
   | Job | URL ends in | Schedule (America/New_York) |
   |---|---|---|
   | league refresh | `/actions/workflows/refresh-league.yml/dispatches` | 5:37 every day |
   | plan refresh | `/actions/workflows/refresh-gameday.yml/dispatches` | 0:10, 15:07, 18:07 and 21:07 every day |
   (Backstop without these: from 6am the alerts job starts the league refresh itself if it has not run, and the plan rebuild check restarts skipped plan builds.)
4. Turn on failure notifications in cron-job.org so you hear if the timer itself breaks. A successful trigger returns HTTP 204.

## What you will get
* **Late scratch (lineup job, every 5 min):** if a player in your lineup is ruled out (official NBA report Out/Doubtful, or ESPN OUT/IR/suspended) before his game tips, you get an alert naming the best swap (a bench player, or a slide-over chain through another starter, or a free-agent option if nobody on the bench can fill the slot), and a reminder 35 minutes before his tip if the lineup still has him in. It works per game, so it covers the 10pm games too. Test it any time with `python research/alerts.py lineup --dry-run --force --now 2026-10-21T17:30 --pretend-out "Player Name"`.
* **Plan move (today / tomorrow evening):** the plan wants an add; names the drop, the gain and a backup, and says when a waiver claim must be in. A reminder follows about 2 hours before his game if the add is still open.
* **Lineup check:** in the 90 minutes before the first tip, only when an Out player is in your lineup or a planned starter is benched.
* **Roster fix:** a free IR move the plan wants (move an injured player to IR, activate a healthy one), checked against ESPN so a done move is not repeated.
* **Keeper-level free agent:** a newly available player who would be one of your top 5 keepers.
* **PC refresh needs attention (8:30 ET slot):** only when the daily model refresh on the PC is stale (36h+) or had failed steps; at most once a day.
* **Weekly brief (9:30 ET, first day of a matchup):** the week's plan, last week's review, light weeks ahead.
* Silent (no message, 2026-10-23): roster moves, a planned target taken, and new free agents the plan has not seen each start a plan rebuild; a worthwhile result arrives as a plan-move alert. Injury status changes are logged only (late scratch covers tonight's starters). The daily brief is retired.
* Nothing is sent before Oct 19 (`ACTIVE_FROM` in `ingest/research/alerts.py`) except the test message. No status or free-agent alerts from 11pm to 8am ET.
* Alerts read the newest plan (refreshed overnight and about 3pm and 6pm ET), so a lineup alert can be a few hours behind the plan; status changes are read live from ESPN and the official injury report.
