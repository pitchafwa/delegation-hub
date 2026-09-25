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

## 4. Reliable timer with cron-job.org (GitHub's own scheduler drops and delays runs)
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
4. Turn on failure notifications in cron-job.org so you hear if the timer itself breaks. A successful trigger returns HTTP 204.

## What you will get
* 8:30 ET daily: today's headline moves (IR fixes, adds with the player to drop, timing, today's lineup note).
* 9:30 ET on the first day of each matchup: the whole week's plan, light weeks for your stars, streamers.
* On changes: injury/status changes for your players, new dynasty free agents who would crack your top 5, and a lineup check in the 90 minutes before the first tip of the day (only when something is wrong).
* Nothing is sent before Oct 19 (`ACTIVE_FROM` in `ingest/research/alerts.py`) except the test message. No status or free-agent alerts from 11pm to 8am ET.
* Alerts read the newest plan (refreshed overnight and about 3pm and 6pm ET), so a lineup alert can be a few hours behind the plan; status changes are read live from ESPN and the official injury report.
