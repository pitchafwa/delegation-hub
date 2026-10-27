// Delegation timer (2026-10-26): starts the hub's GitHub workflows on time. GitHub's own scheduler started them 2-7 hours late; this replaces the
// cron-job.org jobs too. Cloudflare wakes the Worker every 5 minutes (cron "*/5 * * * *", UTC); the Worker works out the time in New York and
// dispatches whatever is due. The GitHub token lives in the Worker secret GH_TOKEN (fine-grained, this repo only, Actions read + write).
// GET / returns the schedule and the last run's dispatches (no secrets).

const REPO = "pitchafwa/delegation-hub";

// [workflow file, inputs, rule(h, m) -> due at this New York time?]   m is a multiple of 5
const JOBS = [
  ["alerts.yml", { mode: "check" }, (h, m) => h >= 8 && h <= 23 && m % 20 === 0],          // status changes, plan moves, pickups
  ["alerts.yml", { mode: "daily" }, (h, m) => h === 8 && m === 30],                         // PC-refresh heartbeat
  ["alerts.yml", { mode: "weekly" }, (h, m) => h === 9 && m === 30],                        // weekly brief (the script only sends on a matchup's first day)
  ["alerts.yml", { mode: "lineup" }, (h, m) => h >= 10 || h < 2],                           // late-scratch watch + live matchup, every 5 minutes; until 2am for the DFF watch / lock-in alert
  ["refresh-league.yml", null, (h, m) => h === 5 && m === 35],                              // morning: rosters, schedule, plan, odds, team usage
  ["refresh-gameday.yml", null, (h, m) => (h === 0 && m === 10) || ((h === 15 || h === 18 || h === 21) && m === 5)],   // plan rebuilds
];

function nyTime(date) {
  const parts = Object.fromEntries(new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", hour: "2-digit", minute: "2-digit", hourCycle: "h23" })
    .formatToParts(date).map((p) => [p.type, p.value]));
  const h = parseInt(parts.hour, 10) % 24;
  const m = Math.round(parseInt(parts.minute, 10) / 5) * 5 % 60;      // cron fires on the 5-minute mark; tolerate a few seconds of drift
  return { h, m };
}

async function dispatch(env, wf, inputs) {
  const r = await fetch(`https://api.github.com/repos/${REPO}/actions/workflows/${wf}/dispatches`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.GH_TOKEN}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "Content-Type": "application/json",
      "User-Agent": "delegation-timer",
    },
    body: JSON.stringify(inputs ? { ref: "main", inputs } : { ref: "main" }),
  });
  return { wf, inputs, status: r.status, ok: r.status === 204, error: r.status === 204 ? null : (await r.text()).slice(0, 200) };
}

let LAST = null;   // per-isolate memory only: shows the latest run when you open the Worker's address

export default {
  async scheduled(event, env, ctx) {
    const { h, m } = nyTime(new Date(event.scheduledTime));
    const due = JOBS.filter(([, , rule]) => rule(h, m));
    const results = [];
    for (const [wf, inputs] of due) {
      try { results.push(await dispatch(env, wf, inputs)); } catch (e) { results.push({ wf, inputs, ok: false, error: String(e).slice(0, 200) }); }
    }
    LAST = { at: new Date(event.scheduledTime).toISOString(), ny: `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`, results };
    const bad = results.filter((x) => !x.ok);
    if (bad.length) console.error("dispatch failed", JSON.stringify(bad));
    else console.log("dispatched", JSON.stringify(results.map((x) => `${x.wf}${x.inputs ? ":" + x.inputs.mode : ""}`)));
  },
  async fetch(request, env) {
    const { h, m } = nyTime(new Date());
    const today = [];
    for (let hh = 0; hh < 24; hh++) for (let mm = 0; mm < 60; mm += 5) for (const [wf, inputs, rule] of JOBS) if (rule(hh, mm)) today.push(`${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")} ${wf}${inputs ? " " + inputs.mode : ""}`);
    const body = { now_new_york: `${h}:${String(m).padStart(2, "0")}`, token_set: Boolean(env.GH_TOKEN), last_run: LAST, schedule_count: today.length,
      schedule_sample: today.filter((x) => !x.includes("lineup")) };
    return new Response(JSON.stringify(body, null, 1), { headers: { "content-type": "application/json" } });
  },
};
