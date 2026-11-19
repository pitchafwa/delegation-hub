// Delegation Hub worker: serves the static site (dashboard/) and, under /api/*, the server pieces.
//
// POST /api/ask {text, ctx} -> {ask}
// Ask anything (2026-11-19, Tommy): a language model (Workers AI) only works out WHICH question the hub knows how to answer
// and fills in its settings. The page then answers from the hub's own numbers (trade engine, daily board, player cards).
// The model never produces numbers or advice itself. ctx is sent by the page: today's date, my team, my roster, league teams.
const ASK_LLM = '@cf/meta/llama-3.3-70b-instruct-fp8-fast';
const recent = new Map();   // per visitor, this instance only: a light brake on runaway use

const strArr = { type: 'array', items: { type: 'string' } };
const ASK_SCHEMA = {
  type: 'object',
  properties: {
    intent: { type: 'string', enum: ['trades', 'pickups', 'players', 'team', 'standings', 'unknown'] },
    targets: strArr, positions: strArr, partners: strArr, keep: strArr, offer: strArr,
    no_picks: { type: 'boolean' }, keep_rounds: { type: 'array', items: { type: 'integer' } },
    focus: { type: 'string', enum: ['any', 'now', 'playoffs', 'title_week', 'future'] },
    injured: { type: 'string', enum: ['any', 'avoid', 'only'] },
    partner_kind: { type: 'string', enum: ['any', 'out_of_race', 'contenders'] },
    tier: { type: 'string', enum: ['any', 'star', 'starter', 'role'] },
    max_give: { type: ['integer', 'null'] }, max_get: { type: ['integer', 'null'] },
    days: strArr, count: { type: ['integer', 'null'] },
    players: strArr, team: { type: ['string', 'null'] },
    understood: { type: 'string' },
  },
  required: ['intent', 'targets', 'positions', 'partners', 'keep', 'offer', 'no_picks', 'keep_rounds', 'focus', 'injured', 'partner_kind', 'tier', 'max_give', 'max_get', 'days', 'count', 'players', 'team', 'understood'],
};

function askPrompt(c) {
  return `You are the question router for a private fantasy basketball tool (ESPN points league, dynasty keepers). The owner types a question; you pick which built-in question it is and fill in its settings. You never answer the question yourself and never give numbers.
Today is ${c.today} (${c.weekday}). The next days: ${c.week}. "This weekend" = the Saturday and Sunday in that list; use only dates from that list. The owner's team is ${c.me}. Positions: PG, SG, SF, PF, C (G = PG+SG, F = SF+PF, "big" = PF+C, "wing" = SG+SF).
League teams (abbreviation: name): ${c.teams.join('; ')}.
The owner's roster: ${c.roster.join(', ')}.

Intents:
- trades: the owner wants trade ideas ("upgrade at center", "who can I get for X", "find me a title-week guy without giving up Y").
  targets = specific players he wants to GET (full names). positions = positions he wants to get. partners = ONLY team abbreviations he names specifically (map owner or team names to abbreviations; empty when he names none; "teams out of the race" -> leave empty and say so in understood).
  keep = his own players he will NOT give up. offer = his own players he wants to trade away. no_picks = true only if he won't give ANY draft picks. keep_rounds = rounds of his own draft picks he won't give ("without my 1st" -> [1]).
  focus: now (this week / this month / win now), playoffs (playoff weeks), title_week (the championship week), future (keepers, next season, young), any.
  tier: star ("star", "stud", "top-20 guy"), starter ("solid starter", "good player"), role ("role player", "mid-level", "cheap", "depth"), else any.
  injured: avoid (healthy only), only (injured players, buy low), any. max_give / max_get: package sizes if stated ("2-for-1" -> max_give 2, max_get 1), else null.
- pickups: free agents to add. days = dates as YYYY-MM-DD ("tonight" = today, "this weekend" = the coming Saturday and Sunday, "tomorrow"). positions as above. count = how many he wants, else null.
- players: questions about specific players: how someone is doing, why a projection moved, news, comparing two or three players ("X or Y?"). players = full names.
- team: questions about one league team (what they have, what they need, their roster). team = its abbreviation.
- standings: playoff odds, standings, title odds, records.
- unknown: anything else.
Use full official NBA player names; fix misspellings and nicknames ("SGA" -> Shai Gilgeous-Alexander, "Wemby" -> Victor Wembanyama). Leave every field that the question does not mention empty / null / "any" / false.
understood: one short plain sentence saying what you will look for, starting with a verb ("Find trades for a center ...").`;
}

const json = (o, s = 200) => new Response(JSON.stringify(o), { status: s, headers: { 'content-type': 'application/json', 'cache-control': 'no-store' } });

async function ask(req, env) {
  if (req.method !== 'POST') return json({ error: 'Use POST' }, 405);
  if (!env.AI) return json({ error: 'Ask isn’t available right now' }, 503);
  if (Number(req.headers.get('content-length') || 0) > 20000) return json({ error: 'Too much data' }, 413);
  let body;
  try { body = JSON.parse(await req.text()); } catch { return json({ error: 'Bad data' }, 400); }
  const text = typeof body.text === 'string' ? body.text.trim().slice(0, 400) : '';
  if (text.length < 3) return json({ error: 'Ask a question' }, 400);
  const s = (x, n, len = 80) => (Array.isArray(x) ? x : []).filter(v => typeof v === 'string' && v.trim()).map(v => v.trim().slice(0, len)).slice(0, n);
  const c = body.ctx || {};
  const ctx = {
    today: /^\d{4}-\d\d-\d\d$/.test(c.today) ? c.today : new Date().toISOString().slice(0, 10),
    weekday: typeof c.weekday === 'string' ? c.weekday.slice(0, 12) : '',
    week: (() => { const d0 = new Date(((/^\d{4}-\d\d-\d\d$/.test(c.today) ? c.today : new Date().toISOString().slice(0, 10))) + 'T12:00:00Z'); return Array.from({ length: 8 }, (_, i) => { const d = new Date(d0.getTime() + i * 864e5); return d.toLocaleDateString('en-US', { weekday: 'long', timeZone: 'UTC' }) + ' ' + d.toISOString().slice(0, 10); }).join(', '); })(),
    me: typeof c.me === 'string' ? c.me.slice(0, 8) : '', teams: s(c.teams, 20, 60), roster: s(c.roster, 25, 40),
  };
  const who = req.headers.get('cf-connecting-ip') || 'x', now = Date.now();
  const hits = (recent.get(who) || []).filter(t => now - t < 60000);
  if (hits.length >= 20) return json({ error: 'That’s a lot of questions in a minute. Try again shortly.' }, 429);
  hits.push(now); recent.set(who, hits);
  let a;
  try {
    const r = await env.AI.run(ASK_LLM, {
      messages: [{ role: 'system', content: askPrompt(ctx) }, { role: 'user', content: text }],
      response_format: { type: 'json_schema', json_schema: ASK_SCHEMA }, max_tokens: 600, temperature: 0.1,
    });
    a = typeof r.response === 'string' ? JSON.parse(r.response.slice(r.response.indexOf('{'), r.response.lastIndexOf('}') + 1)) : r.response;
  } catch (e) {
    return json({ error: 'Couldn’t read that question just now. Try again.' }, 502);
  }
  const one = (x, ok, d) => (ok.includes(x) ? x : d);
  const int = x => (Number.isInteger(x) && x > 0 && x < 20 ? x : null);
  const out = {
    intent: one(a.intent, ['trades', 'pickups', 'players', 'team', 'standings', 'unknown'], 'unknown'),
    targets: s(a.targets, 6), positions: s(a.positions, 5, 8).map(p => p.toUpperCase()).filter(p => ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'BIG', 'WING', 'GUARD', 'FORWARD', 'CENTER'].includes(p)), partners: s(a.partners, 11, 8).map(p => p.toUpperCase()),
    keep: s(a.keep, 15), offer: s(a.offer, 4), no_picks: a.no_picks === true,
    keep_rounds: (Array.isArray(a.keep_rounds) ? a.keep_rounds : []).filter(x => Number.isInteger(x) && x > 0 && x < 30).slice(0, 5),
    focus: one(a.focus, ['any', 'now', 'playoffs', 'title_week', 'future'], 'any'), injured: one(a.injured, ['any', 'avoid', 'only'], 'any'),
    partner_kind: one(a.partner_kind, ['any', 'out_of_race', 'contenders'], 'any'),
    tier: one(a.tier, ['any', 'star', 'starter', 'role'], 'any'),
    max_give: int(a.max_give), max_get: int(a.max_get),
    days: s(a.days, 7, 10).filter(d => /^\d{4}-\d\d-\d\d$/.test(d)), count: int(a.count),
    players: s(a.players, 4), team: typeof a.team === 'string' && a.team.trim() ? a.team.trim().toUpperCase().slice(0, 8) : null,
    understood: typeof a.understood === 'string' ? a.understood.trim().slice(0, 300) : '',
  };
  return json({ ask: out });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === '/api/ask') return ask(request, env);
    if (url.pathname.startsWith('/api/')) return json({ error: 'not found' }, 404);
    return env.ASSETS.fetch(request);
  },
};
