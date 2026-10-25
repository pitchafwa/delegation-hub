/* Delegation home-screen app: network first, last good copy when offline.
   Every same-site page and data file is fetched fresh when there is signal (so the plan is never stale), and the copy is kept; with no signal, or if the
   network takes longer than 6 seconds, the last copy is shown instead. Live data (raw.githubusercontent.com) and GitHub API calls are never cached. */
const CACHE = "delegation-v1";

self.addEventListener("install", e => { self.skipWaiting(); });
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});

self.addEventListener("fetch", e => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  const sameSite = url.origin === self.location.origin;
  const fonts = url.hostname === "fonts.googleapis.com" || url.hostname === "fonts.gstatic.com";
  if (!sameSite && !fonts) return;                       // live data, GitHub API, ESPN images: always straight from the network
  e.respondWith((async () => {
    const cache = await caches.open(CACHE);
    const key = sameSite ? url.origin + url.pathname : req.url;   // ignore cache-busting query strings
    try {
      const net = await Promise.race([fetch(req), new Promise((_, rej) => setTimeout(() => rej(new Error("slow")), 6000))]);
      if (net && net.ok) cache.put(key, net.clone());
      return net;
    } catch (err) {
      const hit = await cache.match(key);
      if (hit) return hit;
      if (req.mode === "navigate") { const home = await cache.match(url.origin + url.pathname.replace(/[^/]*$/, "") + "index.html"); if (home) return home; }
      throw err;
    }
  })());
});
