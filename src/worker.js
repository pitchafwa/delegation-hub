// Delegation Hub worker: serves the static site (dashboard/) and, under /api/*, the server pieces.
// /api/ask is added in the Ask phase; until then every /api/* path answers 404.
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname.startsWith('/api/')) {
      return new Response(JSON.stringify({ error: 'not found' }), { status: 404, headers: { 'content-type': 'application/json' } });
    }
    return env.ASSETS.fetch(request);
  },
};
