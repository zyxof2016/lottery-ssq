(() => {
  const base = new URL(location.href);
  base.pathname = base.pathname.replace(/(?:index\.html|(?:reviews|research|pools)(?:\.html)?\/?$)/, '');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  base.search = ''; base.hash = '';
  const embedded = new URLSearchParams(location.search).get('embed') === '1' && parent !== window;
  if (embedded) document.documentElement.classList.add('is-embedded');
  if (new URLSearchParams(location.search).get('research') === '1') document.documentElement.classList.add('research-only');
  const url = (path = '', params = {}) => {
    const u = new URL(path, base);
    for (const [key, value] of Object.entries(params)) if (value != null && value !== '') u.searchParams.set(key, value);
    return u.href;
  };
  window.SSQApp = {base, embedded, url};
  if (!embedded) return;
  let lastHeight = 0;
  const resize = () => {
    const height = Math.ceil(document.body.getBoundingClientRect().height + 24);
    if (Math.abs(height - lastHeight) > 2) {
      lastHeight = height;
      parent.postMessage({type: 'ssq:resize', height}, location.origin);
    }
  };
  document.addEventListener('DOMContentLoaded', () => {
    new ResizeObserver(resize).observe(document.body);
    resize();
  });
  document.addEventListener('click', event => {
    const a = event.target.closest('a');
    if (!a || a.hasAttribute('download') || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    const u = new URL(a.href);
    if (u.origin !== location.origin || !u.pathname.startsWith(base.pathname)) return;
    if (u.pathname === location.pathname && u.hash) return;
    const path = u.pathname.slice(base.pathname.length).replace(/\.html$/, '').replace(/\/$/, '');
    const tab = ({'': 'generator', 'index': 'generator', reviews: 'reviews', research: 'research', pools: 'research'})[path];
    if (!tab) return;
    event.preventDefault();
    const params = Object.fromEntries(u.searchParams);
    delete params.embed;
    if (path === 'pools') params.view = 'observations';
    parent.postMessage({type: 'ssq:navigate', tab, params}, location.origin);
  });
})();
