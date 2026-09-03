# ASTRA download site

A static page that explains ASTRA and links to the Windows installer.
Independent of the Tauri app's own build — no Node dependencies, no build
step, no shared routing with the root `index.html`.

## Files

- `index.html` — markup only.
- `references.html` — the References page (`/references`): every research
  paper, data archive, and software library the project draws on, sourced
  from `docs/REFERENCES.md`. Reuses `astra.js`/CSS as-is; does not load
  `hero-canvas.js` and omits `#nav-cta` (it never becomes visible without
  `#hero-cta-anchor`, which only `index.html` has).
- `base.css` — tokens (copied verbatim from `src/index.css`), fonts, the
  reset, and page chrome (starfield, nav, buttons).
- `sections.css` — hero through the download panel: the main content blocks.
- `components.css` — FAQ accordion, code blocks, citation, divider, footer,
  and the shared responsive overrides.
- `motion.css` — keyframes and reveal/transition classes, plus the
  `prefers-reduced-motion` overrides that give every animated element a
  final, fully-visible resting state.
- `astra.js` — theme toggle, nav scroll state, scroll reveals, count-up
  stats, SVG draw-ons, FAQ accordion, copy buttons, and the GitHub release
  fetch (see below).
- `hero-canvas.js` — the Canvas 2D hero animation (a reticle locking onto a
  star). Self-pauses via `IntersectionObserver` and `document.hidden`;
  renders one static frame under reduced motion.
- `theme-boot.js` — stamps the stored theme before first paint, loaded
  blocking (no `defer`) in `<head>`. Kept in its own file rather than an
  inline `<script>` because CSP blocks inline scripts — see below.
- `fonts/` — self-hosted `woff2` for Roboto (display and body) and Roboto
  Mono (data/code), plus their SIL OFL licences. Latin subset only.
  Vendored because the page's CSP has no `font-src` allowance for
  `fonts.gstatic.com`, so a Google Fonts `<link>` would be blocked.
- `astra_icon.svg` — copy of `public/astra_icon.svg`, used as favicon and
  hero mark. Re-copy it here if the app's icon changes.

CSS is split three ways, and JS four, to keep every file under this
project's 500-line limit — not for any architectural reason. Add new rules
to whichever file already owns that part of the page.
- `vercel.json` — static hosting config: no build command, security headers.

## Why JS and CSS are external files, not inline

`vercel.json`'s CSP sets `script-src 'self'` with no `'unsafe-inline'` and no
nonce. An earlier version of this page put all behaviour in an inline
`<script>` — which is silently blocked by that CSP on the deployed site,
leaving the download button stuck at "Checking for the latest release…"
forever. It looked fine locally because `npx serve site` (see below) doesn't
apply `vercel.json`'s headers, so the bug only showed up in production.

**When testing changes here, always verify against the real CSP** (see
"Local preview"), not just a plain static server — that's exactly how the
bug shipped unnoticed.

## How the download button works

On page load, the site calls the public GitHub API:

```
GET https://api.github.com/repos/Ranshiv/ASTRA/releases/latest
```

- If the latest release has a `.exe` asset, the button links directly to it,
  and the page shows the file name, size, publish date, and (if a
  `*checksum*.sha256` or `*checksum*.json` asset is attached to the same
  release) a `Get-FileHash` verification snippet.
- If there's no release yet, or the release has no `.exe` asset, the button
  falls back to a link to the GitHub releases page and the page nudges toward
  building from source.
- If the API call fails or times out (6s), the same fallback is shown. There
  is no spinner state left hanging.

This means **no edit to this site is needed to go live** — publishing a
release with an `.exe` asset on `Ranshiv/ASTRA` is the only step.

The unauthenticated GitHub API allows 60 requests/hour/IP, which is enough for
a low/medium-traffic landing page. If that ever becomes a bottleneck, switch
to a scheduled build that writes the release info into the HTML instead of
fetching it client-side.

## Analytics

Both pages load two deferred scripts for Vercel's client-side telemetry:

- `/_vercel/insights/script.js` — [Web Analytics](https://vercel.com/docs/analytics): visitor and page-view counts, no cookies.
- `/_vercel/speed-insights/script.js` — [Speed Insights](https://vercel.com/docs/speed-insights): real-user Core Web Vitals.

This site isn't a Next.js/framework project, so both use the plain-HTML
script-tag integration rather than the `@vercel/analytics` /
`@vercel/speed-insights` npm packages and their `<Analytics/>` /
`<SpeedInsights/>` components (those need a build step this site
deliberately doesn't have). Every script and its beacon endpoint
(`/_vercel/insights/view`, `/_vercel/speed-insights/vitals`) is same-origin
under Vercel's routing, so no `vercel.json` CSP change was needed —
`script-src 'self'` and `connect-src 'self'` already permit them.

**Both must also be turned on for this project** in the Vercel dashboard
(Project → Analytics tab / Speed Insights tab → Enable) — the scripts
alone don't activate collection. Per Vercel's own guidance, allow up to 30
seconds and navigate between pages before expecting data to appear, and
check for content blockers if it doesn't.

## Deploying

This is its own Vercel project, deployed independently of the main app.

```bash
cd site
vercel --prod
```

Or via the Vercel dashboard: **New Project** → import this repo → set **Root
Directory** to `site` → framework preset **Other** → no build command, output
directory `.`.

## Local preview

For a quick look, open `index.html` in a browser or serve it with
`npx serve site` — but that does not apply `vercel.json`'s headers, so it
cannot catch a CSP regression.

To preview with the real headers:

```bash
cd site
vercel dev
```

Or check manually: open the deployed page's DevTools console and confirm
there are zero `Content-Security-Policy` violation messages, and that the
download button leaves its "Checking…" state.

## Cache-busting

Every local `<link>`/`<script src>` in `index.html` carries a `?v=YYYY-MM-DD-N`
query string. Vercel's own `Cache-Control: max-age=0, must-revalidate` header
already forces a real reload to fetch fresh content, so this isn't needed for
correctness — but bump it (increment `N`, or use the new date) on any deploy
that changes `base.css`, `sections.css`, `components.css`, `motion.css`,
`theme-boot.js`, `hero-canvas.js`, or `astra.js`. It's cheap insurance against
any caching layer that ignores the header, and it makes it possible to tell
from a page-source view alone which build a given complaint is about.
