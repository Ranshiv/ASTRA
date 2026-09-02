# ASTRA download site

A single static page that explains ASTRA and links to the Windows installer.
Independent of the Tauri app's own build — no Node dependencies, no build
step, no shared routing with the root `index.html`.

## Files

- `index.html` — the entire page (markup, CSS and JS inline).
- `astra_icon.svg` — copy of `public/astra_icon.svg`, used as favicon and hero
  mark. Re-copy it here if the app's icon changes.
- `vercel.json` — static hosting config: no build command, security headers.

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

Just open `index.html` in a browser, or serve it:

```bash
npx serve site
```
