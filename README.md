# ASTRA

**Astronomical Survey & Transient Research Analyzer**

A Windows-first desktop research application that finds unusual astronomical
objects by combining observations from independent surveys, decides whether an
anomaly is real or an artifact, explains why it was flagged, and ranks what a
researcher should look at next.

The distinction that motivates the project: one survey calling an object odd is
a claim; two instruments with different detectors, cadences and systematics
agreeing is evidence. The most common cause of a single-survey anomaly is the
survey itself.

Five surveys are enabled by default: **ZTF** (time domain), **Gaia**
(astrometry and static photometry), **TESS** (high-cadence photometry, SPOC
and QLP), **SDSS** (spectroscopy) and **Pan-STARRS** (mean photometry). A
further 20 connectors — including ASAS-SN, NEOWISE, ALeRCE and ANTARES
(credential-free LSST/ZTF alert brokers), Chandra/Swift/XMM/eROSITA (X-ray),
WISE/2MASS/GALEX/Herschel (infrared/UV), and Hubble/JWST/DESI/VLASS/OGLE/
Kepler/DES — are available opt-in once their provider contract is validated
for a given campaign; see `engine/astra/surveys/` for the full list.

---

## Architecture

Three layers, deliberately separated:

```
React + TypeScript          UI, 11 views
        │  Tauri IPC        65 commands
      Rust                  process supervision, filesystem, hardware, security
        │  JSON-lines RPC   65 methods
     Python                 Astropy / Astroquery / Lightkurve / NumPy / SciPy
        │
  PyTorch / CUDA            anomaly models, GPU acceleration
```

The Python engine runs as a sidecar process, not as an embedded interpreter.
In a released build it is a PyInstaller bundle; in development it is the
project virtualenv. `src-tauri/src/engine.rs` resolves which, and its unit
tests cover every packaging layout, because a released app silently falling
back to a developer's virtualenv would look perfectly healthy on the machine
that built it and fail on every other one.

Storage is matched to the data rather than unified: FITS for raw observations,
Parquet for light curves and feature matrices, SQLite for project metadata,
and a size-capped cache for downloads. Existing data is never auto-evicted —
acquisition reports a capacity refusal instead.

---

## Development setup

Requires Windows 11, Node.js LTS, Rust, Python 3.12, Visual Studio Build Tools,
and (for GPU work) an NVIDIA driver and CUDA toolkit.

```bash
npm install
uv pip install -e engine          # add [gpu] for PyTorch and CuPy
npm run app                       # Tauri dev: Vite on 1420 + the Rust shell
```

`npm run probe` runs the engine standalone and prints the protocol version,
data root, selected device and cache bindings — the fastest way to tell whether
a problem is in the engine or in the bridge.

---

## Tests

| Suite | Command | Covers |
|---|---|---|
| Python | `npm run test:py` | 690 tests across the science engine |
| Rust | `npm run test:rs` | 8 tests: interpreter resolution, RPC framing |
| Frontend | `npm run test:ui` | 18 tests: IPC marshalling, view states |
| Types | `npm run typecheck` | `tsc --noEmit` |
| Packaging | `npm run test:desktop-entry` | bundled assets, startup fallback |

Frontend tests mock at the `invoke` boundary rather than at `engine.*`. Tauri
matches command arguments by name, so a wrapper sending `radius_arcsec` where
the command declares `radiusArcsec` does not raise — it arrives as `None` and
the engine quietly uses a default. Mocking one level higher would leave exactly
that class of bug invisible.

Use `npm run app` to verify a change in the real application. There is no
`npm run build` step in the normal loop.

---

## Packaging

```bash
npm run package:engine            # PyInstaller sidecar, CPU-only
npm run package:windows           # NSIS installer + SHA-256 manifest
```

The shipped engine deliberately excludes PyTorch, torchvision and CuPy: they
add roughly 3.5 GB (`torch_cuda.dll` alone is 1 GB) for a capability most
sessions never use. Acquisition, feature extraction, baseline anomaly
detection, cross-survey matching, ranking and export all work normally in an
installed copy; `deep.train` and `deep.sweep` do not, and say so explicitly
rather than raising a bare `ModuleNotFoundError`. To train deep models, run
from a development checkout.

Production mode fails closed until a real AuthentiCode certificate and
timestamp URL are supplied.

---

## Documentation

- `site/` — the download page (deployed separately on Vercel), for users who
  just want the installer rather than a source checkout.
- `docs/ASTRA-project-plan.txt` — the full plan: research questions, pipeline,
  scoring, phases, hardware.
- `docs/DEFERRED.txt` — **the honest status document.** Every gap, limitation
  and blocked item, tagged `[DONE]` / `[PARTIAL]` / `[KNOWN]` / `[GAP]` /
  `[SCOPED]` / `[BLOCKED]`, with the measurements behind each. Read this before
  trusting any number the application produces.
- `docs/RESEARCH-INTEGRATION.md` — event packets, calibration, selection
  diagnostics, source-attribution priors, literature/event association,
  follow-up constraints, TAP/alert polling, and their RPC/UI boundaries.

---

## Scientific caveats worth knowing before using results

These are not disclaimers; they change how output should be read.

- **TESS does not resolve individual stars.** 21-arcsec pixels mean a TESS
  match corroborates a neighbourhood, not an object. Such matches are marked
  `blended` and excluded from `resolved_surveys`.
- **Detector scores are min-max normalised per run.** 1.0 means most anomalous
  *in that batch*, not an absolute level. Rankings from different runs are not
  directly comparable.
- **Composite scores renormalise over available evidence.** A score of 0.70
  computed from 75% of the weight is not the same quantity as 0.70 from all of
  it, which is why `weight_used` is reported alongside every score.
- **Run detection within a survey.** Pooling ZTF and TESS makes the detectors
  separate by instrument rather than by behaviour.
- **Record the cross-match anchor.** Grouping defaults to the largest
  catalogue, but cross-match and pipeline calls can choose an explicit anchor
  survey. Changing it changes the population denominator and therefore the
  selection function being measured.
- **Injection recovery measures the anomalies that were injected.** It does not
  demonstrate sensitivity to phenomena nobody thought to inject, which is the
  actual discovery case.

The project is not currently a git repository, so experiment provenance hashes
every `.py` file under `engine/astra` instead of recording a commit. That makes
drift detectable but not recoverable; `git init` would fix it.
