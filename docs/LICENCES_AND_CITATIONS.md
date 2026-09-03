# Licences and citations

How to cite ASTRA and the archives it draws on. For the full list of
research papers, data archives, and software libraries the project uses —
with links and file-level provenance — see `docs/REFERENCES.md`.

## Citing ASTRA

See `CITATION.cff` at the repository root (Citation File Format — GitHub
and Zenodo read this automatically).

## Citing acquired data

Every sealed `DatasetManifest` (`research/datasets/manifests/*.json`)
records its own `license` and `citation` fields at acquisition time
(`manifest.py` v2). Cite the specific release used, not the survey in
general — e.g. "ZTF DR24" rather than "ZTF". `research/sources/
bibliography.bib` collects BibTeX entries for the archives in
`research/sources/source_registry.yaml`.

## Archive terms

- ZTF, Gaia, MAST/TESS, SDSS, IRSA-hosted archives: public, cite the
  release.
- TNS: registration required; ASTRA does not ship a shared credential
  (`engine/astra/credentials.py` handles per-user storage). A TNS-sourced
  label is only pulled when a user has configured their own credential.
- VizieR/SIMBAD (CDS): public, but rate-limited — `netclient.py`'s
  `vizier`/generic buckets throttle accordingly (see that module's
  per-provider interval table).

Always resolve the exact license from the archive's own current terms
page before redistributing acquired data beyond this research use; the
`license` field in a dataset manifest is recorded at acquisition time and
is not re-verified automatically on every read.
