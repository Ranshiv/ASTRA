"""Domain-general cross-instrument corroboration (Direction 3 of the
research plan adopted 2026-08-29: "corroboration as a general
multi-instrument anomaly library").

ASTRA's founding thesis, stated in `README.md`: one instrument calling
something odd is a claim; two INDEPENDENT instruments with different
detectors, cadences and systematics agreeing is evidence. Everywhere else
in this codebase that idea is expressed in terms of sky coordinates,
surveys, and light curves (`crossmatch.py`, `scoring.py`). This package
asks whether the idea itself -- not the astronomy -- is what does the work,
by factoring it into a form with no sky coordinates in it at all, and then
demonstrating it on a second, structurally similar but substantively
different domain.

`core.py` is the domain-agnostic algorithm: association via an injected
distance function over an abstract "position" (which need not be a sky
coordinate -- see `gw_adapter.py`, where it is a time), and agreement
scoring shaped like `scoring.ScoreBreakdown` but parameterised by weights
rather than reading a fixed module-level dict.

`astronomy_adapter.py` re-expresses `crossmatch.py`'s own domain in terms
of that core and is checked for agreement against `crossmatch.group_sources`
directly, rather than modifying `crossmatch.py`/`scoring.py` themselves --
see `astronomy_adapter.py`'s docstring for why that is a deliberate,
lower-risk choice than rewriting the modules every existing candidate score
already depends on.

`gw_adapter.py` is the second domain: gravitational-wave detector
auxiliary-channel-style coincidence vetting, on clearly-labelled SYNTHETIC
data (real GWOSC/Gravity Spy ingestion is out of scope here -- see that
module's docstring for the honest reason, the same discipline
`ztf_forced_photometry.py` already uses for its own stated real-data gap).

`eval.py` is where the actual claims get measured: does corroboration
reduce false positives in BOTH domains using the identical core algorithm,
and does that reduction scale with how uncorrelated the instruments'
systematics are.
"""
