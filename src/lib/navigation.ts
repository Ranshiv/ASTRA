import { useEffect, useState } from "react";

/** Hash-based routing shared by App: reads/writes `#/<section>` and stays in
 * sync with browser back/forward, without pulling in a router dependency. */
export function useHashNavigation<Section extends string>(
  validSections: readonly Section[],
  fallback: Section,
) {
  function sectionFromHash(): Section {
    const value = window.location.hash.replace(/^#\/?/, "").split(/[/?]/)[0] as Section;
    return validSections.includes(value) ? value : fallback;
  }

  const [section, setSection] = useState<Section>(() => sectionFromHash());

  function navigate(next: Section, replace = false) {
    setSection(next);
    const url = `#/${next}`;
    if (window.location.hash !== url) {
      if (replace) window.history.replaceState(null, "", url);
      else window.history.pushState(null, "", url);
    }
  }

  useEffect(() => {
    const onHistoryChange = () => setSection(sectionFromHash());
    window.addEventListener("popstate", onHistoryChange);
    window.addEventListener("hashchange", onHistoryChange);
    if (!window.location.hash) navigate(fallback, true);
    return () => {
      window.removeEventListener("popstate", onHistoryChange);
      window.removeEventListener("hashchange", onHistoryChange);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { section, navigate };
}
