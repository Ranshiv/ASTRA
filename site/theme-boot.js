// Theme boot: stamp data-theme before first paint so there is no
// flash-of-wrong-theme. Loaded blocking (no defer) in <head>, before the
// stylesheets, so it always runs ahead of paint. Kept free of any other
// logic — real behaviour lives in astra.js.
(function () {
  try {
    var stored = localStorage.getItem("astra-site-theme");
    if (stored === "light" || stored === "dark") {
      document.documentElement.setAttribute("data-theme", stored);
    }
  } catch (e) {
    /* private mode or storage disabled — OS preference applies via CSS */
  }
})();
