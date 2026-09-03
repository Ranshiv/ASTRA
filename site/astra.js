// ASTRA download site — page behaviour
// External file so this actually executes under the site's strict CSP
// (script-src 'self', no 'unsafe-inline'). See site/README.md.
(function () {
  "use strict";

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------------- Theme toggle ---------------- */
  (function themeToggle() {
    var btn = document.getElementById("theme-toggle");
    if (!btn) return;
    var root = document.documentElement;

    function current() {
      var explicit = root.getAttribute("data-theme");
      if (explicit) return explicit;
      return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    }

    function apply(theme) {
      root.setAttribute("data-theme", theme);
      try {
        localStorage.setItem("astra-site-theme", theme);
      } catch (e) {
        /* private mode or storage disabled — theme just won't persist */
      }
      btn.setAttribute("aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
      window.dispatchEvent(new CustomEvent("astra:theme-change", { detail: { theme: theme } }));
    }

    var stored = null;
    try {
      stored = localStorage.getItem("astra-site-theme");
    } catch (e) {
      /* ignore */
    }
    if (stored === "light" || stored === "dark") apply(stored);
    else btn.setAttribute("aria-label", current() === "dark" ? "Switch to light theme" : "Switch to dark theme");

    btn.addEventListener("click", function () {
      apply(current() === "dark" ? "light" : "dark");
    });
  })();

  /* ---------------- Sticky nav shrink + nav download pill ---------------- */
  (function navScroll() {
    var nav = document.getElementById("site-nav");
    var heroCta = document.getElementById("hero-cta-anchor");
    var navCta = document.getElementById("nav-cta");
    if (!nav) return;

    // Plain scroll-position comparison, not IntersectionObserver: Safari's
    // collapsing/expanding URL bar resizes the visual viewport as you
    // scroll, which has known inconsistent interactions with
    // IntersectionObserver across engines — the nav pill was confirmed
    // showing before any scroll had happened at all. offsetTop is measured
    // against the layout viewport, which doesn't have that problem.
    var heroCtaBottom = heroCta ? heroCta.offsetTop + heroCta.offsetHeight : 0;

    function recomputeHeroCtaBottom() {
      if (heroCta) heroCtaBottom = heroCta.offsetTop + heroCta.offsetHeight;
    }

    function onScroll() {
      if (window.scrollY > 24) nav.classList.add("scrolled");
      else nav.classList.remove("scrolled");

      if (navCta && heroCta) {
        navCta.classList.toggle("visible", window.scrollY > heroCtaBottom);
      }
    }
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", recomputeHeroCtaBottom);
  })();

  /* ---------------- Scroll reveals ---------------- */
  (function reveals() {
    var els = document.querySelectorAll(".reveal");
    if (!els.length) return;

    if (reduceMotion) {
      els.forEach(function (el) {
        el.classList.add("in");
      });
      return;
    }

    var io = new IntersectionObserver(
      function (entries, obs) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("in");
            obs.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.15, rootMargin: "0px 0px -40px 0px" }
    );
    els.forEach(function (el) {
      io.observe(el);
    });
  })();

  /* ---------------- SVG draw-on (schematic, pipeline, arch, divider) ---------------- */
  (function drawOns() {
    var paths = document.querySelectorAll(".draw-path");
    paths.forEach(function (p) {
      var len = 0;
      try {
        len = p.getTotalLength();
      } catch (e) {
        len = 300;
      }
      p.style.setProperty("--path-len", len);
    });

    var dividers = document.querySelectorAll(".divider");
    var targets = Array.prototype.slice.call(paths).concat(Array.prototype.slice.call(dividers));
    if (!targets.length) return;

    if (reduceMotion) {
      targets.forEach(function (el) {
        el.classList.add("in");
      });
      return;
    }

    var io = new IntersectionObserver(
      function (entries, obs) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("in");
            obs.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.3 }
    );
    targets.forEach(function (el) {
      io.observe(el);
    });
  })();

  /* ---------------- Schematic lock-in (candidate flips claim -> evidence) ---------------- */
  (function schematicLock() {
    var wrap = document.getElementById("premise-schematic");
    if (!wrap) return;
    var node = wrap.querySelector(".node-pulse");
    var ring = wrap.querySelector(".focal-ring");
    if (!node) return;

    if (reduceMotion) {
      node.classList.add("locked");
      if (ring) ring.classList.add("in");
      return;
    }

    var io = new IntersectionObserver(
      function (entries, obs) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            setTimeout(function () {
              node.classList.add("locked");
              if (ring) ring.classList.add("in");
            }, 650);
            obs.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.4 }
    );
    io.observe(wrap);
  })();

  /* ---------------- Count-up stats ---------------- */
  (function countUp() {
    var stats = document.querySelectorAll(".stat .num[data-target]");
    if (!stats.length) return;

    function animate(el) {
      var target = parseInt(el.getAttribute("data-target"), 10) || 0;
      var suffix = el.getAttribute("data-suffix") || "";
      if (reduceMotion) {
        el.textContent = target + suffix;
        return;
      }
      var duration = 1100;
      var startTs = null;
      function step(ts) {
        if (startTs === null) startTs = ts;
        var p = Math.min(1, (ts - startTs) / duration);
        var eased = 1 - Math.pow(1 - p, 3);
        el.textContent = Math.round(eased * target) + suffix;
        if (p < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    }

    var io = new IntersectionObserver(
      function (entries, obs) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            animate(entry.target);
            obs.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.6 }
    );
    stats.forEach(function (el) {
      io.observe(el);
    });
  })();

  /* ---------------- Background star parallax ---------------- */
  (function parallax() {
    if (reduceMotion) return;
    var layers = document.querySelectorAll(".starfield .layer");
    if (!layers.length) return;
    var ticking = false;

    function update() {
      var y = window.scrollY;
      layers.forEach(function (el, i) {
        var speed = (i + 1) * -0.03;
        el.style.transform = "translate3d(0," + (y * speed).toFixed(1) + "px,0)";
      });
      ticking = false;
    }

    window.addEventListener(
      "scroll",
      function () {
        if (!ticking) {
          window.requestAnimationFrame(update);
          ticking = true;
        }
      },
      { passive: true }
    );
    update();
  })();

  /* ---------------- FAQ accordion ---------------- */
  (function faq() {
    var items = document.querySelectorAll(".faq-item");
    items.forEach(function (item) {
      var btn = item.querySelector(".faq-q");
      if (!btn) return;
      btn.addEventListener("click", function () {
        var isOpen = item.classList.contains("open");
        item.classList.toggle("open", !isOpen);
        btn.setAttribute("aria-expanded", String(!isOpen));
      });
    });
  })();

  /* ---------------- Copy-to-clipboard buttons ---------------- */
  (function copyButtons() {
    var buttons = document.querySelectorAll(".copy-btn[data-copy-target]");
    buttons.forEach(function (btn) {
      var targetId = btn.getAttribute("data-copy-target");
      var target = document.getElementById(targetId);
      var labelEl = btn.querySelector(".label");
      if (!target || !labelEl) return;
      var defaultLabel = labelEl.textContent;

      btn.addEventListener("click", function () {
        var text = target.textContent || "";
        var done = function () {
          labelEl.textContent = "Copied";
          btn.classList.add("done");
          setTimeout(function () {
            labelEl.textContent = defaultLabel;
            btn.classList.remove("done");
          }, 1600);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done, function () {
            fallbackCopy(text);
            done();
          });
        } else {
          fallbackCopy(text);
          done();
        }
      });
    });

    function fallbackCopy(text) {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand("copy");
      } catch (e) {
        /* clipboard unavailable — the visible text remains selectable */
      }
      document.body.removeChild(ta);
    }
  })();

  /* ---------------- Release fetch (relocated, logic unchanged) ---------------- */
  (function releaseStatus() {
    var REPO = "Ranshiv/ASTRA";
    var dot = document.getElementById("status-dot");
    var text = document.getElementById("status-text");
    var btn = document.getElementById("download-btn");
    var meta = document.getElementById("dl-meta");
    var checksumEl = document.getElementById("checksum");
    var verifyBand = document.getElementById("verify");
    if (!dot || !text || !btn) return;

    function bytesToMiB(n) {
      return (n / (1024 * 1024)).toFixed(1) + " MiB";
    }

    function fallback(message) {
      dot.className = "dot warn";
      text.textContent = message;
      btn.textContent = "See all releases on GitHub";
      btn.href = "https://github.com/" + REPO + "/releases";
      btn.removeAttribute("aria-disabled");
    }

    var controller = new AbortController();
    var timeout = setTimeout(function () {
      controller.abort();
    }, 6000);

    fetch("https://api.github.com/repos/" + REPO + "/releases/latest", {
      headers: { Accept: "application/vnd.github+json" },
      signal: controller.signal,
    })
      .then(function (res) {
        clearTimeout(timeout);
        if (res.status === 404) {
          fallback("No Windows build published yet — build from source below.");
          return null;
        }
        if (!res.ok) {
          fallback("Couldn't reach GitHub — see releases directly.");
          return null;
        }
        return res.json();
      })
      .then(function (release) {
        if (!release) return;
        var assets = release.assets || [];
        var exe = assets.find(function (a) {
          return a.name.toLowerCase().endsWith(".exe");
        });
        if (!exe) {
          fallback("No Windows build published yet — build from source below.");
          return;
        }
        var checksumAsset = assets.find(function (a) {
          return /checksum/i.test(a.name) && /\.(sha256|json)$/i.test(a.name);
        });

        dot.className = "dot ok";
        text.textContent = "Latest release " + (release.tag_name || "");
        btn.textContent = "Download for Windows";
        btn.href = exe.browser_download_url;
        btn.removeAttribute("aria-disabled");

        var published = release.published_at
          ? new Date(release.published_at).toLocaleDateString(undefined, {
              year: "numeric",
              month: "long",
              day: "numeric",
            })
          : null;

        var lines = [];
        lines.push(exe.name + " · " + bytesToMiB(exe.size));
        if (published) lines.push("Published " + published);
        if (meta) {
          meta.innerHTML = "";
          lines.forEach(function (l) {
            var d = document.createElement("div");
            d.textContent = l;
            meta.appendChild(d);
          });
        }

        if (checksumAsset && checksumEl) {
          if (verifyBand) verifyBand.hidden = false;
          checksumEl.textContent =
            "# Verify after downloading (PowerShell)\n" +
            "Get-FileHash .\\" + exe.name + " -Algorithm SHA256\n" +
            "# compare against " + checksumAsset.name + ":\n" +
            "# " + checksumAsset.browser_download_url;
        }
      })
      .catch(function () {
        clearTimeout(timeout);
        fallback("Couldn't reach GitHub — see releases directly.");
      });
  })();
})();
