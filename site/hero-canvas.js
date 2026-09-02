// ASTRA download site — hero canvas
// A vermilion reticle drifts across a slate star field, locks onto one star,
// and holds — the product thesis (finding the one object that matters among
// many) rendered as a loop. Pure Canvas 2D, no dependencies, CSP-safe.
(function () {
  "use strict";

  var canvas = document.getElementById("hero-canvas");
  if (!canvas || !canvas.getContext) return;

  var ctx = canvas.getContext("2d");
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  var width = 0;
  var height = 0;
  var stars = [];
  var target = null;
  var rafId = null;
  var running = false;
  var startTime = null;

  // Loop phases, in ms.
  var PHASE_DRIFT = 1800;
  var PHASE_LOCK = 700;
  var PHASE_HOLD = 1600;
  var PHASE_RELEASE = 600;
  var TOTAL = PHASE_DRIFT + PHASE_LOCK + PHASE_HOLD + PHASE_RELEASE;

  function readTokens() {
    var cs = getComputedStyle(document.documentElement);
    return {
      slate: cs.getPropertyValue("--color-text").trim() || "#e6e3dc",
      accent: cs.getPropertyValue("--color-accent").trim() || "#e8563f",
      muted: cs.getPropertyValue("--color-muted").trim() || "#8a93a6",
    };
  }

  function resize() {
    var rect = canvas.getBoundingClientRect();
    width = rect.width;
    height = rect.height;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    seedStars();
  }

  function seedStars() {
    var count = Math.max(18, Math.round((width * height) / 4200));
    stars = [];
    for (var i = 0; i < count; i++) {
      stars.push({
        x: Math.random() * width,
        y: Math.random() * height,
        r: Math.random() * 1.6 + 0.6,
        tw: Math.random() * Math.PI * 2,
      });
    }
    // Pick the target star nearest the visual centre-ish for a clean lock.
    var cx = width * 0.58;
    var cy = height * 0.42;
    var best = stars[0];
    var bestD = Infinity;
    stars.forEach(function (s) {
      var d = Math.hypot(s.x - cx, s.y - cy);
      if (d < bestD) {
        bestD = d;
        best = s;
      }
    });
    if (best) best.r = 2.4;
    target = best;
  }

  function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
  }

  function easeInOutCubic(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }

  function drawFrame(elapsed) {
    var tokens = readTokens();
    ctx.clearRect(0, 0, width, height);

    // Star field, with a gentle twinkle.
    stars.forEach(function (s) {
      var alpha = 0.35 + 0.35 * Math.sin(s.tw + elapsed / 900);
      ctx.beginPath();
      ctx.fillStyle = withAlpha(tokens.slate, Math.max(0.15, alpha));
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fill();
    });

    if (!target) return;

    var t = elapsed % TOTAL;
    var rx, ry, ringR, ringAlpha, calloutAlpha;

    var startX = width * 0.14;
    var startY = height * 0.82;

    if (t < PHASE_DRIFT) {
      var p = easeInOutCubic(t / PHASE_DRIFT);
      rx = startX + (target.x - startX) * p;
      ry = startY + (target.y - startY) * p;
      ringR = 34 - 10 * p;
      ringAlpha = 0.55;
      calloutAlpha = 0;
    } else if (t < PHASE_DRIFT + PHASE_LOCK) {
      var p2 = easeOutCubic((t - PHASE_DRIFT) / PHASE_LOCK);
      rx = target.x;
      ry = target.y;
      ringR = 24 - 14 * p2;
      ringAlpha = 0.55 + 0.45 * p2;
      calloutAlpha = p2;
    } else if (t < PHASE_DRIFT + PHASE_LOCK + PHASE_HOLD) {
      rx = target.x;
      ry = target.y;
      var breathe = Math.sin((t - PHASE_DRIFT - PHASE_LOCK) / 260) * 1.2;
      ringR = 10 + breathe;
      ringAlpha = 1;
      calloutAlpha = 1;
    } else {
      var p3 = (t - PHASE_DRIFT - PHASE_LOCK - PHASE_HOLD) / PHASE_RELEASE;
      rx = target.x;
      ry = target.y;
      ringR = 10 + 26 * easeOutCubic(p3);
      ringAlpha = 1 - p3;
      calloutAlpha = 1 - p3;
    }

    // Target star glow when locked.
    if (calloutAlpha > 0) {
      ctx.beginPath();
      ctx.fillStyle = withAlpha(tokens.accent, calloutAlpha);
      ctx.arc(target.x, target.y, target.r + 1.4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Reticle ring.
    ctx.beginPath();
    ctx.strokeStyle = withAlpha(tokens.accent, ringAlpha);
    ctx.lineWidth = 1.75;
    ctx.arc(rx, ry, Math.max(2, ringR), 0, Math.PI * 2);
    ctx.stroke();

    // Crosshair ticks.
    var tickLen = 6;
    var gap = Math.max(2, ringR) + 4;
    ctx.strokeStyle = withAlpha(tokens.accent, ringAlpha * 0.85);
    ctx.lineWidth = 1.25;
    [
      [0, -1],
      [0, 1],
      [-1, 0],
      [1, 0],
    ].forEach(function (d) {
      ctx.beginPath();
      ctx.moveTo(rx + d[0] * gap, ry + d[1] * gap);
      ctx.lineTo(rx + d[0] * (gap + tickLen), ry + d[1] * (gap + tickLen));
      ctx.stroke();
    });

    // Callout line + label once locked.
    if (calloutAlpha > 0.05) {
      var lx = Math.min(width - 6, rx + 46);
      var ly = ry - 34;
      ctx.beginPath();
      ctx.strokeStyle = withAlpha(tokens.muted, calloutAlpha * 0.6);
      ctx.lineWidth = 1;
      ctx.moveTo(rx + gap * 0.7, ry - gap * 0.7);
      ctx.lineTo(lx, ly);
      ctx.stroke();

      ctx.font = "11px " + fontStack();
      ctx.fillStyle = withAlpha(tokens.text || tokens.slate, calloutAlpha);
      ctx.textBaseline = "bottom";
      ctx.fillText("anomaly · corroborated", Math.min(lx, width - 150), ly - 2);
    }
  }

  function fontStack() {
    return '"Roboto Mono", ui-monospace, Consolas, monospace';
  }

  function withAlpha(hex, alpha) {
    hex = (hex || "").trim();
    if (hex[0] !== "#" || hex.length < 7) return "rgba(232,86,63," + alpha + ")";
    var r = parseInt(hex.slice(1, 3), 16);
    var g = parseInt(hex.slice(3, 5), 16);
    var b = parseInt(hex.slice(5, 7), 16);
    return "rgba(" + r + "," + g + "," + b + "," + alpha + ")";
  }

  function frame(ts) {
    if (!running) return;
    if (startTime === null) startTime = ts;
    drawFrame(ts - startTime);
    rafId = requestAnimationFrame(frame);
  }

  function start() {
    if (running) return;
    running = true;
    rafId = requestAnimationFrame(frame);
  }

  function stop() {
    running = false;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;
  }

  resize();

  if (reduceMotion) {
    // Render a single locked frame, mid-hold, and never animate again.
    drawFrame(PHASE_DRIFT + PHASE_LOCK + 10);
  } else {
    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting && !document.hidden) start();
          else stop();
        });
      },
      { threshold: 0.1 }
    );
    io.observe(canvas);

    document.addEventListener("visibilitychange", function () {
      if (document.hidden) stop();
      else if (canvas.getBoundingClientRect().top < window.innerHeight) start();
    });
  }

  var resizeTimer = null;
  window.addEventListener("resize", function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      resize();
      if (reduceMotion) drawFrame(PHASE_DRIFT + PHASE_LOCK + 10);
    }, 150);
  });

  // Redraw once on theme change so the static (reduced-motion) frame and the
  // very first painted frame pick up the new token colours immediately.
  window.addEventListener("astra:theme-change", function () {
    if (reduceMotion) drawFrame(PHASE_DRIFT + PHASE_LOCK + 10);
  });
})();
