const $ = (id) => document.getElementById(id);
const video = $("video");
const secondaryVideo = $("broadcast-video");
const broadcastOnly = location.pathname === "/broadcast";
const aligned = location.pathname === "/aligned";
const canvas = $("timeline");
const overlay = $("overlay");
const scroll = $("scroll");
const status = $("status");
const ink = canvas.getContext("2d");
const RULER = 30;
const ROW = 29;
const palette = getComputedStyle(document.documentElement);
const colors = Object.fromEntries(
  Object.entries({ H: "home", A: "away", D: "dead", gap: "gap", ball: "ball" })
    .map(([key, name]) => [key, palette.getPropertyValue("--" + name).trim()])
);
const recoveries = [];
let match, broadcast, bounds, publicWindow, frame = 0, zoom = 1, width = 0, durationFrames = 0;
let trackingError = "";
let dataReady = false;
let pointerFrame = null;
let lastPair = null;
let primaryWaiting = false;
let paintPrimary = () => {};
let paintSecondary = () => {};

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const number = (value) => value.toLocaleString("en-US");
const two = (value) => String(value).padStart(2, "0");

function timecode(index, withFrame = true) {
  const seconds = Math.floor(index / match.fps);
  const clock = [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, seconds % 60].map(two).join(":");
  return withFrame ? clock + ":" + two(index % match.fps) : clock;
}

function visibleStart() {
  return bounds[0] + scroll.scrollLeft / width * durationFrames;
}

function position(index) {
  return (index - visibleStart()) / durationFrames * width;
}

function firstIntersect(spans, index) {
  let low = 0;
  let high = spans.length;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (spans[mid][1] <= index) low = mid + 1;
    else high = mid;
  }
  return low;
}

function spanAt(spans, index) {
  const span = spans[firstIntersect(spans, index)];
  return span && span[0] <= index ? span : null;
}

function paintTimeline() {
  const start = visibleStart();
  const end = Math.min(bounds[1], start + durationFrames);
  ink.fillStyle = "#121416";
  ink.fillRect(0, 0, width, canvas.clientHeight);
  for (const [i, track] of match.tracks.entries()) {
    ink.fillStyle = i % 2 ? "#191c1e" : "#1c1f21";
    ink.fillRect(0, RULER + i * ROW, width, ROW - 1);
    const spans = track.spans;
    for (let j = firstIntersect(spans, start); j < spans.length && spans[j][0] < end; j++) {
      const [from, to, value] = spans[j];
      const x = position(Math.max(from, start));
      const size = Math.max(1, position(Math.min(to, end)) - x);
      const y = RULER + i * ROW + 5;
      ink.globalAlpha = track.kind === "gap" && i > 0 ? Math.min(1, 0.42 + value * 0.12) : 1;
      const color = track.kind === "ball-availability" ? (value === "Present" ? "ball" : "gap") :
        track.kind === "gap" ? "gap" : value;
      ink.fillStyle = track.color || colors[color];
      ink.fillRect(x, y, size, ROW - 10);
      ink.globalAlpha = 1;
      if ((track.kind === "gt" || track.kind === "prediction") && size >= 20) {
        ink.fillStyle = "#101417";
        ink.font = "bold 11px monospace";
        ink.fillText(value, x + 6, y + 13);
      }
    }
  }
  const halves = match.halves;
  const outside = halves ? [[0, halves[0][0]], [halves[0][1], halves[1][0]],
    [halves[1][1], match.frames]] : [];
  for (const [from, to] of outside) {
    const left = position(Math.max(from, start));
    const right = position(Math.min(to, end));
    if (right <= left) continue;
    ink.save();
    ink.beginPath();
    ink.rect(left, RULER, right - left, canvas.clientHeight - RULER);
    ink.clip();
    ink.fillStyle = "#16181a";
    ink.fillRect(left, RULER, right - left, canvas.clientHeight - RULER);
    ink.strokeStyle = "#282b2d";
    ink.beginPath();
    const height = canvas.clientHeight - RULER;
    for (let x = left - height; x < right; x += 8) {
      ink.moveTo(x, canvas.clientHeight);
      ink.lineTo(x + height, RULER);
    }
    ink.stroke();
    ink.restore();
  }
  const target = durationFrames * 95 / width;
  const steps = [1, 5, match.fps, 5 * match.fps, 15 * match.fps, 30 * match.fps,
    60 * match.fps, 120 * match.fps, 300 * match.fps, 600 * match.fps, 1800 * match.fps];
  const step = steps.find((candidate) => candidate >= target) ?? steps.at(-1);
  ink.font = "11px monospace";
  ink.fillStyle = "#b9bcb6";
  ink.strokeStyle = "#a2a7a41f";
  ink.beginPath();
  for (let i = Math.ceil(start / step) * step; i < end; i += step) {
    const x = position(i);
    ink.moveTo(x + 0.5, RULER - 6);
    ink.lineTo(x + 0.5, canvas.clientHeight);
    ink.fillText(timecode(i, step < match.fps), x + 5, 17);
  }
  ink.stroke();
  paintPlayhead();
}

function paintPlayhead() {
  const x = position(frame + .5);
  const head = $("playhead");
  head.hidden = x < 0 || x > width;
  head.style.left = x + "px";
}

function describe(track, index, row) {
  const span = spanAt(track.spans, index);
  if (span) {
    if (track.kind === "alignment") return "Aligned";
    if (track.kind === "ball-availability") return span[2];
    if (track.kind === "gap" && row === 0) return "Missing";
    if (track.kind === "recovery") return "Recovered";
    if (track.kind === "gap") return `${span[2]} missing`;
    return span[2];
  }
  if (track.kind !== "gap") return "—";
  return match.halves.some(([a, b]) => a <= index && index < b) ? (row === 0 ? "Present" : "0 missing") : "—";
}

function updateDetail(index, row) {
  const tracks = match.tracks;
  const track = tracks[row];
  const prefix = `${timecode(index)} · Frame ${number(index)}`;
  if (track) {
    const span = spanAt(track.spans, index);
    const interval = span ? ` · Frames ${number(span[0])}–${number(span[1] - 1)}` : "";
    $("detail").textContent = `${prefix} · ${track.name} · ${describe(track, index, row)}${interval}`;
  } else {
    const values = tracks.map((item, i) => `${item.name}: ${describe(item, index, i)}`);
    $("detail").textContent = [prefix, ...values].join("  |  ");
  }
}

function updateFrame(index, follow = false) {
  frame = clamp(index, 0, match.frames - 1);
  $("time").textContent = timecode(frame);
  if (document.activeElement !== $("frame")) $("frame").value = frame;
  if (follow && frame >= bounds[0] && frame < bounds[1] && position(frame) >= width - 1) {
    scroll.scrollLeft += width * .8;
  }
  paintPlayhead();
  for (const [row, track] of match.tracks.entries()) {
    const label = $("labels").children[row + 1].lastElementChild;
    const value = describe(track, frame, row);
    label.textContent = value;
    label.style.color = value === "Recovered" ? track.color :
      track.kind === "ball-availability" && value === "Present" ? colors.ball : colors[value] ||
      (value.includes("missing") && !value.startsWith("0") || value === "Missing" ? colors.gap : "");
  }
  if (pointerFrame === null) updateDetail(frame);
  if (aligned && dataReady) syncBroadcast();
}

function syncBroadcast() {
  const pair = spanAt(broadcast.pairs, frame);
  const mapped = pair ? frame - pair[0] + pair[2] : null;
  $("broadcast-frame").textContent = mapped === null ? "Unmapped" : `Frame ${number(mapped)}`;
  const message = $("alignment-message");
  if (mapped === null || secondaryVideo.error || !secondaryVideo.readyState) {
    secondaryVideo.pause();
    message.textContent = secondaryVideo.error ? "Broadcast video unavailable" :
      mapped === null ? "No verified broadcast match at this frame" : "Loading broadcast video…";
    message.hidden = false;
    lastPair = null;
    return;
  }
  const target = (mapped + .5) / match.fps;
  const playing = !video.paused && !primaryWaiting && !video.seeking;
  const secondaryFrame = Math.floor(secondaryVideo.currentTime * match.fps + .001);
  if (pair !== lastPair || (!playing && secondaryFrame !== mapped)
      || (playing && Math.abs(secondaryVideo.currentTime - target) > .12)) {
    secondaryVideo.currentTime = target;
  }
  lastPair = pair;
  message.textContent = "Synchronizing broadcast…";
  message.hidden = !secondaryVideo.seeking && secondaryVideo.readyState >= 2;
  if (playing && secondaryVideo.paused) {
    secondaryVideo.play().catch((error) => {
      if (error.name !== "AbortError") status.textContent = "Could not play the aligned broadcast video.";
    });
  } else if (!playing && !secondaryVideo.paused) {
    secondaryVideo.pause();
  }
}

function resizeCanvas(element, context, w, h) {
  const scale = window.devicePixelRatio || 1;
  element.width = Math.round(w * scale);
  element.height = Math.round(h * scale);
  context.setTransform(scale, 0, 0, scale, 0, 0);
}

function resizeTimeline(anchor = frame, fraction = .5) {
  width = scroll.clientWidth;
  durationFrames = (bounds[1] - bounds[0]) / zoom;
  $("extent").style.width = width * zoom + "px";
  $("timeline-window").style.width = width + "px";
  const height = RULER + ROW * match.tracks.length;
  canvas.style.height = height + "px";
  $("playhead").style.height = height + "px";
  resizeCanvas(canvas, ink, width, height);
  scroll.scrollLeft = (anchor - bounds[0]) / durationFrames * width - fraction * width;
  $("out").disabled = zoom === 1;
  $("in").disabled = durationFrames <= match.fps * 2;
  paintTimeline();
}

function reportStatus() {
  const broadcastTracking = broadcast?.teams ? " · Broadcast ATD overlay" : " · Broadcast ATD unavailable";
  const source = broadcastOnly || aligned
    ? `${number(broadcast.mappedFrames)} verified frame pairs · H/A/D labels from the tactical timeline${broadcastTracking}`
    : "Raw tactical tracking";
  status.textContent = trackingError || (dataReady ? `${number(match.frames)} frames · ${match.fps} fps · ${source}` : "Loading video…");
}

function trackingOverlay(element, player, feed, path, extra, currentFrame) {
  const context = element.getContext("2d");
  const chunks = new Map();
  const loadChunk = (index) => {
    if (chunks.has(index)) return;
    if (chunks.size >= 12) chunks.delete(chunks.keys().next().value);
    const request = fetch(path + index)
      .then((response) => {
        if (!response.ok) throw new Error("Could not load tracking coordinates");
        return response.arrayBuffer();
      })
      .then((buffer) => {
        if (chunks.get(index) !== request) return;
        chunks.set(index, new Float32Array(buffer));
        if (Math.floor(currentFrame() / feed.chunk) === index) paint();
      })
      .catch((error) => {
        trackingError = error.message;
        reportStatus();
      });
    chunks.set(index, request);
  };
  function paint() {
    const w = element.clientWidth;
    const h = element.clientHeight;
    context.clearRect(0, 0, w, h);
    if (!dataReady || player.seeking || !player.videoWidth) return;
    const index = currentFrame();
    if (index < 0 || index >= feed.frames) return;
    const chunk = Math.floor(index / feed.chunk);
    loadChunk(chunk);
    if (!player.paused && index % feed.chunk > feed.chunk - 50 &&
        (chunk + 1) * feed.chunk < feed.frames) loadChunk(chunk + 1);
    const points = chunks.get(chunk);
    if (!(points instanceof Float32Array)) return;
    drawTracking(context, points, index % feed.chunk, feed.teams, extra, w, h, player);
  }
  new ResizeObserver(() => {
    resizeCanvas(element, context, element.clientWidth, element.clientHeight);
    paint();
  }).observe(element);
  return paint;
}

function drawTracking(dots, points, index, teams, extra, w, h, player) {
  const ballIndex = teams.length;
  const n = ballIndex + 1 + extra.length;
  const offset = index * n * 2;
  const sx = w / player.videoWidth;
  const sy = h / player.videoHeight;
  for (let i = 0; i < n; i++) {
    const ball = i >= ballIndex;
    const recovery = extra[i - ballIndex - 1];
    const team = ball ? "ball" : teams[i] === 0 ? "home" : "away";
    if (!(recovery?.toggle ?? $(team)).checked) continue;
    const x = points[offset + i * 2] * sx;
    const y = points[offset + i * 2 + 1] * sy;
    if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || x > w || y < 0 || y > h) continue;
    if (recovery) {
      dots.beginPath();
      dots.arc(x, y, recovery.radius, 0, Math.PI * 2);
      dots.lineWidth = 2;
      dots.strokeStyle = recovery.color;
      dots.stroke();
      continue;
    }
    dots.beginPath();
    dots.arc(x, y, ball ? 6 : 4, 0, Math.PI * 2);
    dots.lineWidth = ball ? 2 : 1.5;
    dots.strokeStyle = "#080a0d";
    dots.fillStyle = colors[ball ? team : team === "home" ? "H" : "A"];
    dots.fill();
    dots.stroke();
    if (ball) {
      dots.beginPath();
      dots.arc(x, y, 10, 0, Math.PI * 2);
      dots.lineWidth = 1.5;
      dots.strokeStyle = colors[team];
      dots.stroke();
    }
  }
}

function paintOverlay() {
  paintPrimary();
  paintSecondary();
}

function renderVideoAt(seconds, follow = false) {
  if (!Number.isFinite(seconds)) return;
  updateFrame(Math.floor(seconds * match.fps + .001), follow);
  paintPrimary();
}

function seek(index, reveal = false) {
  if (!dataReady || !Number.isFinite(index)) return;
  const target = clamp(Math.round(index), 0, match.frames - 1);
  if (target < bounds[0] || target >= bounds[1]) setView(false, target);
  if (reveal && (position(target) < 0 || position(target) >= width)) {
    scroll.scrollLeft = (target - bounds[0]) / durationFrames * width - width / 2;
  }
  video.currentTime = (target + .5) / match.fps;
  updateFrame(target);
  paintPrimary();
}

function setView(showPublic, anchor = frame) {
  bounds = showPublic ? publicWindow : [0, match.frames];
  $("full").setAttribute("aria-pressed", !showPublic);
  $("public").setAttribute("aria-pressed", showPublic);
  zoom = 1;
  resizeTimeline(anchor);
}

function pointerIndex(event) {
  const rect = canvas.getBoundingClientRect();
  return clamp(Math.floor(visibleStart() + (event.clientX - rect.left) / width * durationFrames),
    bounds[0], bounds[1] - 1);
}

function pointerMove(event) {
  pointerFrame = pointerIndex(event);
  const row = Math.floor((event.clientY - canvas.getBoundingClientRect().top - RULER) / ROW);
  updateDetail(pointerFrame, row);
  if (canvas.hasPointerCapture(event.pointerId)) seek(pointerFrame);
}

async function main() {
  const title = broadcastOnly ? "Broadcast Timeline" : aligned ? "Aligned Feeds" : "Tactical Timeline";
  $("title").textContent = title;
  document.title = title + " · MoreThanADonut";
  document.querySelector(".feeds [aria-current]").removeAttribute("aria-current");
  document.querySelector(`.feeds a[href="${location.pathname}"]`)?.setAttribute("aria-current", "page");
  const response = await fetch(broadcastOnly ? "/broadcast.json" : "/match.json");
  if (!response.ok && broadcastOnly) throw new Error("Broadcast view requires the local MP4 and a verified frame map.");
  if (!response.ok) throw new Error("Could not load the match timeline");
  match = await response.json();
  if (broadcastOnly || aligned) {
    if (broadcastOnly) {
      broadcast = match;
      document.querySelector(".overlays").hidden = !broadcast.teams;
      $("gap-legend").hidden = !broadcast.alignedBall;
      $("outside-legend").hidden = true;
      video.setAttribute("aria-label", "Broadcast match footage");
    } else {
      const response = await fetch("/broadcast.json");
      if (!response.ok) throw new Error("Aligned view requires the local broadcast MP4 and a verified frame map.");
      broadcast = await response.json();
      if (broadcast.alignedBall) match.tracks.splice(1, 0, broadcast.alignedBall);
      $("primary-caption").hidden = false;
      $("secondary").hidden = false;
      document.querySelector(".screens").classList.add("aligned");
      if (broadcast.teams) {
        let displayed = -1;
        const paint = trackingOverlay($("broadcast-overlay"), secondaryVideo, broadcast,
          "/broadcast-positions/", [], () => displayed);
        paintSecondary = paint;
        const showFrame = (seconds) => {
          displayed = Math.floor(seconds * broadcast.fps + .001);
          paint();
        };
        secondaryVideo.addEventListener("seeking", () => {
          displayed = -1;
          paint();
        });
        secondaryVideo.addEventListener("seeked", () => showFrame(secondaryVideo.currentTime));
        if ("requestVideoFrameCallback" in secondaryVideo) {
          const onFrame = (_now, { mediaTime }) => {
            if (!secondaryVideo.seeking) showFrame(mediaTime);
            secondaryVideo.requestVideoFrameCallback(onFrame);
          };
          secondaryVideo.requestVideoFrameCallback(onFrame);
        } else {
          secondaryVideo.addEventListener("timeupdate", () => showFrame(secondaryVideo.currentTime));
        }
      }
      for (const event of ["loadeddata", "seeked", "error"]) {
        secondaryVideo.addEventListener(event, () => { if (dataReady) syncBroadcast(); });
      }
      secondaryVideo.src = "/broadcast.mp4";
    }
    const spans = broadcast.pairs.map(([first, stop, start]) => broadcastOnly
      ? [start, start + stop - first, 1] : [first, stop, 1]);
    match.tracks.push({ name: "Live alignment", kind: "alignment", color: palette.getPropertyValue("--alignment"), spans });
    $("alignment-legend").hidden = false;
    $("ball-availability-legend").hidden = !broadcast.alignedBall;
  }
  for (const [index, track] of match.tracks.filter((item) => item.kind === "recovery").entries()) {
    const label = document.createElement("label");
    label.style.color = track.color;
    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.checked = true;
    toggle.onchange = paintOverlay;
    label.append(toggle, track.name);
    document.querySelector(".overlays").append(label);
    recoveries.push({ toggle, color: track.color, radius: 8 + index * 4 });
  }
  if (match.teams) {
    paintPrimary = trackingOverlay(overlay, video, match,
      broadcastOnly ? "/broadcast-positions/" : "/positions/", recoveries, () => frame);
  }
  const gt = match.tracks.find((track) => track.kind === "gt").spans;
  publicWindow = gt.length ? [gt[0][0], gt.at(-1)[1]] : null;
  $("public").disabled = !publicWindow;
  $("frame").max = match.frames - 1;
  for (const name of ["Video timecode", ...match.tracks.map((track) => track.name)]) {
    const row = document.createElement("div");
    row.className = "track-label";
    const title = document.createElement("span");
    title.textContent = name;
    title.title = name;
    row.append(title, document.createElement("span"));
    $("labels").append(row);
  }
  setView(false);
  updateFrame(0);

  $("full").onclick = () => setView(false);
  $("public").onclick = () => {
    setView(true, publicWindow[0]);
    seek(publicWindow[0]);
  };
  for (const [id, factor] of [["in", 2], ["out", .5]]) {
    $(id).onclick = () => {
      const start = visibleStart();
      const anchor = frame >= start && frame < start + durationFrames ? frame : start + durationFrames / 2;
      zoom = clamp(zoom * factor, 1, Math.max(1, (bounds[1] - bounds[0]) / (match.fps * 2)));
      resizeTimeline(anchor, .5);
    };
  }
  new ResizeObserver(() => {
    if (scroll.clientWidth !== width) resizeTimeline();
  }).observe(scroll);
  scroll.addEventListener("scroll", paintTimeline);
  canvas.addEventListener("pointermove", pointerMove);
  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    video.pause();
    canvas.setPointerCapture(event.pointerId);
    pointerMove(event);
  });
  for (const event of ["pointerleave", "lostpointercapture"]) {
    canvas.addEventListener(event, () => { pointerFrame = null; updateDetail(frame); });
  }

  const step = (delta) => { video.pause(); seek(frame + delta, true); };
  const togglePlay = () => {
    if (!dataReady) return;
    if (video.paused) video.play().catch((error) => { status.textContent = error.message; });
    else video.pause();
  };
  $("play").onclick = togglePlay;
  $("previous").onclick = () => step(-1);
  $("next").onclick = () => step(1);
  $("frame").onchange = () => {
    if ($("frame").value !== "") seek($("frame").valueAsNumber, true);
    $("frame").value = frame;
  };
  for (const id of ["ball", "home", "away"]) $(id).onchange = paintOverlay;
  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, button, a")) return;
    if (event.code === "Space") togglePlay();
    else if (event.key === "ArrowLeft") step(-1);
    else if (event.key === "ArrowRight") step(1);
    else return;
    event.preventDefault();
  });
  video.addEventListener("loadedmetadata", () => {
    if (Math.abs(video.duration * match.fps - match.frames) > 1) {
      status.textContent = "Video duration does not match the expected frame count.";
      return;
    }
    dataReady = true;
    overlay.parentElement.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    seek(publicWindow?.[0] ?? 0);
    reportStatus();
  });
  video.addEventListener("play", () => { $("play").textContent = "Pause"; });
  video.addEventListener("pause", () => { $("play").textContent = "Play"; });
  video.addEventListener("seeking", () => {
    paintPrimary();
    if (aligned) {
      secondaryVideo.pause();
      $("alignment-message").hidden = false;
    }
  });
  video.addEventListener("seeked", () => renderVideoAt(video.currentTime));
  if (aligned) {
    for (const event of ["playing", "waiting", "pause"]) {
      video.addEventListener(event, () => {
        primaryWaiting = event === "waiting";
        if (dataReady) syncBroadcast();
      });
    }
  }
  if ("requestVideoFrameCallback" in video) {
    const onFrame = (_now, { mediaTime }) => {
      if (!video.seeking && dataReady) renderVideoAt(mediaTime, !video.paused);
      video.requestVideoFrameCallback(onFrame);
    };
    video.requestVideoFrameCallback(onFrame);
  } else {
    const tick = () => {
      if (!video.paused && dataReady) renderVideoAt(video.currentTime, true);
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  video.addEventListener("error", () => {
    status.textContent = "Could not play the MP4. Check that it is available locally and that your browser supports H.264.";
  });
  video.src = broadcastOnly ? "/broadcast.mp4" : "/video.mp4";
  reportStatus();
}

main().catch((error) => { status.textContent = error.message; });
