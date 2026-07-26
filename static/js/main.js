(function () {
  "use strict";

  const state = {
    mode: "tumor",
    lastFile: null,
    lastResult: null,
    heatmapVisible: false,
    scanCounter: 0,
  };

  const els = {
    modeBtns: document.querySelectorAll(".mode-btn"),
    scanFrame: document.getElementById("scanFrame"),
    scanEmpty: document.getElementById("scanEmpty"),
    fileInput: document.getElementById("fileInput"),
    previewImg: document.getElementById("previewImg"),
    heatmapImg: document.getElementById("heatmapImg"),
    sweepLine: document.getElementById("sweepLine"),
    scanReadout: document.getElementById("scanReadout"),
    readoutId: document.getElementById("readoutId"),
    readoutState: document.getElementById("readoutState"),
    toggleHeatmapBtn: document.getElementById("toggleHeatmapBtn"),
    resetBtn: document.getElementById("resetBtn"),
    panelEmpty: document.getElementById("panelEmpty"),
    panelContent: document.getElementById("panelContent"),
    demoBanner: document.getElementById("demoBanner"),
    verdictLabel: document.getElementById("verdictLabel"),
    verdictConf: document.getElementById("verdictConf"),
    confidenceBars: document.getElementById("confidenceBars"),
    historyList: document.getElementById("historyList"),
    step1: document.getElementById("step1"),
    step2: document.getElementById("step2"),
    step3: document.getElementById("step3"),
    systemStatusText: document.getElementById("systemStatusText"),
  };

  // ---------- system status ----------
  fetch("/api/health")
    .then((r) => r.json())
    .then((data) => {
      const bits = [];
      bits.push(data.tumor_model_trained ? "Tumor: trained" : "Tumor: demo");
      bits.push(data.stroke_model_trained ? "Stroke: trained" : "Stroke: demo");
      els.systemStatusText.textContent = bits.join(" · ");
    })
    .catch(() => {
      els.systemStatusText.textContent = "Backend unreachable";
    });

  // ---------- mode switching ----------
  els.modeBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      els.modeBtns.forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      state.mode = btn.dataset.mode;
      resetScan();
    });
  });

  // ---------- upload handling ----------
  els.fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) handleFile(e.target.files[0]);
  });

  ["dragover", "dragenter"].forEach((evt) =>
    els.scanFrame.addEventListener(evt, (e) => {
      e.preventDefault();
      els.scanFrame.style.borderColor = "var(--signal)";
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    els.scanFrame.addEventListener(evt, (e) => {
      e.preventDefault();
      els.scanFrame.style.borderColor = "var(--line)";
    })
  );
  els.scanFrame.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  function handleFile(file) {
    if (!file.type.startsWith("image/")) return;
    state.lastFile = file;

    const reader = new FileReader();
    reader.onload = (ev) => {
      els.scanEmpty.hidden = true;
      els.previewImg.src = ev.target.result;
      els.previewImg.hidden = false;
      startAnalysis(file);
    };
    reader.readAsDataURL(file);
  }

  function setStep(n) {
    [els.step1, els.step2, els.step3].forEach((s, i) => {
      s.classList.toggle("active", i < n);
    });
  }

  function startAnalysis(file) {
    setStep(2);
    els.sweepLine.hidden = false;
    els.scanReadout.hidden = false;
    state.scanCounter += 1;
    const scanId = "SCAN-" + state.scanCounter.toString(16).toUpperCase().padStart(6, "0");
    els.readoutId.textContent = scanId;
    els.readoutState.textContent = "ANALYZING";

    els.panelEmpty.hidden = true;
    els.panelContent.hidden = true;

    const formData = new FormData();
    formData.append("image", file);

    const endpoint = state.mode === "tumor" ? "/api/predict/tumor" : "/api/predict/stroke";
    const minDelay = new Promise((res) => setTimeout(res, 1400)); // let the sweep animation read as "working"

    Promise.all([fetch(endpoint, { method: "POST", body: formData }).then((r) => r.json()), minDelay])
      .then(([data]) => {
        if (data.error) {
          els.readoutState.textContent = "ERROR";
          els.panelEmpty.hidden = false;
          els.panelEmpty.querySelector("p").textContent = data.error;
          return;
        }
        finishAnalysis(data, scanId);
      })
      .catch((err) => {
        els.readoutState.textContent = "ERROR";
        els.panelEmpty.hidden = false;
        els.panelEmpty.querySelector("p").textContent =
          "Could not reach the backend. Is the Flask server running?";
      });
  }

  function finishAnalysis(data, scanId) {
    state.lastResult = data;
    els.sweepLine.hidden = true;
    els.readoutState.textContent = "COMPLETE";
    setStep(3);

    els.heatmapImg.src = "data:image/png;base64," + data.heatmap_image_b64;
    els.heatmapImg.hidden = false;
    els.toggleHeatmapBtn.disabled = false;
    els.resetBtn.disabled = false;
    state.heatmapVisible = false;
    els.heatmapImg.classList.remove("visible");

    els.panelContent.hidden = false;
    els.demoBanner.hidden = !data.demo_mode;

    const topClass = data.predicted_class;
    const topConf = data.confidences[topClass];
    els.verdictLabel.textContent = topClass.replace(/_/g, " ");
    els.verdictConf.textContent = (topConf * 100).toFixed(1) + "% confidence";

    // sort classes by confidence desc
    const sorted = Object.entries(data.confidences).sort((a, b) => b[1] - a[1]);
    els.confidenceBars.innerHTML = "";
    sorted.forEach(([name, conf], idx) => {
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = `
        <span class="bar-name">${name.replace(/_/g, " ")}</span>
        <span class="bar-track"><span class="bar-fill${idx === 0 && name !== "no_tumor" && name !== "normal" ? " is-top" : ""}"></span></span>
        <span class="bar-pct">${(conf * 100).toFixed(1)}%</span>
      `;
      els.confidenceBars.appendChild(row);
      requestAnimationFrame(() => {
        row.querySelector(".bar-fill").style.width = (conf * 100).toFixed(1) + "%";
      });
    });

    addHistoryItem(scanId, topClass, topConf, data.demo_mode);
  }

  function addHistoryItem(scanId, label, conf, demoMode) {
    const emptyMsg = els.historyList.querySelector(".history-empty");
    if (emptyMsg) emptyMsg.remove();

    const item = document.createElement("div");
    item.className = "history-item";
    item.innerHTML = `
      <div>
        <div class="h-label">${label.replace(/_/g, " ")}${demoMode ? " (demo)" : ""}</div>
        <div class="h-id">${scanId} · ${state.mode.toUpperCase()}</div>
      </div>
      <div class="h-conf">${(conf * 100).toFixed(1)}%</div>
    `;
    els.historyList.prepend(item);
  }

  els.toggleHeatmapBtn.addEventListener("click", () => {
    state.heatmapVisible = !state.heatmapVisible;
    els.heatmapImg.classList.toggle("visible", state.heatmapVisible);
    els.toggleHeatmapBtn.textContent = state.heatmapVisible
      ? "Hide Grad-CAM overlay"
      : "Toggle Grad-CAM overlay";
  });

  els.resetBtn.addEventListener("click", resetScan);

  function resetScan() {
    els.previewImg.hidden = true;
    els.previewImg.src = "";
    els.heatmapImg.hidden = true;
    els.heatmapImg.src = "";
    els.heatmapImg.classList.remove("visible");
    els.scanEmpty.hidden = false;
    els.sweepLine.hidden = true;
    els.scanReadout.hidden = true;
    els.toggleHeatmapBtn.disabled = true;
    els.toggleHeatmapBtn.textContent = "Toggle Grad-CAM overlay";
    els.resetBtn.disabled = true;
    els.panelEmpty.hidden = false;
    els.panelEmpty.querySelector("p").textContent = "Results will appear here once a scan is analyzed.";
    els.panelContent.hidden = true;
    els.fileInput.value = "";
    setStep(1);
  }
})();
