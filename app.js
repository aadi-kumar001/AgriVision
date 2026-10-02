/* ==========================================================================
   AgriVision frontend
   Plain JS, no build step, no framework — talks to the Flask API over
   fetch(). Edit API_BASE below if the backend runs somewhere other than
   http://localhost:5000.
   ========================================================================== */

const API_BASE = "http://localhost:5000";

const AgriVision = (() => {

  // ---------------------------------------------------------------- utils
  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function toast(message, isError = false) {
    const host = document.getElementById("toastHost");
    const el = document.createElement("div");
    el.className = "toast" + (isError ? " error" : "");
    el.textContent = message;
    host.appendChild(el);
    setTimeout(() => el.remove(), 4200);
  }

  async function apiFetch(path, options = {}) {
    const res = await fetch(API_BASE + path, options);
    if (!res.ok) {
      let msg = `Request failed (${res.status})`;
      try {
        const body = await res.json();
        if (body.error) msg = body.error;
      } catch (_) {}
      throw new Error(msg);
    }
    return res.json();
  }

  function fmtTime(iso) {
    try {
      const d = new Date(iso);
      return d.toLocaleString(undefined, {
        month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
      });
    } catch (_) { return iso; }
  }

  // ------------------------------------------------------------ navigation
  function goTo(viewName) {
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    document.querySelectorAll(".nav-item").forEach((n) => n.classList.remove("active"));
    document.getElementById("view-" + viewName).classList.add("active");
    const navBtn = document.querySelector(`.nav-item[data-view="${viewName}"]`);
    if (navBtn) navBtn.classList.add("active");

    if (viewName === "dashboard") loadDashboard();
    if (viewName === "market") loadMarket();
    if (viewName === "history") loadHistory();
  }

  function initNav() {
    document.querySelectorAll(".nav-item").forEach((btn) => {
      btn.addEventListener("click", () => goTo(btn.dataset.view));
    });
  }

  // ------------------------------------------------------------ API status
  async function checkApiStatus() {
    const dot = document.getElementById("apiDot");
    const text = document.getElementById("apiStatusText");
    const note = document.getElementById("apiUrlNote");
    note.textContent = API_BASE;
    try {
      await apiFetch("/api/health");
      dot.className = "api-dot on";
      text.textContent = "API connected";
    } catch (e) {
      dot.className = "api-dot off";
      text.textContent = "API unreachable";
    }
  }

  // ------------------------------------------------------------ dashboard
  async function loadDashboard() {
    try {
      const stats = await apiFetch("/api/dashboard/summary");
      document.getElementById("statTotal").textContent = stats.total_analyses;
      document.getElementById("statAlerts").textContent = stats.disease_alerts;
      document.getElementById("statYield").textContent = stats.by_module.yield || 0;
    } catch (e) {
      // leave placeholders — API status indicator covers the error
    }
    try {
      const hist = await apiFetch("/api/history?limit=6");
      const host = document.getElementById("dashRecent");
      if (!hist.length) {
        host.innerHTML = '<div class="result-empty">No activity yet — run a module to populate this log.</div>';
        return;
      }
      host.innerHTML = hist.map(rowHtml).join("");
    } catch (e) { /* ignore */ }
  }

  function rowHtml(row) {
    return `
      <div class="history-row">
        <span class="mod">${escapeHtml(row.module)}</span>
        <span class="ts">${fmtTime(row.created_at)}</span>
        <span>${escapeHtml(row.summary)}</span>
        <span></span>
      </div>`;
  }

  // ------------------------------------------------------------ disease
  function initDisease() {
    const dz = document.getElementById("diseaseDropzone");
    const input = document.getElementById("diseaseFileInput");
    const submitBtn = document.getElementById("diseaseSubmit");
    const resetBtn = document.getElementById("diseaseReset");
    let currentFile = null;

    dz.addEventListener("click", () => input.click());
    dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
    dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
    dz.addEventListener("drop", (e) => {
      e.preventDefault();
      dz.classList.remove("drag");
      if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
    });
    input.addEventListener("change", () => {
      if (input.files.length) setFile(input.files[0]);
    });

    function setFile(file) {
      if (!file.type.startsWith("image/")) {
        toast("Please choose an image file.", true);
        return;
      }
      currentFile = file;
      const url = URL.createObjectURL(file);
      document.getElementById("dzContent").innerHTML =
        `<img class="preview" src="${url}" alt="Leaf preview" /><div class="dz-sub">${escapeHtml(file.name)} — click to replace</div>`;
      submitBtn.disabled = false;
    }

    resetBtn.addEventListener("click", () => {
      currentFile = null;
      input.value = "";
      submitBtn.disabled = true;
      document.getElementById("dzContent").innerHTML = `
        <svg width="30" height="30" viewBox="0 0 24 24" fill="none" style="margin:0 auto;display:block;"><path d="M12 16V4M12 4l-4 4M12 4l4 4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/><path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
        <div class="dz-title">Drop an image, or click to browse</div>
        <div class="dz-sub">JPG or PNG · a single leaf, filling most of the frame, works best</div>`;
      document.getElementById("diseaseResult").innerHTML =
        '<div class="result-empty">Upload a leaf photo to see a diagnosis here.</div>';
    });

    submitBtn.addEventListener("click", async () => {
      if (!currentFile) return;
      submitBtn.disabled = true;
      submitBtn.innerHTML = '<span class="spinner"></span> Analyzing…';
      const resultHost = document.getElementById("diseaseResult");
      try {
        const form = new FormData();
        form.append("image", currentFile);
        const res = await fetch(API_BASE + "/api/disease/detect", { method: "POST", body: form });
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.error || "Analysis failed");
        }
        const data = await res.json();
        resultHost.innerHTML = renderDiseaseResult(data);
      } catch (e) {
        toast(e.message, true);
        resultHost.innerHTML = `<div class="result-empty">${escapeHtml(e.message)}</div>`;
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = "Analyze leaf";
      }
    });
  }

  function renderDiseaseResult(data) {
    const candidates = data.top_candidates.map((c) => `
      <div class="candidate-row">
        <span class="name">${escapeHtml(c.label)}</span>
        <span class="candidate-bar"><span style="width:${c.confidence}%"></span></span>
        <span class="pct">${c.confidence}%</span>
      </div>`).join("");

    return `
      <div class="headline-result">
        <span class="value">${escapeHtml(data.label)}</span>
        <span class="badge ${data.severity}">${escapeHtml(data.severity)} severity</span>
      </div>
      <div class="small-note">Confidence ${data.confidence}% · ${escapeHtml(data.model)}</div>
      <ul class="advisory-list">
        ${data.advisory.map((a) => `<li>${escapeHtml(a)}</li>`).join("")}
      </ul>
      <div style="margin-top:16px; padding-top:14px; border-top:1px solid var(--line);">
        <div class="panel-title" style="border:none; padding:0; margin-bottom:10px;">CLASSIFIER CONFIDENCE</div>
        ${candidates}
      </div>`;
  }

  // ------------------------------------------------------------ yield
  function initYield() {
    document.getElementById("yieldForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      const btn = e.target.querySelector("button[type=submit]");
      const resultHost = document.getElementById("yieldResult");
      const payload = {
        crop: document.getElementById("y_crop").value,
        area_ha: parseFloat(document.getElementById("y_area").value),
        avg_temp_c: parseFloat(document.getElementById("y_temp").value),
        seasonal_rainfall_mm: parseFloat(document.getElementById("y_rain").value),
        soil_ph: parseFloat(document.getElementById("y_ph").value),
        fertilizer_kg_per_ha: parseFloat(document.getElementById("y_fert").value),
        irrigation_type: document.getElementById("y_irrigation").value,
      };
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Predicting…';
      try {
        const data = await apiFetch("/api/yield/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        resultHost.innerHTML = `
          <div class="headline-result">
            <span class="value">${data.predicted_yield_t_per_ha}</span>
            <span class="unit">t/ha &nbsp;±${data.yield_uncertainty_t_per_ha}</span>
          </div>
          <div class="small-note">Total for ${data.area_ha} ha: <strong>${data.predicted_total_production_t} t</strong> · ${escapeHtml(data.model)}</div>
          <ul class="advisory-list">${data.advisory.map((a) => `<li>${escapeHtml(a)}</li>`).join("")}</ul>`;
        loadDashboardQuiet();
      } catch (err) {
        toast(err.message, true);
        resultHost.innerHTML = `<div class="result-empty">${escapeHtml(err.message)}</div>`;
      } finally {
        btn.disabled = false;
        btn.textContent = "Predict yield";
      }
    });
  }

  // ------------------------------------------------------------ soil
  function initSoil() {
    document.getElementById("soilForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      const btn = e.target.querySelector("button[type=submit]");
      const resultHost = document.getElementById("soilResult");
      const payload = {
        nitrogen_mg_kg: parseFloat(document.getElementById("s_n").value),
        phosphorus_mg_kg: parseFloat(document.getElementById("s_p").value),
        potassium_mg_kg: parseFloat(document.getElementById("s_k").value),
        ph: parseFloat(document.getElementById("s_ph").value),
        organic_carbon_pct: parseFloat(document.getElementById("s_oc").value),
        crop: document.getElementById("s_crop").value,
      };
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Scoring…';
      try {
        const data = await apiFetch("/api/soil/health", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const scoreColor = data.overall_score >= 65 ? "var(--forest-light)" : data.overall_score >= 45 ? "var(--amber)" : "var(--danger)";
        const comps = Object.entries(data.component_scores).map(([k, v]) => `
          <tr><td style="text-transform:capitalize;">${escapeHtml(k.replace(/_/g, " "))}</td>
              <td><div class="score-meter" style="width:110px;"><span style="width:${v}%; background:${scoreColor}"></span></div></td>
              <td style="font-family:var(--font-mono);">${v}</td></tr>`).join("");

        resultHost.innerHTML = `
          <div class="headline-result">
            <span class="value" style="color:${scoreColor}">${data.overall_score}</span>
            <span class="unit">/ 100 — ${escapeHtml(data.category)}</span>
          </div>
          <div class="score-meter"><span style="width:${data.overall_score}%; background:${scoreColor}"></span></div>
          <table class="data-table" style="margin-top:14px;"><tbody>${comps}</tbody></table>
          <div style="margin-top:14px;">
            ${data.deficiencies.map((d) => `<span class="tag-pill ${d === "None detected" ? "ok" : ""}">${escapeHtml(d)}</span>`).join("")}
          </div>
          <ul class="advisory-list">${data.recommendations.map((a) => `<li>${escapeHtml(a)}</li>`).join("")}</ul>`;
        loadDashboardQuiet();
      } catch (err) {
        toast(err.message, true);
        resultHost.innerHTML = `<div class="result-empty">${escapeHtml(err.message)}</div>`;
      } finally {
        btn.disabled = false;
        btn.textContent = "Score soil health";
      }
    });
  }

  // ------------------------------------------------------------ irrigation
  function initIrrigation() {
    document.getElementById("irrForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      const btn = e.target.querySelector("button[type=submit]");
      const resultHost = document.getElementById("irrResult");
      const payload = {
        soil_moisture_pct: parseFloat(document.getElementById("i_moisture").value),
        temperature_c: parseFloat(document.getElementById("i_temp").value),
        humidity_pct: parseFloat(document.getElementById("i_humidity").value),
        crop: document.getElementById("i_crop").value,
        growth_stage: document.getElementById("i_stage").value,
      };
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Calculating…';
      try {
        const data = await apiFetch("/api/irrigation/advisory", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const urgencyClass = data.urgency === "high" ? "high" : data.urgency === "medium" ? "medium" : "low";
        resultHost.innerHTML = `
          <div class="headline-result">
            <span class="value">${data.recommended_water_liters_per_sqm}</span>
            <span class="unit">L / m² today</span>
            <span class="badge ${urgencyClass}">${escapeHtml(data.urgency)} urgency</span>
          </div>
          <div class="small-note">Reference ET₀ ${data.reference_et0_mm_day} mm/day · crop water demand ${data.crop_water_demand_mm_day} mm/day · next irrigation in ${data.next_irrigation_in_days} day(s)</div>
          <ul class="advisory-list">${data.advisory.map((a) => `<li>${escapeHtml(a)}</li>`).join("")}</ul>`;
        loadDashboardQuiet();
      } catch (err) {
        toast(err.message, true);
        resultHost.innerHTML = `<div class="result-empty">${escapeHtml(err.message)}</div>`;
      } finally {
        btn.disabled = false;
        btn.textContent = "Get advisory";
      }
    });
  }

  // ------------------------------------------------------------ market
  function sparkline(series) {
    const w = 640, h = 130, pad = 6;
    const prices = series.map((p) => p.price_per_quintal);
    const min = Math.min(...prices), max = Math.max(...prices);
    const range = (max - min) || 1;
    const pts = series.map((p, i) => {
      const x = pad + (i / (series.length - 1)) * (w - pad * 2);
      const y = h - pad - ((p.price_per_quintal - min) / range) * (h - pad * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    const areaPts = `${pad},${h - pad} ` + pts.join(" ") + ` ${w - pad},${h - pad}`;
    return `
      <svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" preserveAspectRatio="none">
        <polygon points="${areaPts}" fill="rgba(47,82,51,0.10)"></polygon>
        <polyline points="${pts.join(" ")}" fill="none" stroke="#2F5233" stroke-width="1.8"></polyline>
      </svg>`;
  }

  async function loadMarket() {
    const crop = document.getElementById("m_crop").value;
    const host = document.getElementById("marketResult");
    host.innerHTML = '<div class="result-empty">Loading market data…</div>';
    try {
      const data = await apiFetch(`/api/market/insights?crop=${encodeURIComponent(crop)}`);
      const trendColor = data.trend_7d === "rising" ? "var(--forest-light)" : data.trend_7d === "falling" ? "var(--danger)" : "var(--amber)";
      const mandiRows = data.nearby_mandis.map((m) => `
        <tr><td>${escapeHtml(m.mandi)}</td><td style="font-family:var(--font-mono)">₹${m.price_per_quintal}</td><td>${m.distance_km} km</td></tr>
      `).join("");

      host.innerHTML = `
        <div class="headline-result">
          <span class="value">₹${data.current_price_per_quintal}</span>
          <span class="unit">per quintal</span>
          <span class="badge" style="color:${trendColor}">${escapeHtml(data.trend_7d)}</span>
        </div>
        <div class="small-note">${escapeHtml(data.advisory)}</div>
        <div style="margin-top:16px;">${sparkline(data.history_30d)}</div>
        <div class="small-note" style="margin-top:2px;">30-day price trend, ${escapeHtml(data.crop)}</div>
        <table class="data-table" style="margin-top:18px;">
          <thead><tr><th>Nearby mandi</th><th>Price / quintal</th><th>Distance</th></tr></thead>
          <tbody>${mandiRows}</tbody>
        </table>`;
    } catch (err) {
      toast(err.message, true);
      host.innerHTML = `<div class="result-empty">${escapeHtml(err.message)}</div>`;
    }
  }

  function initMarket() {
    document.getElementById("m_crop").addEventListener("change", loadMarket);
  }

  // ------------------------------------------------------------ history
  async function loadHistory() {
    const host = document.getElementById("historyResult");
    host.innerHTML = '<div class="result-empty">Loading…</div>';
    try {
      const rows = await apiFetch("/api/history?limit=100");
      if (!rows.length) {
        host.innerHTML = '<div class="result-empty">No analyses logged yet.</div>';
        return;
      }
      host.innerHTML = rows.map(rowHtml).join("");
    } catch (err) {
      host.innerHTML = `<div class="result-empty">${escapeHtml(err.message)}</div>`;
    }
  }

  async function loadDashboardQuiet() {
    // refresh dashboard numbers silently after any module run
    try { await loadDashboard(); } catch (_) {}
  }

  // ------------------------------------------------------------ init
  function init() {
    initNav();
    initDisease();
    initYield();
    initSoil();
    initIrrigation();
    initMarket();
    checkApiStatus();
    loadDashboard();
    setInterval(checkApiStatus, 15000);
  }

  document.addEventListener("DOMContentLoaded", init);

  return { goTo, loadHistory };
})();
