/**
 * Liquid Tree AI - Frontend Application Logic
 * Simulated telemetry polling, experimental growth estimates, and canvas-based
 * microscopy cell detection visualizer.
 */

document.addEventListener('DOMContentLoaded', () => {
  initApp();
});

let currentDetections = [];
let currentImageObj = null;
let microscopyRequestId = 0;
let previewRequestId = 0;
let currentDashboardView = 'overview';
let historicalChartData = null;

async function parseJsonResponse(response) {
  if (!response.ok) throw new Error(`Dashboard request failed (${response.status})`);
  return response.json();
}

function setConnectionStatus(connected) {
  const chip = document.getElementById('reactor-status-chip');
  const label = document.getElementById('chip-status-text');
  if (!chip || !label) return;
  label.textContent = connected ? 'SIMULATED DATA · CONNECTED' : 'LOCAL API UNAVAILABLE';
  chip.classList.toggle('offline', !connected);
  chip.classList.toggle('optimal', connected);
}

function initApp() {
  initSectionNavigation();
  fetchOverviewAndTelemetry();
  initPredictionEngine();
  initMicroscopyInspector();
  fetchAlertsAndRecommendations();
  initActuatorHub();
  initMobileAMR();

  document.getElementById('btn-refresh-all').addEventListener('click', () => {
    fetchOverviewAndTelemetry();
    fetchAlertsAndRecommendations();
  });
}

function initSectionNavigation() {
  const buttons = [...document.querySelectorAll('[data-section-button]')];
  const availableViews = new Set(buttons.map(button => button.dataset.sectionButton));
  let savedView = 'overview';
  try {
    const storedView = localStorage.getItem('4algae-dashboard-view');
    if (availableViews.has(storedView)) savedView = storedView;
  } catch (_) { /* Storage can be disabled; the default view still works. */ }

  function showView(view) {
    currentDashboardView = view;
    buttons.forEach(button => {
      const active = button.dataset.sectionButton === view;
      button.classList.toggle('active', active);
      button.setAttribute('aria-selected', String(active));
      button.tabIndex = active ? 0 : -1;
    });
    document.querySelectorAll('[data-dashboard-view]').forEach(section => {
      section.hidden = section.dataset.dashboardView !== view;
    });
    document.querySelectorAll('.section-columns-2').forEach(row => {
      const visiblePanels = [...row.querySelectorAll(':scope > [data-dashboard-view]')].filter(panel => !panel.hidden);
      row.hidden = visiblePanels.length === 0;
      row.dataset.singlePanel = String(visiblePanels.length === 1);
    });
    try { localStorage.setItem('4algae-dashboard-view', view); } catch (_) { /* Navigation works without storage. */ }
    if (view === 'trends') renderHistoricalCharts();
  }

  buttons.forEach((button, index) => {
    button.addEventListener('click', () => showView(button.dataset.sectionButton));
    button.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const nextIndex = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 :
        (index + (event.key === 'ArrowRight' ? 1 : buttons.length - 1)) % buttons.length;
      buttons[nextIndex].focus();
      showView(buttons[nextIndex].dataset.sectionButton);
    });
  });
  showView(savedView);
}

/**
 * 1. Telemetry and Overview Fetching
 */
async function fetchOverviewAndTelemetry() {
  try {
    const [resStatus, resTelem] = await Promise.all([
      fetch('/api/status').then(parseJsonResponse),
      fetch('/api/telemetry').then(parseJsonResponse)
    ]);
    setConnectionStatus(true);

    // Update Overview Cards
    document.getElementById('stat-cell-density').textContent = resStatus.current_cell_density_million_ml;
    document.getElementById('stat-dry-biomass').textContent = `${resStatus.estimated_dry_biomass_g_l} g/L`;
    document.getElementById('stat-co2-fixed').textContent = resStatus.daily_co2_sequestered_kg;
    document.getElementById('stat-culture-day').textContent = `Day ${resStatus.culture_day}`;

    // Update Telemetry Sensors
    const s = resTelem.sensors;
    if (s) {
      document.getElementById('val-ph').textContent = s.ph.value;
      document.getElementById('val-temp').textContent = s.temperature.value;
      document.getElementById('val-do').textContent = s.dissolved_oxygen.value;
      document.getElementById('val-co2').textContent = s.co2_inflow_concentration.value;

      // Update bars
      document.getElementById('bar-ph').style.width = `${Math.min(Math.max((s.ph.value - 6.0) / 3.0 * 100, 5), 100)}%`;
      document.getElementById('bar-temp').style.width = `${Math.min(Math.max((s.temperature.value - 10.0) / 25.0 * 100, 5), 100)}%`;
      document.getElementById('bar-do').style.width = `${Math.min(Math.max(s.dissolved_oxygen.value / 15.0 * 100, 5), 100)}%`;
      document.getElementById('bar-co2').style.width = `${Math.min(Math.max(s.co2_inflow_concentration.value / 1000.0 * 100, 5), 100)}%`;
    }
  } catch (err) {
    setConnectionStatus(false);
    console.warn('Telemetry fetch notice:', err);
  }
}

/**
 * 2. Experimental Formula-Based Growth Estimate
 */
function initPredictionEngine() {
  const btnPredict = document.getElementById('btn-run-prediction');
  btnPredict.addEventListener('click', runPrediction);

  // Auto-run once at start
  runPrediction();
}

async function runPrediction() {
  const density = parseFloat(document.getElementById('input-density').value) || 18.4;
  const ph = parseFloat(document.getElementById('input-ph').value) || 7.42;
  const light = parseFloat(document.getElementById('input-light').value) || 240.0;
  const co2 = parseFloat(document.getElementById('input-co2').value) || 412.0;
  const day = parseInt(document.getElementById('input-culture-day')?.value, 10) || 8;

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        current_density: density,
        current_ph: ph,
        light: light,
        co2_ppm: co2,
        culture_day: day
      })
    }).then(parseJsonResponse);

    const p = res.predictions;
    if (p) {
      document.getElementById('pred-next-density').textContent = p.predicted_next_day_density_million_ml;
      const deltaSign = p.predicted_density_delta >= 0 ? '+' : '';
      const pct = ((p.predicted_density_delta / density) * 100).toFixed(1);
      document.getElementById('pred-delta-text').textContent =
        `Formula estimate change: ${deltaSign}${p.predicted_density_delta}M cells/mL (${deltaSign}${pct}%)`;
      document.getElementById('pred-growth-rate').textContent = `${p.predicted_specific_growth_rate_mu_day} day⁻¹`;
      document.getElementById('pred-doubling-time').textContent = `${p.doubling_time_days} days`;
    }

    // Update Early-Warning Culture Crash Detector
    const cr = res.crash_risk_assessment;
    if (cr) {
      const riskPill = document.getElementById('risk-status-pill');
      const riskBar = document.getElementById('risk-meter-fill');
      const riskStability = document.getElementById('risk-stability-hrs');
      const riskStressor = document.getElementById('risk-top-stressor');
      const riskAdvisory = document.getElementById('risk-advisory-text');

      if (riskPill && riskBar) {
        const pct = cr.crash_risk_index_pct;
        const levelClass = pct >= 60 ? 'critical' : (pct >= 30 ? 'elevated' : 'low');
        riskPill.textContent = `${cr.risk_level.replace(/_/g, ' ')} (${pct}%)`;
        riskPill.className = `risk-status-pill ${levelClass}`;
        riskBar.style.width = `${Math.min(100, Math.max(5, pct))}%`;
        riskBar.className = `risk-meter-fill ${levelClass}`;
      }
      if (riskStability) {
        riskStability.textContent = `${cr.predicted_stability_window_hours}h`;
      }
      if (riskStressor) {
        let topKey = 'None';
        let maxScore = 0;
        if (cr.stress_factor_breakdown) {
          for (const [k, v] of Object.entries(cr.stress_factor_breakdown)) {
            if (v.score > maxScore) {
              maxScore = v.score;
              topKey = v.status.replace(/_/g, ' ');
            }
          }
        }
        riskStressor.textContent = maxScore > 0 ? topKey : 'Balanced';
      }
      if (riskAdvisory) {
        riskAdvisory.textContent = cr.early_warning_advisory;
      }
    }

    // Update Biophysical Grey-Box Kinetics
    const gb = res.biophysical_grey_box;
    if (gb) {
      document.getElementById('gb-monod-mu').textContent = `${gb.monod_specific_growth_rate} day⁻¹`;
      document.getElementById('gb-light-eff').textContent = `${gb.effective_photon_flux_ppfd} μmol`;
      document.getElementById('gb-analytical-density').textContent = `${gb.analytical_first_principles_density}M`;
      const resSign = gb.grey_box_residual >= 0 ? '+' : '';
      document.getElementById('gb-residual').textContent = `${resSign}${gb.grey_box_residual}M`;
    }
  } catch (err) {
    document.getElementById('pred-delta-text').textContent = 'Estimate unavailable. Check the local API and try again.';
    console.error('Prediction request error:', err);
  }
}

/**
 * 3. Automated Microscopy Inspection (YOLOv8n Proof of Concept)
 */
function initMicroscopyInspector() {
  const sampleSelect = document.getElementById('select-microscopy-sample');
  const confSlider = document.getElementById('slider-conf-thresh');
  const confValText = document.getElementById('conf-thresh-val');
  const btnRunVision = document.getElementById('btn-run-vision-detection');
  const toggleBoxes = document.getElementById('toggle-boxes');
  const toggleLabels = document.getElementById('toggle-labels');

  confSlider.addEventListener('input', (e) => {
    confValText.textContent = e.target.value;
  });

  sampleSelect.addEventListener('change', () => {
    document.getElementById('viewport-filename').textContent = `${sampleSelect.value}.png`;
    loadMicroscopyPreview(sampleSelect.value);
    executeCellDetection();
  });

  btnRunVision.addEventListener('click', executeCellDetection);
  toggleBoxes.addEventListener('change', redrawCanvas);
  toggleLabels.addEventListener('change', redrawCanvas);

  // Show the source frame immediately; inference may take longer on a CPU.
  loadMicroscopyPreview(sampleSelect.value);
  executeCellDetection();
}

async function loadMicroscopyPreview(sampleId) {
  const requestId = ++previewRequestId;
  try {
    const response = await fetch(`/api/vision/sample-image?sample_id=${encodeURIComponent(sampleId)}`);
    if (!response.ok) throw new Error(`Sample image request failed (${response.status})`);
    const payload = await response.json();
    if (requestId !== previewRequestId) return;
    const image = new Image();
    image.onload = () => {
      if (requestId !== previewRequestId) return;
      currentDetections = [];
      currentImageObj = image;
      redrawCanvas();
      document.getElementById('microscopy-empty-state')?.classList.add('hidden');
    };
    image.onerror = () => showMicroscopyEmptyState('The selected microscopy image could not be displayed.');
    image.src = payload.image_data_uri;
  } catch (err) {
    if (requestId === previewRequestId) showMicroscopyEmptyState('Sample image unavailable. Check that the local validation images are present.');
    console.warn('Microscopy preview warning:', err);
  }
}

function showMicroscopyEmptyState(message) {
  const emptyState = document.getElementById('microscopy-empty-state');
  if (emptyState) {
    emptyState.textContent = message;
    emptyState.classList.remove('hidden');
  }
}

async function executeCellDetection() {
  const requestId = ++microscopyRequestId;
  const sampleId = document.getElementById('select-microscopy-sample').value;
  const confThreshold = parseFloat(document.getElementById('slider-conf-thresh').value) || 0.25;
  const loader = document.getElementById('canvas-loader');

  loader.classList.remove('hidden');
  document.getElementById('vision-status').textContent = 'Running local CPU inference...';

  try {
    const res = await fetch('/api/vision/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_id: sampleId, conf_threshold: confThreshold })
    }).then(async response => {
      if (!response.ok) throw new Error(`Inference request failed (${response.status})`);
      return response.json();
    });

    if (requestId !== microscopyRequestId) return;

    loader.classList.add('hidden');

    if (res.status === 'SUCCESS') {
      currentDetections = res.detections || [];
      const counts = res.counts || {};
      document.getElementById('vision-status').textContent = 'Inference completed successfully.';

      document.getElementById('count-class0').textContent = counts.predicted_individual_cells_class0 ?? '--';
      document.getElementById('count-class1').textContent = counts.predicted_clusters_debris_class1 ?? '--';
      document.getElementById('gt-cells').textContent = `${counts.ground_truth_cells_class0} cells`;

      const err = counts.counting_error;
      const errStr = err > 0 ? `+${err}` : `${err}`;
      document.getElementById('count-error').textContent = `${errStr} cells`;
      document.getElementById('vision-density-calc').textContent = counts.vision_derived_density_million_ml == null ? 'Calibration unavailable' : counts.vision_derived_density_million_ml;

      // Update Morphological Biovolume & Dry Mass Card
      const bio = res.biomass_estimation;
      if (bio) {
        document.getElementById('stat-total-biovolume').textContent = Number(bio.total_biovolume_um3).toLocaleString();
        document.getElementById('stat-mean-diameter').textContent = bio.mean_cell_diameter_um;
        document.getElementById('stat-mean-volume').textContent = bio.mean_cell_volume_um3;
        document.getElementById('stat-total-dry-mass').textContent = Number(bio.total_estimated_cell_dry_mass_pg).toLocaleString();
      }

      // Update Uncertainty Quantification & Active Learning Triage
      const tri = res.uncertainty_and_triage;
      if (tri) {
        const triagePill = document.getElementById('triage-status-pill');
        const triageAdvisory = document.getElementById('triage-advisory-text');
        if (triagePill) {
          const score = tri.optical_quality_score;
          const levelClass = score >= 80 ? 'verified' : (score >= 60 ? 'moderate' : 'flagged');
          triagePill.textContent = `${tri.triage_decision.replace(/_/g, ' ')} (${score}%)`;
          triagePill.className = `triage-pill ${levelClass}`;
        }
        document.getElementById('triage-mean-conf').textContent = `${(tri.mean_detection_confidence * 100).toFixed(0)}%`;
        document.getElementById('triage-borderline-count').textContent = `${tri.borderline_detection_count} cells`;
        document.getElementById('triage-clump-pct').textContent = `${tri.cluster_interference_pct}%`;
        if (triageAdvisory) {
          triageAdvisory.textContent = tri.triage_advisory;
        }
      }

      // Load image into canvas
      const imageRequestId = requestId;
      currentImageObj = new Image();
      currentImageObj.onload = () => {
        if (imageRequestId !== microscopyRequestId) return;
        redrawCanvas();
        document.getElementById('microscopy-empty-state')?.classList.add('hidden');
      };
      currentImageObj.src = res.image_data_uri;
    } else {
      document.getElementById('vision-status').textContent = res.error || 'Microscopy inference is unavailable.';
      showMicroscopyEmptyState('Inference unavailable. The source frame is still shown for inspection.');
    }
  } catch (err) {
    loader.classList.add('hidden');
    if (requestId !== microscopyRequestId) return;
    document.getElementById('vision-status').textContent = 'Microscopy inference failed. The source frame is still shown.';
    showMicroscopyEmptyState('Inference failed. The source frame is still shown for inspection.');
    console.error('Vision detection error:', err);
  }
}

function clearVisionResults(message) {
  currentDetections = [];
  if (currentImageObj) redrawCanvas();
  document.getElementById('vision-status').textContent = message;
  document.getElementById('count-class0').textContent = '--';
  document.getElementById('count-class1').textContent = '--';
  document.getElementById('gt-cells').textContent = '--';
  document.getElementById('count-error').textContent = '--';
  document.getElementById('vision-density-calc').textContent = 'Calibration unavailable';
  document.getElementById('stat-total-biovolume').textContent = '--';
  document.getElementById('stat-mean-diameter').textContent = '--';
  document.getElementById('stat-mean-volume').textContent = '--';
  document.getElementById('stat-total-dry-mass').textContent = '--';
  document.getElementById('triage-mean-conf').textContent = '--';
  document.getElementById('triage-borderline-count').textContent = '--';
  document.getElementById('triage-clump-pct').textContent = '--';
  document.getElementById('triage-advisory-text').textContent = 'Waiting for microscopy inference…';
}

function redrawCanvas() {
  const canvas = document.getElementById('microscopy-canvas');
  if (!canvas || !currentImageObj) return;

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw underlying microscopy frame
  ctx.drawImage(currentImageObj, 0, 0, canvas.width, canvas.height);

  const showBoxes = document.getElementById('toggle-boxes').checked;
  const showLabels = document.getElementById('toggle-labels').checked;

  if (!showBoxes) return;

  currentDetections.forEach(det => {
    const [xc, yc, w, h] = det.box;
    const isCell = det.class_id === 0;

    // Convert normalized coordinates to 640x640 canvas
    const boxW = w * canvas.width;
    const boxH = h * canvas.height;
    const boxX = (xc * canvas.width) - (boxW / 2);
    const boxY = (yc * canvas.height) - (boxH / 2);

    ctx.lineWidth = isCell ? 2.5 : 3;
    ctx.strokeStyle = isCell ? '#3fb950' : '#e3b341';
    ctx.fillStyle = isCell ? 'rgba(46, 160, 67, 0.15)' : 'rgba(210, 153, 34, 0.2)';

    ctx.strokeRect(boxX, boxY, boxW, boxH);
    ctx.fillRect(boxX, boxY, boxW, boxH);

    if (showLabels) {
      const labelText = isCell
        ? `cell ${Math.round(det.confidence * 100)}%`
        : `cluster ${Math.round(det.confidence * 100)}%`;

      ctx.font = 'bold 12px "JetBrains Mono", monospace';
      const textWidth = ctx.measureText(labelText).width;

      ctx.fillStyle = isCell ? '#1f6feb' : '#b08800';
      ctx.fillRect(boxX, boxY > 15 ? boxY - 14 : boxY, textWidth + 6, 14);

      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, boxX + 3, boxY > 15 ? boxY - 3 : boxY + 10);
    }
  });
}

/**
 * 4. Alerts and Operational Recommendations
 */
async function fetchAlertsAndRecommendations() {
  try {
    const [resAlerts, resRecs] = await Promise.all([
      fetch('/api/alerts').then(parseJsonResponse),
      fetch('/api/recommendations').then(parseJsonResponse)
    ]);

    // Populate alerts
    const alertsContainer = document.getElementById('alerts-list-container');
    if (resAlerts.alerts && resAlerts.alerts.length > 0) {
      alertsContainer.innerHTML = '';
      resAlerts.alerts.forEach((alert, idx) => {
        const item = document.createElement('div');
        const alertClass = alert.severity === 'WARNING' ? 'alert-warning' : (alert.severity === 'INFO' ? 'alert-info' : 'alert-normal');
        const icon = alert.severity === 'WARNING' ? '⚠' : (alert.severity === 'INFO' ? 'ℹ' : '✓');

        item.className = `alert-item ${alertClass}`;
        item.id = `alert-item-${idx}`;
        item.innerHTML = `
          <div class="alert-icon">${icon}</div>
          <div class="alert-content">
            <span class="alert-category">${alert.category}</span>
            <p class="alert-msg">${alert.message}</p>
          </div>
        `;
        alertsContainer.appendChild(item);
      });
    }

    // Populate recommendations
    const h = resRecs.harvest_decision;
    if (h) {
      document.getElementById('rec-readiness-badge').textContent = `${h.readiness_percentage}% READY`;
      document.getElementById('rec-harvest-desc').textContent = h.rationale;
      document.getElementById('harvest-progress-bar').style.width = `${h.readiness_percentage}%`;
    }

    const s = resRecs.sparging_recommendation;
    if (s) {
      document.getElementById('rec-co2-sparging').textContent = `${s.duty_cycle} (${s.co2_enrichment_ratio})`;
    }
  } catch (err) {
    document.getElementById('alerts-list-container').textContent = 'Alerts could not be loaded. Refresh to try again.';
    document.getElementById('rec-harvest-desc').textContent = 'Recommendations are temporarily unavailable.';
    console.warn('Alerts fetch warning:', err);
  }
}

/**
 * 5. Canvas-based Multi-Chart Renderer for 14-Day Trajectories
 */
async function renderHistoricalCharts() {
  try {
    if (!historicalChartData) {
      const response = await fetch('/api/history');
      if (!response.ok) throw new Error(`History request failed (${response.status})`);
      const res = await response.json();
      historicalChartData = { metrics: res.metrics, days: res.days || [] };
    }
    if (currentDashboardView !== 'trends') return;
    const { metrics: m, days } = historicalChartData;
    if (!m) return;

    drawResponsiveLineChart('chart-density-canvas', m.cell_density_million_ml, '#16845b', 'million cells/mL', days);
    drawResponsiveLineChart('chart-co2-canvas', m.daily_co2_kg, '#2878b8', 'kg/day', days);
    drawResponsiveLineChart('chart-ph-canvas', m.ph, '#8066b3', 'pH', days);
    drawResponsiveLineChart('chart-temp-canvas', m.temperature_c, '#c26c2e', '°C', days);
  } catch (err) {
    console.warn('Charts render warning:', err);
    document.querySelectorAll('.chart-empty-state').forEach(state => {
      state.textContent = 'Trend data could not be loaded. Reopen this section to retry.';
      state.classList.remove('hidden');
    });
  }
}

function drawResponsiveLineChart(canvasId, dataPoints, strokeColor, unit, days = []) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const points = Array.isArray(dataPoints) ? dataPoints.map(Number).filter(Number.isFinite) : [];
  const empty = canvas.parentElement.querySelector('.chart-empty-state');
  if (points.length < 2) {
    if (empty) { empty.textContent = 'Not enough trend data to draw this chart.'; empty.classList.remove('hidden'); }
    return;
  }
  if (empty) empty.classList.add('hidden');

  const width = Math.max(300, Math.round(canvas.clientWidth || 540));
  const height = 230;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.width = '100%';
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const left = 62, right = 18, top = 24, bottom = 34;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const rawMin = Math.min(...points), rawMax = Math.max(...points);
  const spread = rawMax - rawMin || Math.max(Math.abs(rawMax) * 0.08, 0.5);
  const min = rawMin - spread * 0.12, max = rawMax + spread * 0.12;
  const range = max - min;
  const xFor = index => left + (index / (points.length - 1)) * plotWidth;
  const yFor = value => top + plotHeight - ((value - min) / range) * plotHeight;

  ctx.font = '11px system-ui, sans-serif';
  ctx.textAlign = 'right';
  ctx.lineWidth = 1;
  for (let index = 0; index <= 4; index++) {
    const y = top + (plotHeight / 4) * index;
    const label = max - (range / 4) * index;
    ctx.strokeStyle = '#e7ede9';
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(width - right, y); ctx.stroke();
    ctx.fillStyle = '#738078'; ctx.fillText(label.toFixed(1), left - 9, y + 4);
  }

  ctx.beginPath();
  points.forEach((value, index) => index ? ctx.lineTo(xFor(index), yFor(value)) : ctx.moveTo(xFor(index), yFor(value)));
  ctx.lineTo(xFor(points.length - 1), top + plotHeight);
  ctx.lineTo(xFor(0), top + plotHeight);
  ctx.closePath();
  const clean = strokeColor.replace('#', '');
  const red = parseInt(clean.slice(0, 2), 16), green = parseInt(clean.slice(2, 4), 16), blue = parseInt(clean.slice(4, 6), 16);
  const gradient = ctx.createLinearGradient(0, top, 0, top + plotHeight);
  gradient.addColorStop(0, `rgba(${red}, ${green}, ${blue}, 0.16)`);
  gradient.addColorStop(1, `rgba(${red}, ${green}, ${blue}, 0.015)`);
  ctx.fillStyle = gradient; ctx.fill();

  ctx.beginPath();
  points.forEach((value, index) => index ? ctx.lineTo(xFor(index), yFor(value)) : ctx.moveTo(xFor(index), yFor(value)));
  ctx.strokeStyle = strokeColor; ctx.lineWidth = 2.5; ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.stroke();
  points.forEach((value, index) => {
    ctx.beginPath(); ctx.arc(xFor(index), yFor(value), index === points.length - 1 ? 4 : 2.5, 0, Math.PI * 2);
    ctx.fillStyle = '#fff'; ctx.fill(); ctx.strokeStyle = strokeColor; ctx.lineWidth = 1.5; ctx.stroke();
  });

  ctx.textAlign = 'left'; ctx.fillStyle = '#738078';
  ctx.fillText(`Day ${days[0] ?? 1}`, left, height - 9);
  ctx.textAlign = 'right'; ctx.fillText(`Day ${days[points.length - 1] ?? points.length}`, width - right, height - 9);
  ctx.textAlign = 'left'; ctx.fillStyle = strokeColor; ctx.font = '600 11px system-ui, sans-serif';
  ctx.fillText(`${points[points.length - 1].toFixed(1)} ${unit}`, left, 14);
}

window.addEventListener('resize', () => {
  if (currentDashboardView === 'trends' && historicalChartData) renderHistoricalCharts();
});

function drawSmoothLineChart(canvasId, dataPoints, strokeColor, unit) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const paddingLeft = 35;
  const paddingRight = 15;
  const paddingTop = 25;
  const paddingBottom = 25;

  const plotW = w - paddingLeft - paddingRight;
  const plotH = h - paddingTop - paddingBottom;

  const minVal = Math.min(...dataPoints);
  const maxVal = Math.max(...dataPoints);
  const range = maxVal - minVal || 1.0;

  // Draw grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = paddingTop + (plotH / 3) * i;
    ctx.beginPath();
    ctx.moveTo(paddingLeft, y);
    ctx.lineTo(w - paddingRight, y);
    ctx.stroke();

    const labelVal = (maxVal - (range / 3) * i).toFixed(1);
    ctx.fillStyle = '#6e7681';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillText(labelVal, 4, y + 3);
  }

  // Draw filled gradient area
  ctx.beginPath();
  const stepX = plotW / (dataPoints.length - 1);

  dataPoints.forEach((val, idx) => {
    const x = paddingLeft + idx * stepX;
    const y = paddingTop + plotH - ((val - minVal) / range) * plotH;
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.lineTo(paddingLeft + (dataPoints.length - 1) * stepX, paddingTop + plotH);
  ctx.lineTo(paddingLeft, paddingTop + plotH);
  ctx.closePath();


  const grad = ctx.createLinearGradient(0, paddingTop, 0, h - paddingBottom);

  function hexToRgba(hex, alpha) {
    const cleanHex = hex.replace('#', '');
    const r = parseInt(cleanHex.slice(0, 2), 16);
    const g = parseInt(cleanHex.slice(2, 4), 16);
    const b = parseInt(cleanHex.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  grad.addColorStop(0, hexToRgba(strokeColor, 0.35));
  grad.addColorStop(1, 'rgba(0, 0, 0, 0)');

  ctx.fillStyle = grad;
  ctx.fill();

  // Draw main stroke line
  ctx.beginPath();
  dataPoints.forEach((val, idx) => {
    const x = paddingLeft + idx * stepX;
    const y = paddingTop + plotH - ((val - minVal) / range) * plotH;
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // Draw points
  dataPoints.forEach((val, idx) => {
    const x = paddingLeft + idx * stepX;
    const y = paddingTop + plotH - ((val - minVal) / range) * plotH;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
}

/**
 * 6. Autonomous Cyber-Physical Actuator Hub
 */
function initActuatorHub() {
  const btnActuate = document.getElementById('btn-trigger-actuation');
  if (btnActuate) {
    btnActuate.addEventListener('click', triggerClosedLoopActuation);
  }
}

async function triggerClosedLoopActuation() {
  const ph = parseFloat(document.getElementById('input-ph')?.value) || 7.42;
  const density = parseFloat(document.getElementById('input-density')?.value) || 18.4;
  const light = parseFloat(document.getElementById('input-light')?.value) || 240.0;
  const co2 = parseFloat(document.getElementById('input-co2')?.value) || 412.0;

  try {
    const res = await fetch('/api/automation/actuate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        current_ph: ph,
        current_density: density,
        current_light: light,
        co2_ppm: co2,
        mode: 'AUTO_CLOSED_LOOP'
      })
    }).then(parseJsonResponse);

    const acts = res.actuators;
    if (acts) {
      // 1. Sparging Valve
      const sp = acts.co2_sparging_valve;
      const pillSp = document.getElementById('pill-co2-valve');
      if (pillSp && sp) {
        pillSp.textContent = sp.valve_state === 'PULSING_OPEN' ? `PULSING (${sp.pulse_duration_seconds}s)` : sp.valve_state;
        pillSp.className = 'act-pill ' + (sp.valve_state === 'PULSING_OPEN' ? 'pulsing' : (sp.valve_state === 'EMERGENCY_LOCKOUT' ? 'pulsing' : 'standby'));
        document.getElementById('val-co2-pulse').textContent = `${sp.pulse_duration_seconds}s`;
        document.getElementById('val-co2-target-ph').textContent = sp.target_stabilized_ph;
        document.getElementById('note-co2-action').textContent = sp.action_summary;
      }

      // 2. LED Driver
      const led = acts.adaptive_led_lighting;
      const pillLed = document.getElementById('pill-led-driver');
      if (pillLed && led) {
        pillLed.textContent = `${led.driver_output_power_pct}% POWER`;
        document.getElementById('val-led-ppfd').textContent = `${led.target_ppfd_umol} μmol`;
        document.getElementById('val-led-penetration').textContent = `${led.calculated_interior_core_penetration_pct}%`;
        document.getElementById('note-led-spectrum').textContent = led.optimized_photonic_spectrum;
      }

      // 3. Harvest Valve
      const harv = acts.automated_harvest_valves;
      const pillHarv = document.getElementById('pill-harvest-valve');
      if (pillHarv && harv) {
        pillHarv.textContent = harv.cycle_status === 'CYCLE_TRIGGERED' ? 'DRAIN/REFILL' : 'STANDBY';
        pillHarv.className = 'act-pill ' + (harv.cycle_status === 'CYCLE_TRIGGERED' ? 'active' : 'standby');
        document.getElementById('val-harvest-vol').textContent = `${harv.drain_volume_liters} L`;
        document.getElementById('val-harvest-reset').textContent = `${harv.predicted_reset_density_million_ml}M`;
        document.getElementById('note-harvest-desc').textContent = harv.action_summary;
      }
    }
  } catch (err) {
    document.getElementById('note-co2-action').textContent = 'Simulation unavailable. Check the local API and try again.';
    console.error('Actuation trigger error:', err);
  }
}

/**
 * 7. Autonomous Mobile Liquid Tree (AMR) Corporate Dispatch
 */
let selectedAMRZone = 'boardroom_a';

function initMobileAMR() {
  const zoneCards = document.querySelectorAll('.zone-card');
  zoneCards.forEach(card => {
    card.addEventListener('click', () => {
      zoneCards.forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      selectedAMRZone = card.getAttribute('data-zone');
    });
  });

  const btnDispatch = document.getElementById('btn-dispatch-amr');
  if (btnDispatch) {
    btnDispatch.addEventListener('click', dispatchMobileAMRPod);
  }
}

async function dispatchMobileAMRPod() {
  const progText = document.getElementById('amr-scrub-pct');
  const progBar = document.getElementById('amr-progress-bar');
  const pillStatus = document.getElementById('amr-mission-pill');

  if (progText) progText.textContent = 'Navigating AMR through corridor (LiDAR SLAM)…';
  if (progBar) progBar.style.width = '35%';
  if (pillStatus) pillStatus.textContent = 'IN TRANSIT';

  try {
    const res = await fetch('/api/mobile/dispatch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_zone: selectedAMRZone })
    }).then(parseJsonResponse);

    const m = res.mission;
    const rob = res.robot;

    if (m) {
      document.getElementById('amr-target-zone').textContent = m.dispatch_zone;
      document.getElementById('amr-battery-pct').textContent = `${rob.battery_level_pct}%`;
      document.getElementById('amr-transit-eta').textContent = `${m.transit_time_minutes} min`;
      document.getElementById('amr-scrub-time').textContent = `${m.estimated_scrub_time_minutes} min`;
      document.getElementById('amr-target-co2').textContent = `${m.initial_co2_ppm} → ${m.target_co2_ppm} ppm`;
      document.getElementById('amr-o2-produced').textContent = `+${m.estimated_pure_o2_released_grams} g O₂`;
      document.getElementById('amr-advisory-text').textContent = m.dispatch_advisory;

      setTimeout(() => {
        if (selectedAMRZone === 'docking_station') {
          if (progText) progText.textContent = 'Docked at Mother Base Station (Charging / Flushed)';
          if (progBar) progBar.style.width = '100%';
          if (pillStatus) pillStatus.textContent = 'DOCKED & CHARGING';
        } else {
          if (progText) progText.textContent = 'Docked in room — Monod Micro-Sparging Active';
          if (progBar) progBar.style.width = '75%';
          if (pillStatus) pillStatus.textContent = 'ACTIVE SCRUBBING';
        }
      }, 700);
    }
  } catch (err) {
    document.getElementById('amr-advisory-text').textContent = 'Dispatch simulation unavailable. Check the local API and try again.';
    console.error('AMR dispatch error:', err);
  }
}
