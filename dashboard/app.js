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

function initApp() {
  fetchOverviewAndTelemetry();
  initPredictionEngine();
  initMicroscopyInspector();
  fetchAlertsAndRecommendations();
  renderHistoricalCharts();

  document.getElementById('btn-refresh-all').addEventListener('click', () => {
    fetchOverviewAndTelemetry();
    fetchAlertsAndRecommendations();
  });
}

/**
 * 1. Telemetry and Overview Fetching
 */
async function fetchOverviewAndTelemetry() {
  try {
    const [resStatus, resTelem] = await Promise.all([
      fetch('/api/status').then(r => r.json()),
      fetch('/api/telemetry').then(r => r.json())
    ]);

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

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        current_density: density,
        current_ph: ph,
        light: light,
        co2_ppm: co2
      })
    }).then(r => r.json());

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
  } catch (err) {
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
    executeCellDetection();
  });

  btnRunVision.addEventListener('click', executeCellDetection);
  toggleBoxes.addEventListener('change', redrawCanvas);
  toggleLabels.addEventListener('change', redrawCanvas);

  // Initial detection execution
  executeCellDetection();
}

async function executeCellDetection() {
  const sampleId = document.getElementById('select-microscopy-sample').value;
  const confThreshold = parseFloat(document.getElementById('slider-conf-thresh').value) || 0.25;
  const loader = document.getElementById('canvas-loader');

  loader.classList.remove('hidden');
  clearVisionResults('Running microscopy inference…');

  try {
    const res = await fetch('/api/vision/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_id: sampleId, conf_threshold: confThreshold })
    }).then(r => r.json());

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

      // Load image into canvas
      currentImageObj = new Image();
      currentImageObj.onload = () => {
        redrawCanvas();
      };
      currentImageObj.src = res.image_data_uri;
    } else {
      clearVisionResults(res.error || 'Microscopy inference is unavailable.');
    }
  } catch (err) {
    loader.classList.add('hidden');
    clearVisionResults('Microscopy inference is unavailable.');
    console.error('Vision detection error:', err);
  }
}

function clearVisionResults(message) {
  currentDetections = [];
  currentImageObj = null;
  const canvas = document.getElementById('microscopy-canvas');
  if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
  document.getElementById('vision-status').textContent = message;
  document.getElementById('count-class0').textContent = '--';
  document.getElementById('count-class1').textContent = '--';
  document.getElementById('gt-cells').textContent = '--';
  document.getElementById('count-error').textContent = '--';
  document.getElementById('vision-density-calc').textContent = 'Calibration unavailable';
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
      fetch('/api/alerts').then(r => r.json()),
      fetch('/api/recommendations').then(r => r.json())
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
    console.warn('Alerts fetch warning:', err);
  }
}

/**
 * 5. Canvas-based Multi-Chart Renderer for 14-Day Trajectories
 */
async function renderHistoricalCharts() {
  try {
    const res = await fetch('/api/history').then(r => r.json());
    const m = res.metrics;
    if (!m) return;

    drawSmoothLineChart('chart-density-canvas', m.cell_density_million_ml, '#58a6ff', '×10⁶/mL');
    drawSmoothLineChart('chart-co2-canvas', m.daily_co2_kg, '#3fb950', 'kg/day');
    drawSmoothLineChart('chart-ph-canvas', m.ph, '#d2a8ff', 'pH');
    drawSmoothLineChart('chart-temp-canvas', m.temperature_c, '#f0883e', '°C');
  } catch (err) {
    console.warn('Charts render warning:', err);
  }
}

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
