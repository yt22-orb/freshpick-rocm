const screens = [...document.querySelectorAll('.screen')];
const navButtons = [...document.querySelectorAll('.nav-button')];
const endpointInput = document.querySelector('#endpoint-input');
const settingsDialog = document.querySelector('#settings-dialog');
const toast = document.querySelector('#toast');
const cameraVideo = document.querySelector('#camera-video');
const cameraFrame = document.querySelector('#camera-frame');
const scanNotice = document.querySelector('#scan-notice');
const galleryInput = document.querySelector('#gallery-input');

let cameraStream = null;
let cameraGeneration = 0;
let activeFile = null;
let activeImageUrl = null;
let requestController = null;
let lastResult = null;
let toastTimer = null;
let cameraFacing = 'environment';

const defaultEndpoint = new URL('/predict', window.location.href).href;

const stageLabels = { fresh: 'Fresh', semi_fresh: 'Semi-fresh', rotten: 'Rotten' };
const produceLabels = {
  banana: 'Banana', bittermelon: 'Bitter melon', cucumber: 'Cucumber',
  eggplant: 'Eggplant', orange: 'Orange', papaya: 'Papaya',
  pineapple: 'Pineapple', tomato: 'Tomato',
};

function showToast(message) {
  toast.textContent = message;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 3200);
}

function stopCamera() {
  cameraGeneration += 1;
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  cameraFrame.classList.remove('live');
  cameraVideo.srcObject = null;
}

function showScreen(name) {
  if (name !== 'scan') stopCamera();
  screens.forEach((screen) => screen.classList.toggle('active', screen.id === `${name}-screen`));
  navButtons.forEach((button) => button.classList.toggle('active', button.dataset.screen === name));
  document.querySelector('.bottom-nav').hidden = !['home', 'history'].includes(name);
  if (name === 'inspector') renderInspector('prediction');
  if (name === 'history' || name === 'home') renderHistory();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function useFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    showToast('Choose an image file to scan.');
    return;
  }
  if (activeImageUrl && activeImageUrl !== lastResult?.imageUrl) URL.revokeObjectURL(activeImageUrl);
  activeFile = file;
  activeImageUrl = URL.createObjectURL(file);
  document.querySelector('#evaluating-preview').src = activeImageUrl;
  submitImage(file);
}

async function openCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    scanNotice.textContent = 'Camera access is not available here. Choose an image from your gallery instead.';
    return;
  }
  stopCamera();
  const generation = cameraGeneration;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: cameraFacing }, audio: false,
    });
    if (generation !== cameraGeneration) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    cameraStream = stream;
    cameraVideo.srcObject = cameraStream;
    cameraFrame.classList.add('live');
    scanNotice.textContent = 'Center one supported fruit or vegetable in the frame, then capture.';
  } catch (error) {
    if (generation !== cameraGeneration) return;
    scanNotice.textContent = error.name === 'NotAllowedError'
      ? 'Camera permission was blocked. Allow camera access or choose an image from your gallery.'
      : 'Could not open the camera. Choose an image from your gallery.';
  }
}

function capturePhoto() {
  if (!cameraStream || !cameraVideo.videoWidth) {
    showToast('Start the camera or choose an image from your gallery.');
    return;
  }
  const canvas = document.querySelector('#capture-canvas');
  canvas.width = cameraVideo.videoWidth;
  canvas.height = cameraVideo.videoHeight;
  canvas.getContext('2d').drawImage(cameraVideo, 0, 0);
  canvas.toBlob((blob) => {
    if (blob) useFile(new File([blob], 'freshpick-capture.jpg', { type: 'image/jpeg' }));
    else showToast('The photo could not be captured. Please try again.');
  }, 'image/jpeg', 0.92);
}

function setProgress(step) {
  const items = [
    document.querySelector('#progress-received'),
    document.querySelector('#progress-sending'),
    document.querySelector('#progress-model'),
  ];
  items.forEach((item, index) => {
    item.classList.toggle('done', index < step);
    item.classList.toggle('current', index === step);
    item.querySelector('.progress-icon').textContent = index < step ? '✓' : '';
  });
}

async function submitImage(file) {
  const endpoint = localStorage.getItem('freshpick-api-url')?.trim() || defaultEndpoint;
  showScreen('evaluating');
  setProgress(1);
  requestController?.abort();
  const controller = new AbortController();
  requestController = controller;
  const form = new FormData();
  form.append('image', file, file.name || 'freshpick-image');
  try {
    const pendingResponse = fetch(endpoint, {
      method: 'POST', body: form, signal: controller.signal,
    });
    setProgress(2);
    const response = await pendingResponse;
    if (!response.ok) {
      const failure = await response.json().catch(() => ({}));
      throw new Error(failure.detail || `The model service returned ${response.status}.`);
    }
    const payload = await response.json();
    if (controller.signal.aborted) return;
    const normalized = normalizeResult(payload);
    if (!['ok', 'uncertain'].includes(normalized.status)) throw new Error(payload.message || 'The model could not assess this image.');
    if (lastResult?.imageUrl && lastResult.imageUrl !== activeImageUrl) URL.revokeObjectURL(lastResult.imageUrl);
    lastResult = { ...normalized, capturedAt: new Date().toISOString() };
    saveHistory(lastResult);
    renderReport(lastResult);
    requestController = null;
    showScreen('report');
  } catch (error) {
    if (requestController === controller) requestController = null;
    if (error.name === 'AbortError') return;
    showScreen('scan');
    const reason = error instanceof TypeError
      ? 'Could not reach the model service. Check the endpoint and try again.'
      : error.message;
    scanNotice.textContent = `${reason} Your image was not saved.`;
  }
}

function normalizeResult(payload) {
  if (!payload || !['ok', 'uncertain'].includes(payload.status)) {
    return { status: payload?.status || 'error', message: payload?.message };
  }
  const classProbabilities = payload.class_probabilities;
  if (!classProbabilities || typeof classProbabilities !== 'object') {
    throw new Error('The model response is missing class probabilities.');
  }
  const rows = Object.entries(classProbabilities).map(([key, value]) => ({
    key, value: Number(value), ...parseClassName(key),
  }));
  if (rows.length !== 24) throw new Error('The model response must include all 24 class probabilities.');
  if (rows.some((row) => !Number.isFinite(row.value) || row.value < 0)) {
    throw new Error('The model returned invalid class probabilities.');
  }
  const supported = new Set(Object.keys(produceLabels));
  if (rows.some((row) => !row.produce || !supported.has(row.produce) || !row.freshness)) {
    throw new Error('The model returned an unsupported class label.');
  }
  if (new Set(rows.map((row) => `${row.freshness}_${row.produce}`)).size !== 24) {
    throw new Error('The model returned duplicate class labels.');
  }
  const winner = rows.reduce((best, row) => row.value > best.value ? row : best, rows[0]);
  if (!winner?.produce || !winner?.freshness) throw new Error('The model returned an unsupported class label.');
  const freshnessProbabilities = { fresh: 0, semi_fresh: 0, rotten: 0 };
  const produceProbabilities = Object.fromEntries(Object.keys(produceLabels).map((key) => [key, 0]));
  rows.forEach((row) => {
    if (row.freshness in freshnessProbabilities) freshnessProbabilities[row.freshness] += row.value;
    if (row.produce in produceProbabilities) produceProbabilities[row.produce] += row.value;
  });
  const total = Object.values(freshnessProbabilities).reduce((sum, value) => sum + value, 0);
  if (!(total > 0)) throw new Error('The model returned no usable freshness probabilities.');
  Object.keys(freshnessProbabilities).forEach((key) => { freshnessProbabilities[key] /= total; });
  Object.keys(produceProbabilities).forEach((key) => { produceProbabilities[key] /= total; });
  const predictedFreshness = Object.entries(freshnessProbabilities)
    .reduce((best, item) => item[1] > best[1] ? item : best)[0];
  const predictedProduce = Object.entries(produceProbabilities)
    .reduce((best, item) => item[1] > best[1] ? item : best)[0];
  const freshnessConfidence = freshnessProbabilities[predictedFreshness];
  const produceConfidence = produceProbabilities[predictedProduce];
  const classConfidence = winner.value;
  if (![freshnessConfidence, produceConfidence, classConfidence].every((value) => Number.isFinite(value) && value >= 0 && value <= 1)) {
    throw new Error('The model returned an invalid confidence score.');
  }
  return {
    status: payload.status, produce: predictedProduce, freshness: predictedFreshness,
    classConfidence, produceConfidence, freshnessConfidence,
    produceProbabilities, freshnessProbabilities,
    classProbabilities, modelVersion: payload.model_version || 'Unspecified',
    imageName: activeFile?.name || 'Captured image', imageUrl: activeImageUrl,
  };
}

function parseClassName(name) {
  const match = /^(fresh|rotten|semi_fresh)_(.+)$/.exec(name);
  return match ? { freshness: match[1], produce: match[2].replace(/[ _]/g, '').toLowerCase() } : {};
}

function renderReport(result) {
  const stage = result.status === 'uncertain' ? 'Uncertain' : (stageLabels[result.freshness] || 'Uncertain');
  const produce = produceLabels[result.produce] || titleCase(result.produce);
  const uncertain = result.status === 'uncertain';
  const confidence = clamp(uncertain ? result.produceConfidence : result.freshnessConfidence);
  const reportFruit = document.querySelector('#report-fruit');
  reportFruit.src = result.imageUrl || 'assets/fruit.svg';
  reportFruit.alt = result.imageUrl ? 'Scanned produce' : '';
  reportFruit.closest('.result-fruit').classList.toggle('has-photo', Boolean(result.imageUrl));
  const reportStatus = document.querySelector('#report-status');
  reportStatus.classList.remove('fresh', 'semi_fresh', 'rotten', 'uncertain');
  reportStatus.classList.add(uncertain ? 'uncertain' : result.freshness);
  reportStatus.querySelector('span:first-child').textContent = uncertain ? '?' : '✓';
  reportStatus.querySelector('span:last-child').textContent = stage;
  document.querySelector('#report-produce').textContent = uncertain ? 'Could not classify clearly' : produce;
  document.querySelector('#report-time').textContent = `Scanned ${new Date(result.capturedAt).toLocaleString()}`;
  document.querySelector('#confidence-label').textContent = uncertain ? 'Top produce score' : 'Freshness confidence';
  document.querySelector('#report-confidence').textContent = `${(confidence * 100).toFixed(1)}%`;
  document.querySelector('#confidence-fill').style.width = `${confidence * 100}%`;
  document.querySelector('#guidance-copy').textContent = guidance(result.status === 'uncertain' ? 'uncertain' : result.freshness);
}

function guidance(stage) {
  if (stage === 'uncertain') return 'The model did not return a confident classification. Try a clearer photo with one supported item centered in the frame.';
  if (stage === 'fresh') return 'Visible condition appears consistent with the Fresh class. Check for damage and consider storage before purchasing.';
  if (stage === 'semi_fresh') return 'Visible condition appears consistent with the Semi-fresh class. Check the item closely and consider storage before purchasing.';
  if (stage === 'rotten') return 'Visible condition appears consistent with the Rotten class. This image assessment cannot confirm safety or internal quality.';
  return 'The model could not make a reliable classification. Inspect the item directly and try another clear image.';
}

function renderInspector(tab) {
  document.querySelectorAll('[data-inspector-tab]').forEach((button) => {
    const selected = button.dataset.inspectorTab === tab;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-selected', String(selected));
  });
  const content = document.querySelector('#inspector-content');
  if (!lastResult) {
    content.innerHTML = '<div class="unavailable"><strong>No model result yet</strong>Run a scan through a connected model endpoint to see assessment details.</div>';
    return;
  }
  if (tab === 'features') {
    content.innerHTML = '<div class="unavailable"><strong>Feature diagnostics unavailable</strong>The current classifier predicts from the whole image and does not expose validated color, texture, or shape explanations.</div>';
    return;
  }
  if (tab === 'model') {
    const label = lastResult.status === 'uncertain' ? 'Top candidate' : 'Predicted produce';
    content.innerHTML = `<div class="detail-row"><div><strong>Model version</strong><span>Reported by inference service</span></div><span>${escapeHtml(lastResult.modelVersion)}</span></div><div class="detail-row"><div><strong>${label}</strong><span>Marginal produce result · ${(clamp(lastResult.produceConfidence) * 100).toFixed(1)}%</span></div><span>${escapeHtml(produceLabels[lastResult.produce] || titleCase(lastResult.produce))}</span></div><div class="unavailable"><strong>Scope</strong>Eight produce types and three inherited freshness labels. Unsupported items are rejected only by checkpoints calibrated with out-of-distribution images.</div>`;
    return;
  }
  const imageName = escapeHtml(lastResult.imageName || 'Analyzed image');
  const bars = Object.entries(lastResult.freshnessProbabilities).map(([key, value]) => {
    const amount = clamp(Number(value));
    return `<div class="prob-row"><div class="prob-head"><span>${stageLabels[key] || titleCase(key)}</span><strong>${(amount * 100).toFixed(1)}%</strong></div><div class="bar-track"><span class="bar-fill" style="width:${amount * 100}%"></span></div></div>`;
  }).join('');
  const classes = Object.entries(lastResult.classProbabilities).map(([key, value]) => ({ key, value: Number(value) }))
    .sort((a, b) => b.value - a.value).map(({ key, value }) => `<div class="prob-row"><div class="prob-head"><span>${escapeHtml(titleCase(key.replaceAll('_', ' ')))}</span><strong>${(clamp(value) * 100).toFixed(1)}%</strong></div><div class="bar-track"><span class="bar-fill" style="width:${clamp(value) * 100}%"></span></div></div>`).join('');
  const thumb = lastResult.imageUrl ? escapeHtml(lastResult.imageUrl) : 'assets/fruit.svg';
  content.innerHTML = `<div class="inspector-image-row"><div class="mini-fruit"><img src="${thumb}" alt="" /></div><div><strong>Analyzed image</strong><span>${imageName}</span></div></div><section class="probability-card"><h2>Freshness probabilities</h2>${bars}<p class="helper">Summed across the supported produce classes. These are model scores, not food-safety probabilities.</p></section><details class="all-classes"><summary>All 24 class probabilities</summary><div class="class-list">${classes}</div></details>`;
}

function saveHistory(result) {
  const history = readHistory();
  history.unshift({ status: result.status, produce: result.produce, freshness: result.freshness,
    confidence: result.status === 'uncertain' ? result.produceConfidence : result.freshnessConfidence,
    capturedAt: result.capturedAt });
  localStorage.setItem('freshpick-history', JSON.stringify(history.slice(0, 12)));
}

function readHistory() {
  try { const items = JSON.parse(localStorage.getItem('freshpick-history') || '[]'); return Array.isArray(items) ? items : []; }
  catch { return []; }
}

function renderHistory() {
  const history = readHistory();
  const target = document.querySelector('#history-list');
  const recent = document.querySelector('.home-history');
  recent.innerHTML = history.length
    ? `<div class="empty-mark" aria-hidden="true">${history[0].status === 'uncertain' ? '?' : '✓'}</div><div><strong>${history[0].status === 'uncertain' ? 'Uncertain result' : escapeHtml(produceLabels[history[0].produce] || titleCase(history[0].produce))}</strong><span>${history[0].status === 'uncertain' ? 'Top produce score' : (stageLabels[history[0].freshness] || 'Uncertain')} · ${(clamp(history[0].confidence) * 100).toFixed(1)}%</span></div><span class="small-arrow" aria-hidden="true">›</span>`
    : '<div class="empty-mark" aria-hidden="true">+</div><div><strong>No scans yet</strong><span>Your latest result will appear here.</span></div>';
  if (!history.length) {
    target.innerHTML = '<div class="history-empty"><strong>No scans yet</strong>Your scan history will appear here on this device.</div>';
    return;
  }
  target.innerHTML = history.map((item) => `<article class="history-item"><div><strong>${item.status === 'uncertain' ? 'Uncertain result' : escapeHtml(produceLabels[item.produce] || titleCase(item.produce))}</strong><span>${new Date(item.capturedAt).toLocaleString()}</span></div><strong>${item.status === 'uncertain' ? 'Uncertain' : (stageLabels[item.freshness] || 'Uncertain')} · ${(clamp(item.confidence) * 100).toFixed(1)}%</strong></article>`).join('');
}

function exportReport() {
  if (!lastResult) return showToast('There is no report to export yet.');
  const uncertain = lastResult.status === 'uncertain';
  const text = (uncertain ? [
    'FreshPick visible-condition report',
    'Status: Uncertain',
    `Top produce candidate: ${produceLabels[lastResult.produce] || titleCase(lastResult.produce)}`,
    `Top produce score: ${(clamp(lastResult.produceConfidence) * 100).toFixed(1)}%`,
    'Try another clear photo of one supported item.',
  ] : [
    'FreshPick visible-condition report',
    `Produce: ${produceLabels[lastResult.produce] || titleCase(lastResult.produce)}`,
    `Freshness: ${stageLabels[lastResult.freshness] || lastResult.freshness}`,
    `Freshness confidence: ${(clamp(lastResult.freshnessConfidence) * 100).toFixed(1)}%`,
  ]).concat([
    `Scanned: ${new Date(lastResult.capturedAt).toLocaleString()}`,
    'This image assessment cannot confirm internal quality, contamination, or food safety.',
  ]).join('\n');
  const url = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'freshpick-report.txt';
  link.click();
  URL.revokeObjectURL(url);
}

function titleCase(value) { return String(value || '').replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()); }
function clamp(value) { return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0; }
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[character]);
}

document.addEventListener('click', async (event) => {
  const action = event.target.closest('[data-action]')?.dataset.action;
  const screen = event.target.closest('[data-screen]')?.dataset.screen;
  const source = event.target.closest('[data-source]')?.dataset.source;
  const inspectorTab = event.target.closest('[data-inspector-tab]')?.dataset.inspectorTab;
  if (screen) return showScreen(screen);
  if (source) {
    document.querySelectorAll('[data-source]').forEach((button) => {
      const selected = button.dataset.source === source;
      button.classList.toggle('selected', selected);
      button.setAttribute('aria-selected', String(selected));
    });
    if (source === 'gallery') { stopCamera(); galleryInput.click(); }
    else openCamera();
    return;
  }
  if (inspectorTab) return renderInspector(inspectorTab);
  if (!action) return;
  if (action === 'scan') { showScreen('scan'); openCamera(); return; }
  if (action === 'back-home') { showScreen('home'); return; }
  if (action === 'report') { showScreen('report'); return; }
  if (action === 'inspector') { showScreen('inspector'); return; }
  if (action === 'select-gallery') { stopCamera(); galleryInput.click(); return; }
  if (action === 'capture') { capturePhoto(); return; }
  if (action === 'settings') {
    endpointInput.value = localStorage.getItem('freshpick-api-url') || defaultEndpoint;
    settingsDialog.showModal();
    return;
  }
  if (action === 'cancel-scan') {
    requestController?.abort();
    requestController = null;
    showScreen('scan');
    showToast('Analysis cancelled.');
    return;
  }
  if (action === 'switch-camera') {
    cameraFacing = cameraFacing === 'environment' ? 'user' : 'environment';
    openCamera();
    return;
  }
  if (action === 'flash') { showToast('Flash control is not available in this browser.'); return; }
  if (action === 'export') { exportReport(); }
});

galleryInput.addEventListener('change', (event) => {
  useFile(event.target.files?.[0]);
  event.target.value = '';
});

document.querySelector('#settings-form').addEventListener('submit', (event) => {
  if (event.submitter?.id !== 'save-endpoint') return;
  event.preventDefault();
  const value = endpointInput.value.trim();
  if (!value) {
    localStorage.removeItem('freshpick-api-url');
    settingsDialog.close();
    showToast('Using the built-in model endpoint.');
    return;
  }
  try {
    const url = new URL(value);
    if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Invalid protocol');
    localStorage.setItem('freshpick-api-url', url.toString());
    settingsDialog.close();
    showToast('Model endpoint saved on this device.');
  } catch {
    showToast('Enter a valid HTTP or HTTPS endpoint URL.');
  }
});

document.addEventListener('visibilitychange', () => { if (document.hidden) stopCamera(); });

renderHistory();
