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
  storageController?.abort();
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
    document.querySelector('#storage-form').reset();
    document.querySelector('#storage-result').innerHTML = '<p>Complete the details above to see sourced storage guidance.</p>';
    lastResult = { ...normalized, capturedAt: new Date().toISOString() };
    document.querySelector('#produce-correction').hidden = true;
    document.querySelector('#confirmed-produce').disabled = true;
    document.querySelector('#confirmed-produce').required = false;
    document.querySelector('#photo-details-editor').open = false;
    populatePhotoDetails(lastResult.produce);
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
  if (payload.produce_probabilities) {
    const supplied = payload.produce_probabilities;
    if (Object.keys(supplied).length !== 8 || Object.keys(produceLabels).some((key) => !Number.isFinite(supplied[key]) || supplied[key] < 0 || supplied[key] > 1)) throw new Error('Invalid produce probabilities.');
    const suppliedTotal = Object.values(supplied).reduce((sum, value) => sum + value, 0);
    if (Math.abs(suppliedTotal - 1) > 0.01) throw new Error('Invalid produce probability total.');
    Object.keys(produceLabels).forEach((key) => { produceProbabilities[key] = supplied[key] / suppliedTotal; });
  }
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
    freshnessAvailable: payload.freshness_available !== false,
    freshnessModelProduce: payload.freshness_model_produce || null,
    photoObservations: payload.photo_observations || null,
    identifierVersion: payload.identifier_version || null,
    uncertaintyReason: payload.uncertainty_reason || null,
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
  const corrected = result.confirmedProduce && result.confirmedProduce !== result.produce;
  const freshnessAvailable = !uncertain && result.freshnessAvailable !== false && !corrected;
  const confidence = clamp(result.freshnessConfidence);
  document.querySelector('#produce-confidence-label').textContent = corrected ? `Model’s ${produce} score` : 'Produce confidence';
  document.querySelector('#produce-confidence').textContent = `${(clamp(result.produceConfidence) * 100).toFixed(1)}%`;
  document.querySelector('#produce-score-note').textContent = `Model candidate: ${produce}. ${result.confirmedProduce ? `User confirmed: ${produceLabels[result.confirmedProduce] || 'Other / unsure'}.` : 'Confirm or correct it below.'} A high score can still be wrong.`;
  const reportFruit = document.querySelector('#report-fruit');
  reportFruit.src = result.imageUrl || 'assets/fruit.svg';
  reportFruit.alt = result.imageUrl ? 'Scanned produce' : '';
  reportFruit.closest('.result-fruit').classList.toggle('has-photo', Boolean(result.imageUrl));
  const reportStatus = document.querySelector('#report-status');
  reportStatus.classList.remove('fresh', 'semi_fresh', 'rotten', 'uncertain');
  reportStatus.classList.add(!freshnessAvailable ? 'uncertain' : result.freshness);
  reportStatus.querySelector('span:first-child').textContent = !freshnessAvailable ? '?' : '✓';
  reportStatus.querySelector('span:last-child').textContent = freshnessAvailable ? stage : 'Freshness not verified';
  document.querySelector('#report-produce').textContent = result.confirmedProduce ? `${produceLabels[result.confirmedProduce] || 'Other / unsure'} · confirmed by you` : (uncertain ? 'Could not classify clearly' : produce);
  document.querySelector('#report-time').textContent = `Scanned ${new Date(result.capturedAt).toLocaleString()}`;
  document.querySelector('#confidence-label').textContent = 'Freshness confidence';
  document.querySelector('#report-confidence').textContent = freshnessAvailable ? `${(confidence * 100).toFixed(1)}%` : 'Unavailable';
  document.querySelector('#confidence-fill').style.width = `${freshnessAvailable ? confidence * 100 : 0}%`;
  const legacyProduce = produceLabels[result.freshnessModelProduce] || result.freshnessModelProduce;
  document.querySelector('#freshness-reason').textContent = freshnessAvailable
    ? 'Uncalibrated freshness score, not a food-safety probability.'
    : corrected ? 'The identification was corrected. The old freshness score does not apply to the corrected item.'
    : result.status === 'uncertain' ? 'The produce identification is uncertain. A reliable freshness assessment is unavailable.'
    : legacyProduce && result.freshnessModelProduce !== result.produce ? `The identifier sees ${produce}, but the freshness model sees ${legacyProduce}. Its freshness score cannot be applied reliably to ${produce}. Photo-based storage suggestions are separate.`
    : 'The freshness model could not provide a reliable assessment. Photo-based storage suggestions are separate.';
  document.querySelector('#identity-question').textContent = `Is this ${produce}?`;
  document.querySelector('#identity-yes-label').textContent = `Yes, this is ${produce}`;
  document.querySelector('#guidance-copy').textContent = freshnessAvailable ? guidance(result.freshness) : 'A verified freshness assessment is unavailable. Confirm the identification to use the separate photo-based storage suggestions. Ripe-looking appearance does not establish freshness or safety.';
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
    content.innerHTML = `<div class="detail-row"><div><strong>Freshness model</strong><span>Reported by inference service</span></div><span>${escapeHtml(lastResult.modelVersion)}</span></div><div class="detail-row"><div><strong>Identifier version</strong><span>Separate produce decision</span></div><span>${escapeHtml(lastResult.identifierVersion || '24-class baseline')}</span></div><div class="detail-row"><div><strong>${label}</strong><span>Marginal produce result · ${(clamp(lastResult.produceConfidence) * 100).toFixed(1)}%</span></div><span>${escapeHtml(produceLabels[lastResult.produce] || titleCase(lastResult.produce))}</span></div><div class="unavailable"><strong>Scope</strong>Experimental eight-produce identification and inherited freshness labels. Rejection was calibrated on a small image pilot and is not verified general unknown-item recognition. ${lastResult.freshnessAvailable === false ? 'The identity and freshness models disagree, or the identity check failed; freshness is unavailable.' : ''}</div>`;
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
  content.innerHTML = `<div class="inspector-image-row"><div class="mini-fruit"><img src="${thumb}" alt="" /></div><div><strong>Analyzed image</strong><span>${imageName}</span></div></div><section class="probability-card"><h2>Freshness probabilities</h2>${bars}<p class="helper">Raw summed scores from the 24-class freshness model. These are not a verified assessment when identity is uncertain or corrected, and are not food-safety probabilities.</p></section><details class="all-classes"><summary>All 24 class probabilities</summary><div class="class-list">${classes}</div></details>`;
}

function saveHistory(result) {
  const history = readHistory();
  const existing = history.findIndex((item) => item.capturedAt === result.capturedAt);
  if (existing >= 0) history.splice(existing, 1);
  history.unshift({ status: result.status, produce: result.produce, freshness: result.freshness,
    confidence: result.produceConfidence, produceConfidence: result.produceConfidence, freshnessConfidence: result.freshnessConfidence, freshnessAvailable: result.freshnessAvailable,
    photoObservations: result.photoObservations || null, selectedPhotoDetails: result.selectedPhotoDetails || null, photoDetailSource: result.photoDetailsEdited ? 'user_edited_photo' : 'photo_model',
    capturedAt: result.capturedAt, confirmedProduce: result.confirmedProduce || null, storageGuidance: result.storageGuidance || null });
  localStorage.setItem('freshpick-history', JSON.stringify(history.slice(0, 12)));
}

function readHistory() {
  try { const items = JSON.parse(localStorage.getItem('freshpick-history') || '[]'); return Array.isArray(items) ? items : []; }
  catch { return []; }
}

function historyIdentity(item) {
  const model = produceLabels[item.produce] || titleCase(item.produce);
  return item.confirmedProduce ? `${produceLabels[item.confirmedProduce] || 'Other / unsure'} (user confirmed; model: ${model})` : model;
}

function historyTitle(item) { return item.confirmedProduce ? historyIdentity(item) : (item.status === 'uncertain' ? 'Uncertain result' : historyIdentity(item)); }
function historyScoreLabel(item) { return item.produceConfidence === undefined ? 'Saved score' : 'Produce score'; }

function renderHistory() {
  const history = readHistory();
  const target = document.querySelector('#history-list');
  const recent = document.querySelector('.home-history');
  recent.innerHTML = history.length
    ? `<div class="empty-mark" aria-hidden="true">${history[0].status === 'uncertain' ? '?' : '✓'}</div><div><strong>${escapeHtml(historyTitle(history[0]))}</strong><span>${historyScoreLabel(history[0])} · ${(clamp(history[0].confidence) * 100).toFixed(1)}%</span></div><span class="small-arrow" aria-hidden="true">›</span>`
    : '<div class="empty-mark" aria-hidden="true">+</div><div><strong>No scans yet</strong><span>Your latest result will appear here.</span></div>';
  if (!history.length) {
    target.innerHTML = '<div class="history-empty"><strong>No scans yet</strong>Your scan history will appear here on this device.</div>';
    return;
  }
  target.innerHTML = history.map((item) => `<article class="history-item"><div><strong>${escapeHtml(historyTitle(item))}</strong><span>${new Date(item.capturedAt).toLocaleString()}</span></div><strong>${historyScoreLabel(item)} · ${(clamp(item.confidence) * 100).toFixed(1)}%</strong></article>`).join('');
}

function exportReport() {
  if (!lastResult) return showToast('There is no report to export yet.');
  const freshnessAvailable = lastResult.status !== 'uncertain' && lastResult.freshnessAvailable !== false && (!lastResult.confirmedProduce || lastResult.confirmedProduce === lastResult.produce);
  const text = [
    'FreshPick experimental image assessment',
    `Status: ${lastResult.status}`,
    `Model produce candidate: ${produceLabels[lastResult.produce] || lastResult.produce}`,
    `Produce confidence: ${(clamp(lastResult.produceConfidence) * 100).toFixed(1)}%`,
    `User-confirmed produce: ${lastResult.confirmedProduce ? (produceLabels[lastResult.confirmedProduce] || 'Other / unsure') : 'Not confirmed'}`,
    `Freshness: ${freshnessAvailable ? stageLabels[lastResult.freshness] : 'Unavailable; candidate is not a verified assessment for this produce'}`,
    ...(freshnessAvailable ? [`Freshness confidence: ${(clamp(lastResult.freshnessConfidence) * 100).toFixed(1)}%`] : []),
    ...(lastResult.storageGuidance ? ['Typical storage guidance: ' + lastResult.storageGuidance.window_label, ...lastResult.storageGuidance.suggestions, ...lastResult.storageGuidance.assumptions, lastResult.storageGuidance.disclaimer, ...lastResult.storageGuidance.sources.map((source) => source.title + ': ' + source.url + ' (reviewed ' + source.reviewed_on + ')')] : ['Storage guidance: Not requested']),
    ...(lastResult.selectedPhotoDetails ? [`Photo/detail source: ${lastResult.photoDetailsEdited ? 'edited by user' : 'model suggestion'}`, `Whole/cut detail: ${lastResult.selectedPhotoDetails.condition}`, `Ripeness appearance: ${lastResult.selectedPhotoDetails.ripeness}`] : []),
    `Model version: ${lastResult.modelVersion}`,
    `Identifier version: ${lastResult.identifierVersion || '24-class baseline'}`,
    `Scanned: ${new Date(lastResult.capturedAt).toLocaleString()}`,
    'This image assessment cannot confirm internal quality, contamination, food safety, or remaining shelf life.',
  ].join('\n');
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

let storageController = null;
function chosenProduce() {
  return document.querySelector('#produce-confirmed').checked ? lastResult?.produce : document.querySelector('#confirmed-produce').value;
}
function populatePhotoDetails(produce) {
  if (!lastResult) return;
  const proposals = lastResult.photoObservations?.by_produce?.[produce];
  const condition = proposals?.condition?.value;
  const ripeness = proposals?.ripeness?.value;
  document.querySelector('#storage-condition').value = ['whole','cut'].includes(condition) ? condition : 'unknown';
  document.querySelector('#storage-ripeness').value = ['ripe','unripe'].includes(ripeness) ? ripeness : 'unknown';
  lastResult.detailsProduce = produce;
  lastResult.photoDetailsEdited = false;
  updatePhotoSummary();
}
function updatePhotoSummary() {
  if (!lastResult) return;
  const condition = document.querySelector('#storage-condition').value;
  const ripeness = document.querySelector('#storage-ripeness').value;
  const suffix = lastResult.photoDetailsEdited ? ' · edited by you' : ' · suggested from photo';
  document.querySelector('#photo-condition').textContent = ({whole:'Whole, unpeeled',cut:'Cut or peeled'})[condition] ? ({whole:'Whole, unpeeled',cut:'Cut or peeled'})[condition] + suffix : 'Not clear from this photo';
  document.querySelector('#photo-ripeness').textContent = ({ripe:'Ripe-looking',unripe:'Unripe-looking'})[ripeness] ? ({ripe:'Ripe-looking',unripe:'Unripe-looking'})[ripeness] + suffix : 'Not clear / not assessed for this produce';
  lastResult.selectedPhotoDetails = {condition,ripeness};
}
function invalidateStorage(event) {
  storageController?.abort();
  storageController = null;
  if (!lastResult) return;
  const no = document.querySelector('#produce-corrected').checked;
  const correction = document.querySelector('#confirmed-produce');
  if (event?.target?.id === 'produce-corrected') correction.value = '';
  document.querySelector('#produce-correction').hidden = !no;
  correction.disabled = !no;
  correction.required = no;
  const produce = chosenProduce();
  if (produce && produce !== lastResult.detailsProduce) populatePhotoDetails(produce);
  if (['storage-condition','storage-ripeness'].includes(event?.target?.id)) lastResult.photoDetailsEdited = true;
  updatePhotoSummary();
  lastResult.storageGuidance = null;
  lastResult.confirmedProduce = document.querySelector('#produce-confirmed').checked || (no && produce) ? produce || null : null;
  document.querySelector('#storage-result').textContent = produce === 'other' ? 'No item-specific guidance is available for other or unidentified items.' : 'Details changed. Submit again for matching storage guidance.';
  renderReport(lastResult);
  saveHistory(lastResult);
}
document.querySelector('#storage-form').addEventListener('input', invalidateStorage);
document.querySelector('#storage-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!lastResult) return;
  const result = lastResult;
  const target = document.querySelector('#storage-result');
  const produce = chosenProduce();
  if (!produceLabels[produce]) { target.textContent = 'No item-specific guidance is available for other or unidentified items.'; return; }
  const data = {produce, confirmed: Boolean(result.confirmedProduce),
    condition: document.querySelector('#storage-condition').value,
    ripeness: document.querySelector('#storage-ripeness').value,
    observation_source: result.photoDetailsEdited ? 'user_edited_photo' : 'photo_model',
    location: document.querySelector('#storage-location').value,
    spoilage: document.querySelector('#storage-spoilage').value,
    days_stored: document.querySelector('#days-stored').value === '' ? null : Number(document.querySelector('#days-stored').value)};
  storageController?.abort();
  const controller = new AbortController(); storageController = controller;
  target.textContent = 'Loading storage guidance…';
  try {
    const response = await fetch('/storage-guidance', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data),signal:controller.signal});
    if (!response.ok) throw new Error('Storage guidance could not be loaded. Check your answers and try again.');
    const advice = await response.json();
    if (controller.signal.aborted || result !== lastResult) return;
    result.confirmedProduce = produce; result.storageGuidance = advice;
    target.innerHTML = `<strong>${escapeHtml(advice.window_label)}</strong><ul>${[...advice.suggestions,...advice.assumptions].map((text) => `<li>${escapeHtml(text)}</li>`).join('')}</ul><p class="helper">${escapeHtml(advice.disclaimer)}</p><p>Sources: ${advice.sources.map((source) => `<a href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(source.title)}</a> (reviewed ${escapeHtml(source.reviewed_on)})`).join('; ')}</p>`;
    saveHistory(result); renderReport(result);
  } catch (error) {
    if (error.name !== 'AbortError' && result === lastResult) target.textContent = error.message;
  }
});
