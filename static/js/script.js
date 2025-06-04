// static/js/script.js
let currentOpenTab = 'TrainTab'; 
let trainingStatusIntervalId = null;
let runningStatusIntervalId = null;

// --- Trainer Canvas Elements & State ---
const trainerDrawCanvasEl = document.getElementById('trainerDrawCanvas');
const trainerProgressImgEl = document.getElementById('trainerProgressImg'); 
const trainDrawColorPicker = document.getElementById('trainDrawColorPicker');
const clearTrainCanvasBtn = document.getElementById('clearTrainCanvasBtn');
const confirmDrawingBtnTrain = document.getElementById('confirmDrawingBtnTrain');
let trainerCtx = null; 
let isDrawingOnTrainerCanvas = false;
// let trainCanvasSnapshotForUndo = null; // Basic undo: one step (Undo/Redo deferred for now)

// --- Runner Canvas & Interaction ---
const previewCanvasRunImgEl = document.getElementById('previewCanvasRun');

// --- DOM Elements (Trainer) ---
const imageFileInputTrain = document.getElementById('imageFileInputTrain');
const loadImageFileBtnTrain = document.getElementById('loadImageFileBtnTrain');

const experimentTypeSelectTrain = document.getElementById('experimentType');
const fireRateInputTrain = document.getElementById('fireRateTrain');
const damageNInputTrain = document.getElementById('damageNTrain');
const damageNLabelTrain = document.getElementById('damageNLabelTrain');
const batchSizeInputTrain = document.getElementById('batchSizeTrain');
const poolSizeInputTrain = document.getElementById('poolSizeTrain');

const initTrainerBtn = document.getElementById('initTrainerBtn');
const startTrainingBtn = document.getElementById('startTrainingBtn');
const stopTrainingBtn = document.getElementById('stopTrainingBtn');
const saveTrainerModelBtn = document.getElementById('saveTrainerModelBtn');
const trainingStatusDiv = document.getElementById('trainingStatus');
const trainModelParamsText = document.getElementById('trainModelParamsText');

// --- DOM Elements (Runner) ---
const modelFileInputRun = document.getElementById('modelFileInputRun');
const loadModelBtnRun = document.getElementById('loadModelBtnRun');
const loadCurrentTrainingModelBtnRun = document.getElementById('loadCurrentTrainingModelBtnRun');
const startRunningLoopBtn = document.getElementById('startRunningLoopBtn');
const stopRunningLoopBtn = document.getElementById('stopRunningLoopBtn');
const resetRunnerStateBtn = document.getElementById('resetRunnerStateBtn');
const rewindBtnRun = document.getElementById('rewindBtnRun');
const skipForwardBtnRun = document.getElementById('skipForwardBtnRun');

const runToolModeEraseRadio = document.getElementById('runToolModeErase'); 
const runToolModeDrawRadio = document.getElementById('runToolModeDraw');   
const runDrawColorPicker = document.getElementById('runDrawColorPicker'); 
const runBrushSizeSlider = document.getElementById('runBrushSizeSlider'); 
const runBrushSizeValue = document.getElementById('runBrushSizeValue');   

const runFpsSlider = document.getElementById('runFpsSlider'); 
const runFpsValue = document.getElementById('runFpsValue');   

const runStatusDiv = document.getElementById('runStatus');
const runModelParamsText = document.getElementById('runModelParamsText');
const globalStatusMessageEl = document.getElementById('globalStatusMessage');

// --- State Variables ---
let trainerInitialized = false; 
let trainerTargetConfirmed = false; 
let trainingLoopActive = false;
let runnerModelLoaded = false;
let runnerLoopActive = false;
let currentRunToolMode = 'erase'; 
let isInteractingWithRunCanvas = false; 

// --- Constants ---
const DRAW_CANVAS_WIDTH = 256; 
const DRAW_CANVAS_HEIGHT = 256;


// --- Utility Functions ---
function showGlobalStatus(message, isSuccess) {
    globalStatusMessageEl.textContent = message;
    globalStatusMessageEl.className = 'global-status-message ' + (isSuccess ? 'success' : 'error');
    globalStatusMessageEl.classList.remove('hidden');
    setTimeout(() => { globalStatusMessageEl.classList.add('hidden'); }, 6000);
}
async function postRequest(url = '', data = {}) { 
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', },
        body: JSON.stringify(data),
    });
    return response.json();
}
async function postFormRequest(url = '', formData = new FormData()) { 
    const response = await fetch(url, {
        method: 'POST',
        body: formData, 
    });
    return response.json();
}

// --- UI Update Functions ---
function updateTrainerControlsAvailability() {
    // Trainer target definition
    confirmDrawingBtnTrain.disabled = trainingLoopActive; 
    loadImageFileBtnTrain.disabled = trainingLoopActive;
    clearTrainCanvasBtn.disabled = trainingLoopActive;
    trainDrawColorPicker.disabled = trainingLoopActive;
    
    initTrainerBtn.disabled = !trainerTargetConfirmed || trainingLoopActive; 
    startTrainingBtn.disabled = !trainerInitialized || trainingLoopActive;
    stopTrainingBtn.disabled = !trainerInitialized || !trainingLoopActive;
    saveTrainerModelBtn.disabled = !trainerInitialized; 
    loadCurrentTrainingModelBtnRun.disabled = !trainerInitialized || trainingLoopActive; 
}

function updateRunnerControlsAvailability() {
    loadModelBtnRun.disabled = runnerLoopActive;
    loadCurrentTrainingModelBtnRun.disabled = runnerLoopActive || !trainerInitialized; 
    startRunningLoopBtn.disabled = !runnerModelLoaded || runnerLoopActive;
    stopRunningLoopBtn.disabled = !runnerModelLoaded || !runnerLoopActive; 
    resetRunnerStateBtn.disabled = !runnerModelLoaded; 
    
    rewindBtnRun.disabled = !runnerModelLoaded; 
    skipForwardBtnRun.disabled = !runnerModelLoaded;
    runBrushSizeSlider.disabled = !runnerModelLoaded;
    runToolModeEraseRadio.disabled = !runnerModelLoaded;
    runToolModeDrawRadio.disabled = !runnerModelLoaded;
    runDrawColorPicker.disabled = !runnerModelLoaded || currentRunToolMode !== 'draw';
    runFpsSlider.disabled = !runnerModelLoaded; 

    if (runnerModelLoaded) {
        previewCanvasRunImgEl.classList.toggle('erase-mode', currentRunToolMode === 'erase');
        previewCanvasRunImgEl.classList.toggle('draw-mode', currentRunToolMode === 'draw');
    } else {
        previewCanvasRunImgEl.className = ''; 
        previewCanvasRunImgEl.style.cursor = 'default';
    }
}

// --- Tab Management ---
function openTab(evt, tabId) { 
    currentOpenTab = tabId;
    document.querySelectorAll(".tab-content").forEach(tc => { tc.style.display = "none"; tc.classList.remove("active"); });
    document.querySelectorAll(".tablinks").forEach(tl => tl.classList.remove("active"));
    
    document.getElementById(tabId).style.display = "block";
    document.getElementById(tabId).classList.add("active");
    evt.currentTarget.classList.add("active");

    document.getElementById('TrainPreviewArea').style.display = (tabId === 'TrainTab') ? 'block' : 'none';
    document.getElementById('RunPreviewArea').style.display = (tabId === 'RunTab') ? 'block' : 'none';

    if (tabId === 'TrainTab') {
        if (runningStatusIntervalId) clearInterval(runningStatusIntervalId); runningStatusIntervalId = null;
        // Start trainer polling if conditions are met (or just fetch once)
        if ((trainerInitialized || trainingLoopActive) && !trainingStatusIntervalId) {
             trainingStatusIntervalId = setInterval(fetchTrainerStatus, 1200); 
        }
        fetchTrainerStatus(); 
        updateTrainerControlsAvailability();
    } else if (tabId === 'RunTab') {
        if (trainingStatusIntervalId) clearInterval(trainingStatusIntervalId); trainingStatusIntervalId = null;
        if (runnerModelLoaded && !runningStatusIntervalId) { 
            const fps = parseInt(runFpsSlider.value);
            runningStatusIntervalId = setInterval(fetchRunnerStatus, Math.max(50, 1000 / fps)); 
        }
        fetchRunnerStatus(); 
        updateRunnerControlsAvailability();
    }
}


// --- Trainer Target Drawing Logic ---
function initializeTrainerDrawCanvas() {
    trainerDrawCanvasEl.width = DRAW_CANVAS_WIDTH;
    trainerDrawCanvasEl.height = DRAW_CANVAS_HEIGHT;
    trainerCtx = trainerDrawCanvasEl.getContext('2d');
    trainerCtx.fillStyle = 'white'; // Default background for drawing canvas
    trainerCtx.fillRect(0, 0, DRAW_CANVAS_WIDTH, DRAW_CANVAS_HEIGHT);
    trainerDrawCanvasEl.classList.add('active-draw'); 
    trainerProgressImgEl.style.display = 'none'; 
    trainerDrawCanvasEl.style.display = 'inline-block';
    trainerTargetConfirmed = false; // Reset confirmation
    initTrainerBtn.disabled = true; // Disable until target is confirmed
    updateTrainerControlsAvailability();
}

function clearTrainerDrawCanvas() {
    if (trainerCtx) {
        trainerCtx.fillStyle = 'white'; 
        trainerCtx.fillRect(0, 0, trainerDrawCanvasEl.width, trainerDrawCanvasEl.height);
        trainerTargetConfirmed = false; 
        initTrainerBtn.disabled = true; 
        updateTrainerControlsAvailability();
        trainingStatusDiv.textContent = "Status: Drawing cleared. Draw a pattern or load a file.";
        // trainerProgressImgEl.style.display = 'none'; // Ensure image is hidden
        // trainerDrawCanvasEl.style.display = 'inline-block'; // Ensure canvas is shown
    }
}

function drawOnTrainerCanvas(event, isDragging) {
    if (!isDrawingOnTrainerCanvas && !isDragging) return; 
    if (!trainerCtx || trainingLoopActive) return; // Don't draw if training active

    const rect = trainerDrawCanvasEl.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    trainerCtx.fillStyle = trainDrawColorPicker.value;
    trainerCtx.beginPath();
    // Simple circle brush, size 5 for now, can be made configurable
    const brushSize = 5; 
    trainerCtx.arc(x, y, brushSize, 0, Math.PI * 2); 
    trainerCtx.fill();

    // Drawing has changed, target needs reconfirmation
    if (trainerTargetConfirmed) { 
        trainingStatusDiv.textContent = "Status: Drawing modified. Confirm again to use as target.";
    }
    trainerTargetConfirmed = false; 
    initTrainerBtn.disabled = true;
    updateTrainerControlsAvailability();
}

trainerDrawCanvasEl.addEventListener('mousedown', (e) => {
    if (e.button !== 0 || trainingLoopActive) return;
    isDrawingOnTrainerCanvas = true;
    drawOnTrainerCanvas(e, false); 
    e.preventDefault();
});
trainerDrawCanvasEl.addEventListener('mousemove', (e) => {
    if (trainingLoopActive) return;
    if (isDrawingOnTrainerCanvas) {
        drawOnTrainerCanvas(e, true);
    }
    e.preventDefault();
});
document.addEventListener('mouseup', (e) => { 
    if (e.button !== 0) return;
    isDrawingOnTrainerCanvas = false;
});
trainerDrawCanvasEl.addEventListener('mouseleave', () => {
    isDrawingOnTrainerCanvas = false; 
});

clearTrainCanvasBtn.addEventListener('click', () => {
    if (trainingLoopActive) {
        showGlobalStatus("Cannot clear drawing while training is active.", false);
        return;
    }
    initializeTrainerDrawCanvas(); // This also clears and resets states
});

confirmDrawingBtnTrain.addEventListener('click', async () => {
    if (trainingLoopActive || !trainerCtx) return;
    const imageDataUrl = trainerDrawCanvasEl.toDataURL('image/png');
    const response = await postRequest('/upload_drawn_pattern_target', { image_data_url: imageDataUrl });
    showGlobalStatus(response.message, response.success);
    if (response.success) {
        trainerTargetConfirmed = true;
        trainerDrawCanvasEl.style.display = 'none'; // Hide drawing
        trainerProgressImgEl.src = `/get_trainer_target_preview?t=${new Date().getTime()}`; // Show processed target
        trainerProgressImgEl.style.display = 'inline-block';
        trainingStatusDiv.textContent = "Status: Drawn pattern confirmed. Initialize Trainer.";
    }
    updateTrainerControlsAvailability();
});


// --- Training Tab Logic (Rest of it) ---
experimentTypeSelectTrain.addEventListener('change', () => {
    const isRegen = experimentTypeSelectTrain.value === 'Regenerating';
    damageNInputTrain.style.display = isRegen ? 'block' : 'none';
    damageNLabelTrain.style.display = isRegen ? 'block' : 'none';
});

async function handleLoadTargetForTrainerFromFile(formData) { 
    const response = await postFormRequest('/load_target_from_file', formData);
    showGlobalStatus(response.message, response.success);
    if (response.success) {
        trainerDrawCanvasEl.style.display = 'none';
        trainerProgressImgEl.src = `/get_trainer_target_preview?t=${new Date().getTime()}`; 
        trainerProgressImgEl.style.display = 'inline-block';
        
        trainerTargetConfirmed = true; 
        trainerInitialized = false; 
        trainingLoopActive = false;
        if (trainingStatusIntervalId) clearInterval(trainingStatusIntervalId); trainingStatusIntervalId = null;
        trainModelParamsText.textContent = "N/A (File target loaded, initialize Trainer)";
        trainingStatusDiv.textContent = "Status: File target loaded. Initialize Trainer.";
    }
    updateTrainerControlsAvailability();
}
loadImageFileBtnTrain.addEventListener('click', () => {
    if (trainingLoopActive) {
        showGlobalStatus("Cannot load file while training.", false); return;
    }
    if (!imageFileInputTrain.files.length) {
        showGlobalStatus('Please select an image file for the trainer.', false); return;
    }
    const formData = new FormData();
    formData.append('image_file', imageFileInputTrain.files[0]);
    handleLoadTargetForTrainerFromFile(formData);
});

initTrainerBtn.addEventListener('click', async () => {
    if (!trainerTargetConfirmed) {
        showGlobalStatus("Please confirm a drawn target or load a file first.", false); return;
    }
    trainerDrawCanvasEl.style.display = 'none'; 
    trainerProgressImgEl.style.display = 'inline-block'; 

    const payload = {
        experiment_type: experimentTypeSelectTrain.value,
        fire_rate: parseFloat(fireRateInputTrain.value),
        damage_n: parseInt(damageNInputTrain.value),
        batch_size: parseInt(batchSizeInputTrain.value),
        pool_size: parseInt(poolSizeInputTrain.value),
    };
    const response = await postRequest('/initialize_trainer', payload);
    showGlobalStatus(response.message, response.success);
    if (response.success) {
        trainModelParamsText.textContent = response.model_summary || 'N/A';
        trainerProgressImgEl.src = `${response.initial_state_preview_url}?t=${new Date().getTime()}`;
        trainerInitialized = true;
        trainingLoopActive = false;
        if (trainingStatusIntervalId) clearInterval(trainingStatusIntervalId); 
        if (!trainingStatusIntervalId && currentOpenTab === 'TrainTab') {
             trainingStatusIntervalId = setInterval(fetchTrainerStatus, 1200);
        }
        trainingStatusDiv.textContent = "Status: Trainer Initialized.";
    } else {
        trainerInitialized = false;
        // If init fails, revert to showing drawing canvas IF no file was the source
        // This state handling can be complex. For now, rely on user to re-init target.
        // initializeTrainerDrawCanvas(); // Or show error and let user decide
    }
    updateTrainerControlsAvailability();
});

startTrainingBtn.addEventListener('click', async () => {
    if (!trainerInitialized) return;
    const response = await postRequest('/start_training');
    showGlobalStatus(response.message, response.success);
    if (response.success) {
        trainingLoopActive = true;
        trainerDrawCanvasEl.style.display = 'none'; // Ensure drawing canvas is hidden
        trainerProgressImgEl.style.display = 'inline-block'; // Ensure progress img is visible
        if (!trainingStatusIntervalId && currentOpenTab === 'TrainTab') { 
            trainingStatusIntervalId = setInterval(fetchTrainerStatus, 1200); 
        }
    }
    updateTrainerControlsAvailability();
});

stopTrainingBtn.addEventListener('click', async () => {
    if (!trainerInitialized) { 
         showGlobalStatus("Trainer not initialized.", false); return;
    }
    if (!trainingLoopActive && trainerInitialized) { 
         showGlobalStatus("Training is not currently running.", false); return;
    }
    const response = await postRequest('/stop_training'); 
    showGlobalStatus(response.message, response.success);
    trainingLoopActive = false; 
    fetchTrainerStatus(); // Update status immediately
    updateTrainerControlsAvailability();
});

saveTrainerModelBtn.addEventListener('click', async () => {
    if (!trainerInitialized) return; 
    const response = await postRequest('/save_trainer_model');
    showGlobalStatus(response.message, response.success);
});

async function fetchTrainerStatus() {
    if (currentOpenTab !== 'TrainTab' && trainingStatusIntervalId) { return; }
    if (!trainerInitialized && !trainerTargetConfirmed && trainingStatusIntervalId) { 
        // If completely reset (no target, not init), stop polling and show drawing canvas
        clearInterval(trainingStatusIntervalId); trainingStatusIntervalId = null;
        initializeTrainerDrawCanvas(); // Show drawing canvas
        trainingStatusDiv.textContent = "Status: Define a target pattern.";
        updateTrainerControlsAvailability(); return;
    }
     if (!trainerInitialized && trainerTargetConfirmed && trainingStatusIntervalId) {
        // Target confirmed, but not initialized yet, show target preview, don't poll training status
        // This case is handled by confirm/load file which shows target preview.
        // Polling for trainer steps should only start after init.
        // No, actually, even if not initialized, we might want to poll if target is set
        // to keep the "target preview" fresh if that endpoint changes.
        // For now, let's simplify: only poll training steps if trainer IS initialized.
        // Or, have a different endpoint for just "get_trainer_target_preview" if needed separately.
        // Let's assume get_training_status also returns preview_url for initial state if just initialized.
    }


    try {
        const response = await fetch('/get_training_status');
        if (!response.ok) { 
            trainingStatusDiv.textContent = `Trainer status error: ${response.status}`;
            // trainingLoopActive = false; // Don't assume, could be transient network error
            updateTrainerControlsAvailability(); return; 
        }
        const data = await response.json();
        trainingStatusDiv.textContent = data.status_message || `Step: ${data.step || 0}, Loss: ${data.loss || 'N/A'}, Time: ${data.training_time || 'N/A'}`;
        
        if (data.is_training || trainerInitialized) { 
            trainerDrawCanvasEl.style.display = 'none';
            trainerProgressImgEl.style.display = 'inline-block';
            if (data.preview_url) trainerProgressImgEl.src = `${data.preview_url}?t=${new Date().getTime()}`;
        } else if (trainerTargetConfirmed) { // Target is set, but not initialized or training
            trainerDrawCanvasEl.style.display = 'none'; // Should show static target preview
            trainerProgressImgEl.style.display = 'inline-block';
            // Ensure trainerProgressImg shows the confirmed target
             trainerProgressImgEl.src = `/get_trainer_target_preview?t=${new Date().getTime()}`;
        }
         else { // No target confirmed, not initialized
            initializeTrainerDrawCanvas(); // Default to drawing mode
        }

        trainingLoopActive = data.is_training; 
        updateTrainerControlsAvailability();

        // If training just finished or stopped, and interval is still active, clear it.
        if (!trainingLoopActive && trainerInitialized && trainingStatusIntervalId && data.step > 0) {
            // clearInterval(trainingStatusIntervalId); trainingStatusIntervalId = null;
            // Or slow it down considerably if we want to keep last state visible via polling
        }


    } catch (error) { 
        trainingStatusDiv.textContent = `Trainer status fetch error: ${error}.`;
        // trainingLoopActive = false; // Don't assume on fetch error
        // if (trainingStatusIntervalId) clearInterval(trainingStatusIntervalId); trainingStatusIntervalId = null;
        updateTrainerControlsAvailability();
    }
}

// --- Run Tab Logic ---
runBrushSizeSlider.addEventListener('input', () => { runBrushSizeValue.textContent = runBrushSizeSlider.value; });
runFpsSlider.addEventListener('input', async () => {
    const fps = parseInt(runFpsSlider.value);
    runFpsValue.textContent = fps;
    if (runnerModelLoaded) { 
        await postRequest('/set_runner_speed', { fps: fps });
        // Polling interval will be adjusted by fetchRunnerStatus based on server's current_fps
    }
});
runToolModeEraseRadio.addEventListener('change', () => { if (runToolModeEraseRadio.checked) currentRunToolMode = 'erase'; updateRunnerControlsAvailability(); });
runToolModeDrawRadio.addEventListener('change', () => { if (runToolModeDrawRadio.checked) currentRunToolMode = 'draw'; updateRunnerControlsAvailability(); });

loadCurrentTrainingModelBtnRun.addEventListener('click', async () => {
    if (!trainerInitialized) {
        showGlobalStatus("Trainer model not available (trainer not initialized).", false); return;
    }
    if (trainingLoopActive) {
         showGlobalStatus("Wait for training to stop or pause before loading its model.", false); return;
    }
    const response = await postRequest('/load_current_training_model_for_runner');
    showGlobalStatus(response.message, response.success);
    if (response.success) {
        runModelParamsText.textContent = response.model_summary || 'N/A';
        previewCanvasRunImgEl.src = `${response.runner_preview_url}?t=${new Date().getTime()}`;
        runnerModelLoaded = true;
        runnerLoopActive = false; 
        if (runningStatusIntervalId) clearInterval(runningStatusIntervalId); 
        const initialFps = parseInt(runFpsSlider.value);
        if (!runningStatusIntervalId && currentOpenTab === 'RunTab') {
             runningStatusIntervalId = setInterval(fetchRunnerStatus, Math.max(50, 1000 / initialFps)); 
        }
        runStatusDiv.textContent = "Status: Runner: Loaded current training model. FPS: " + initialFps;
        runFpsSlider.dispatchEvent(new Event('input')); 
    } else {
        // runnerModelLoaded remains as is
    }
    updateRunnerControlsAvailability();
});

loadModelBtnRun.addEventListener('click', async () => { /* ... (Same as previous good version) ... */
    const formData = new FormData();
    if (modelFileInputRun.files.length > 0) formData.append('model_file', modelFileInputRun.files[0]);
    const response = await postFormRequest('/load_model_for_runner', formData);
    showGlobalStatus(response.message, response.success);
    if (response.success) {
        runModelParamsText.textContent = response.model_summary || 'N/A';
        previewCanvasRunImgEl.src = `${response.runner_preview_url}?t=${new Date().getTime()}`;
        runnerModelLoaded = true;
        runnerLoopActive = false; 
        if (runningStatusIntervalId) clearInterval(runningStatusIntervalId); 
        const initialFps = parseInt(runFpsSlider.value); 
        if (!runningStatusIntervalId && currentOpenTab === 'RunTab') {
             runningStatusIntervalId = setInterval(fetchRunnerStatus, Math.max(50, 1000 / initialFps)); 
        }
        runStatusDiv.textContent = "Status: Runner: Model loaded. FPS: " + initialFps;
        runFpsSlider.dispatchEvent(new Event('input')); 
    } else {
        runnerModelLoaded = false;
    }
    updateRunnerControlsAvailability();
});
startRunningLoopBtn.addEventListener('click', async () => { /* ... (Same as previous good version) ... */
    if (!runnerModelLoaded) return;
    const response = await postRequest('/start_running');
    showGlobalStatus(response.message, response.success);
    if (response.success) {
        runnerLoopActive = true;
        if (!runningStatusIntervalId && currentOpenTab === 'RunTab') { 
            runningStatusIntervalId = setInterval(fetchRunnerStatus, Math.max(50, 1000 / parseInt(runFpsSlider.value))); 
        }
    }
    updateRunnerControlsAvailability();
});
stopRunningLoopBtn.addEventListener('click', async () => { /* ... (Same as previous good version) ... */
    if(!runnerModelLoaded) {
        showGlobalStatus("Runner: No model loaded to stop.", false); return;
    }
    const response = await postRequest('/stop_running');
    showGlobalStatus(response.message, response.success);
    // runnerLoopActive will be updated by fetchRunnerStatus after server confirms
    fetchRunnerStatus(); 
    updateRunnerControlsAvailability();
});
resetRunnerStateBtn.addEventListener('click', async () => { /* ... (Same as previous good version) ... */
    if (!runnerModelLoaded) return;
    await handleRunnerAction('reset_runner'); 
    runnerLoopActive = false; 
    updateRunnerControlsAvailability();
});
rewindBtnRun.addEventListener('click', async () => handleRunnerAction('rewind'));
skipForwardBtnRun.addEventListener('click', async () => handleRunnerAction('skip_forward'));
async function handleRunnerAction(action, params = {}) { /* ... (Same as previous good version, ensure correct IDs for color picker etc.) ... */
    if (!runnerModelLoaded) return;
    const payload = { action, ...params };
    const response = await postRequest('/runner_action', payload);
    if(response.success && response.preview_url) {
        previewCanvasRunImgEl.src = `${response.preview_url}?t=${new Date().getTime()}`; 
        const currentFPS = parseFloat(runFpsSlider.value).toFixed(1); 
        runStatusDiv.textContent = `${response.message} (Target: ${currentFPS} FPS)`;
         if (action !== 'modify_area') { /* showGlobalStatus for non-drag actions */ }
    } else if (!response.success) { showGlobalStatus(response.message, false); }
    if (action === 'reset_runner') { runnerLoopActive = false; } // Reset implies loop stops
    updateRunnerControlsAvailability(); 
}
function performCanvasAction(event, isDrag = false) { /* ... (Same as previous good version, ensure correct IDs) ... */
    if (!runnerModelLoaded || currentOpenTab !== 'RunTab' ) return; 
    const rect = previewCanvasRunImgEl.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return; 
    const x = event.clientX - rect.left; const y = event.clientY - rect.top;
    if (x < 0 || x > rect.width || y < 0 || y > rect.height) {
        if (isDrag) isInteractingWithRunCanvas = false; return;
    }
    const normX = Math.max(0, Math.min(1, x / rect.width)); 
    const normY = Math.max(0, Math.min(1, y / rect.height));
    const brushSliderVal = parseInt(runBrushSizeSlider.value); 
    const normBrushFactor = (brushSliderVal / 30) * 0.20 + 0.01; 
    handleRunnerAction('modify_area', { 
        tool_mode: currentRunToolMode, // Use currentRunToolMode
        draw_color_hex: runDrawColorPicker.value, // Use runDrawColorPicker
        norm_x: normX, norm_y: normY, brush_size_norm: normBrushFactor,
        canvas_render_width: rect.width, canvas_render_height: rect.height
    });
}
previewCanvasRunImgEl.addEventListener('mousedown', (event) => { /* ... */ });
previewCanvasRunImgEl.addEventListener('mousemove', (event) => { /* ... */ });
document.addEventListener('mouseup', (event) => { /* ... */ });
previewCanvasRunImgEl.addEventListener('mouseleave', () => { /* ... */ });
// (Paste full mousedown, mousemove, mouseup, mouseleave for runner from previous good version)
previewCanvasRunImgEl.addEventListener('mousedown', (event) => {
    if (event.button !== 0) return; 
    if (!runnerModelLoaded || currentOpenTab !== 'RunTab') return;
    isInteractingWithRunCanvas = true;
    performCanvasAction(event); 
    event.preventDefault(); 
});
previewCanvasRunImgEl.addEventListener('mousemove', (event) => {
    if (!isInteractingWithRunCanvas) return; 
    performCanvasAction(event, true); 
    event.preventDefault();
});
document.addEventListener('mouseup', (event) => {
    if (event.button !== 0) return;
    if (isInteractingWithRunCanvas) {
        isInteractingWithRunCanvas = false;
    }
});
previewCanvasRunImgEl.addEventListener('mouseleave', () => {
    if (isInteractingWithRunCanvas) {
        isInteractingWithRunCanvas = false;
    }
});

async function fetchRunnerStatus() { /* ... (Same as previous good version) ... */
    if (currentOpenTab !== 'RunTab' && runningStatusIntervalId) { return; }
    if (!runnerModelLoaded && runningStatusIntervalId) { 
        clearInterval(runningStatusIntervalId); runningStatusIntervalId = null;
        runStatusDiv.textContent = "Status: Runner: No model loaded.";
        updateRunnerControlsAvailability(); return;
    }
    try {
        const response = await fetch('/get_runner_status');
        if (!response.ok) { 
            runStatusDiv.textContent = `Runner status error: ${response.status}`;
            runnerLoopActive = false; updateRunnerControlsAvailability(); return;
        }
        const data = await response.json();
        const serverTargetFPS = data.current_fps === "Max" ? "Max" : parseFloat(data.current_fps).toFixed(1);
        runStatusDiv.textContent = `${data.status_message || 'Status unavailable'} (Target: ${serverTargetFPS} FPS)`;
        
        if (!isInteractingWithRunCanvas || data.is_loop_active) {
            if (data.preview_url) previewCanvasRunImgEl.src = `${data.preview_url}?t=${new Date().getTime()}`;
        }
        runnerLoopActive = data.is_loop_active; 
        if (runningStatusIntervalId) {
            clearInterval(runningStatusIntervalId); 
            let newIntervalTime = 1000; 
            if (runnerLoopActive) {
                const targetFpsNum = parseFloat(data.current_fps);
                newIntervalTime = (targetFpsNum && targetFpsNum > 0) ? Math.max(33, 1000 / targetFpsNum) : 50;
            }
            runningStatusIntervalId = setInterval(fetchRunnerStatus, newIntervalTime);
        } else if (runnerModelLoaded && currentOpenTab === 'RunTab') { 
             const targetFpsNum = parseFloat(data.current_fps);
             const intervalTime = (runnerLoopActive && targetFpsNum && targetFpsNum > 0) ? Math.max(33, 1000/targetFpsNum) : 1000;
             runningStatusIntervalId = setInterval(fetchRunnerStatus, intervalTime);
        }
        updateRunnerControlsAvailability();
    } catch (error) { 
        runStatusDiv.textContent = `Runner status fetch error: ${error}. Polling might stop.`;
        runnerLoopActive = false; 
        if (runningStatusIntervalId) clearInterval(runningStatusIntervalId); runningStatusIntervalId = null; 
        updateRunnerControlsAvailability();
    }
}


// --- Initial Setup ---
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('trainTabButton').click(); 
    initializeTrainerDrawCanvas(); 

    experimentTypeSelectTrain.addEventListener('change', () => {
        const isRegen = experimentTypeSelectTrain.value === 'Regenerating';
        damageNInputTrain.style.display = isRegen ? 'block' : 'none';
        damageNLabelTrain.style.display = isRegen ? 'block' : 'none';
    });
    experimentTypeSelectTrain.dispatchEvent(new Event('change'));

    runBrushSizeValue.textContent = runBrushSizeSlider.value; 
    runFpsValue.textContent = runFpsSlider.value;       
    currentRunToolMode = runToolModeEraseRadio.checked ? 'erase' : 'draw'; 


    const placeholderDim = DRAW_CANVAS_WIDTH; 
    const placeholderColor = '#f0f0f0';
    const svgPlaceholder = (text) => `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='${placeholderDim}' height='${placeholderDim}' viewBox='0 0 ${placeholderDim} ${placeholderDim}'%3E%3Crect width='100%25' height='100%25' fill='${placeholderColor}'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' font-family='sans-serif' font-size='16' fill='%236c757d'%3E${text}%3C/text%3E%3C/svg%3E`;
    
    // trainerDrawCanvas is initialized directly. trainerProgressImg is initially hidden.
    trainerProgressImgEl.src = svgPlaceholder('Target / Training Progress'); 
    previewCanvasRunImgEl.src = svgPlaceholder('Runner Preview');

    updateTrainerControlsAvailability();
    updateRunnerControlsAvailability();

    // Initial status fetch for the default active tab
    if (currentOpenTab === 'TrainTab') fetchTrainerStatus();
    else if (currentOpenTab === 'RunTab') fetchRunnerStatus();
});