// static/js/script.js
let currentOpenTab = 'TrainTab'; 
let trainingStatusIntervalId = null;
let runningStatusIntervalId = null;

const previewCanvasTrainImgEl = document.getElementById('previewCanvasTrain');
const previewCanvasRunImgEl = document.getElementById('previewCanvasRun');

// --- DOM Elements (Trainer) ---
const loadEmojiBtnTrain = document.getElementById('loadEmojiBtnTrain');
const imageFileInputTrain = document.getElementById('imageFileInputTrain');
const loadImageFileBtnTrain = document.getElementById('loadImageFileBtnTrain');
const emojiInputTrain = document.getElementById('emojiInput'); // Assuming same ID is fine if contextually clear

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
const startRunningLoopBtn = document.getElementById('startRunningLoopBtn');
const stopRunningLoopBtn = document.getElementById('stopRunningLoopBtn');
const resetRunnerStateBtn = document.getElementById('resetRunnerStateBtn');
const rewindBtnRun = document.getElementById('rewindBtnRun');
const skipForwardBtnRun = document.getElementById('skipForwardBtnRun');
const eraserSizeSliderRun = document.getElementById('eraserSizeSliderRun');
const eraserSizeValueRun = document.getElementById('eraserSizeValueRun');
const runStatusDiv = document.getElementById('runStatus');
const runModelParamsText = document.getElementById('runModelParamsText');

const globalStatusMessageEl = document.getElementById('globalStatusMessage');

// --- State Variables ---
let trainerInitialized = false;
let trainingLoopActive = false;
let runnerModelLoaded = false;
let runnerLoopActive = false;


// --- Utility Functions ---
function showGlobalStatus(message, isSuccess) {
    globalStatusMessageEl.textContent = message;
    globalStatusMessageEl.className = 'global-status-message ' + (isSuccess ? 'success' : 'error');
    globalStatusMessageEl.classList.remove('hidden');
    setTimeout(() => { globalStatusMessageEl.classList.add('hidden'); }, 6000);
}

async function postRequest(url = '', data = {}) { // Renamed for clarity
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', },
        body: JSON.stringify(data),
    });
    return response.json();
}

async function postFormRequest(url = '', formData = new FormData()) { // Renamed
    const response = await fetch(url, {
        method: 'POST',
        body: formData, 
    });
    return response.json();
}

// --- UI Update Functions ---
function updateTrainerControlsAvailability() {
    // Trainer controls
    loadEmojiBtnTrain.disabled = trainingLoopActive;
    loadImageFileBtnTrain.disabled = trainingLoopActive;
    initTrainerBtn.disabled = trainingLoopActive; 
    startTrainingBtn.disabled = !trainerInitialized || trainingLoopActive;
    stopTrainingBtn.disabled = !trainerInitialized || !trainingLoopActive;
    saveTrainerModelBtn.disabled = !trainerInitialized; // Can save if initialized, even if not run/stopped
}

function updateRunnerControlsAvailability() {
    // Runner controls
    loadModelBtnRun.disabled = runnerLoopActive;
    startRunningLoopBtn.disabled = !runnerModelLoaded || runnerLoopActive;
    stopRunningLoopBtn.disabled = !runnerModelLoaded || !runnerLoopActive;
    resetRunnerStateBtn.disabled = !runnerModelLoaded; // Can reset if loaded, even if loop paused
    rewindBtnRun.disabled = !runnerModelLoaded || runnerLoopActive; 
    skipForwardBtnRun.disabled = !runnerModelLoaded || runnerLoopActive;
    eraserSizeSliderRun.disabled = !runnerModelLoaded;
    previewCanvasRunImgEl.classList.toggle('active-erase', runnerModelLoaded && !runnerLoopActive); // Erase on paused
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

    // Manage status polling intervals based on active tab
    if (tabId === 'TrainTab') {
        if (runningStatusIntervalId) clearInterval(runningStatusIntervalId); runningStatusIntervalId = null;
        if (trainerInitialized && !trainingStatusIntervalId) { // Start polling if trainer is init but interval not set
             // trainingStatusIntervalId = setInterval(fetchTrainerStatus, 1500); // Start polling for trainer
        }
        fetchTrainerStatus(); // Immediate fetch for current tab
        updateTrainerControlsAvailability();
    } else if (tabId === 'RunTab') {
        if (trainingStatusIntervalId) clearInterval(trainingStatusIntervalId); trainingStatusIntervalId = null;
        if (runnerModelLoaded && !runningStatusIntervalId) { // Start polling if runner is loaded
            // runningStatusIntervalId = setInterval(fetchRunnerStatus, 300); // Start polling for runner
        }
        fetchRunnerStatus(); 
        updateRunnerControlsAvailability();
    }
}

// --- Training Tab Logic ---
experimentTypeSelectTrain.addEventListener('change', () => {
    const isRegen = experimentTypeSelectTrain.value === 'Regenerating';
    damageNInputTrain.style.display = isRegen ? 'block' : 'none';
    damageNLabelTrain.style.display = isRegen ? 'block' : 'none';
});

async function handleLoadTargetForTrainer(formData) {
    const response = await postFormRequest('/load_target', formData);
    showGlobalStatus(response.message, response.success);
    if (response.success) {
        previewCanvasTrainImgEl.src = `/get_trainer_target_preview?t=${new Date().getTime()}`;
        trainerInitialized = false; 
        trainingLoopActive = false;
        if (trainingStatusIntervalId) clearInterval(trainingStatusIntervalId); trainingStatusIntervalId = null;
        trainModelParamsText.textContent = "N/A (Target changed, re-initialize Trainer)";
        trainingStatusDiv.textContent = "Status: Trainer target loaded. Initialize Trainer.";
    }
    updateTrainerControlsAvailability();
}

loadEmojiBtnTrain.addEventListener('click', () => {
    const formData = new FormData();
    formData.append('emoji', emojiInputTrain.value);
    handleLoadTargetForTrainer(formData);
});

loadImageFileBtnTrain.addEventListener('click', () => {
    if (!imageFileInputTrain.files.length) {
        showGlobalStatus('Please select an image file for the trainer.', false); return;
    }
    const formData = new FormData();
    formData.append('image_file', imageFileInputTrain.files[0]);
    handleLoadTargetForTrainer(formData);
});

initTrainerBtn.addEventListener('click', async () => {
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
        previewCanvasTrainImgEl.src = `${response.initial_state_preview_url}?t=${new Date().getTime()}`;
        trainerInitialized = true;
        trainingLoopActive = false;
        if (trainingStatusIntervalId) clearInterval(trainingStatusIntervalId); trainingStatusIntervalId = null;
        // Start polling trainer status if not already
        if (!trainingStatusIntervalId && currentOpenTab === 'TrainTab') {
             trainingStatusIntervalId = setInterval(fetchTrainerStatus, 1200);
        }
        trainingStatusDiv.textContent = "Status: Trainer Initialized.";
    } else {
        trainerInitialized = false;
    }
    updateTrainerControlsAvailability();
});

startTrainingBtn.addEventListener('click', async () => {
    if (!trainerInitialized) return;
    const response = await postRequest('/start_training');
    showGlobalStatus(response.message, response.success);
    if (response.success) {
        trainingLoopActive = true;
        if (!trainingStatusIntervalId && currentOpenTab === 'TrainTab') { // Ensure polling starts
            trainingStatusIntervalId = setInterval(fetchTrainerStatus, 1200); 
        }
    }
    updateTrainerControlsAvailability();
});

stopTrainingBtn.addEventListener('click', async () => {
    if (!trainingLoopActive && !trainerInitialized) return; // Nothing to stop if not running or init
    const response = await postRequest('/stop_training'); // Backend stops thread, preserves model
    showGlobalStatus(response.message, response.success);
    trainingLoopActive = false; 
    // Keep polling for a final status update, then it will see is_training is false
    // Or clear interval if stop means "done with this session"
    // Let's keep it polling slowly, fetchTrainerStatus will adapt if training is false.
    fetchTrainerStatus(); // Get immediate status after stop request
    updateTrainerControlsAvailability();
});

saveTrainerModelBtn.addEventListener('click', async () => {
    if (!trainerInitialized) return; 
    const response = await postRequest('/save_trainer_model');
    showGlobalStatus(response.message, response.success);
});

async function fetchTrainerStatus() {
    if (currentOpenTab !== 'TrainTab' && trainingStatusIntervalId) { 
        // If tab switched away, clear specific interval.
        // clearInterval(trainingStatusIntervalId); trainingStatusIntervalId = null;
        return; 
    }
    if (!trainerInitialized && trainingStatusIntervalId) { // If trainer reset, stop polling
        clearInterval(trainingStatusIntervalId); trainingStatusIntervalId = null;
        trainingStatusDiv.textContent = "Status: Trainer not initialized.";
        updateTrainerControlsAvailability();
        return;
    }

    try {
        const response = await fetch('/get_training_status');
        if (!response.ok) {
            trainingStatusDiv.textContent = `Trainer status error: ${response.status}`;
            trainingLoopActive = false; 
            // clearInterval(trainingStatusIntervalId); trainingStatusIntervalId = null; // Stop on error
            updateTrainerControlsAvailability();
            return;
        }
        const data = await response.json();
        trainingStatusDiv.textContent = data.status_message || `Step: ${data.step || 0}, Loss: ${data.loss || 'N/A'}, Time: ${data.training_time || 'N/A'}`;
        if (data.preview_url) {
             previewCanvasTrainImgEl.src = `${data.preview_url}?t=${new Date().getTime()}`;
        }
        
        trainingLoopActive = data.is_training; 
        
        if (!trainingLoopActive && data.step > 0 && trainingStatusIntervalId) {
            // If training finished/stopped by server, slow down polling or stop
            // For now, we keep polling slowly to reflect final state.
            // clearInterval(trainingStatusIntervalId); trainingStatusIntervalId = null;
        }
        updateTrainerControlsAvailability();

    } catch (error) {
        trainingStatusDiv.textContent = `Trainer status fetch error: ${error}. Polling stopped.`;
        trainingLoopActive = false;
        if (trainingStatusIntervalId) clearInterval(trainingStatusIntervalId); trainingStatusIntervalId = null;
        updateTrainerControlsAvailability();
    }
}

// --- Run Tab Logic ---
eraserSizeSliderRun.addEventListener('input', () => {
    eraserSizeValueRun.textContent = eraserSizeSliderRun.value;
});

loadModelBtnRun.addEventListener('click', async () => {
    const formData = new FormData();
    if (modelFileInputRun.files.length > 0) {
        formData.append('model_file', modelFileInputRun.files[0]);
    } 
    const response = await postFormRequest('/load_model_for_runner', formData);
    showGlobalStatus(response.message, response.success);
    if (response.success) {
        runModelParamsText.textContent = response.model_summary || 'N/A';
        previewCanvasRunImgEl.src = `${response.runner_preview_url}?t=${new Date().getTime()}`;
        runnerModelLoaded = true;
        runnerLoopActive = false; 
        if (runningStatusIntervalId) clearInterval(runningStatusIntervalId); runningStatusIntervalId = null;
        if (!runningStatusIntervalId && currentOpenTab === 'RunTab') { // Start polling if not already
             runningStatusIntervalId = setInterval(fetchRunnerStatus, 300); // Faster for run
        }
        runStatusDiv.textContent = "Status: Runner: Model loaded.";
    } else {
        runnerModelLoaded = false;
    }
    updateRunnerControlsAvailability();
});

startRunningLoopBtn.addEventListener('click', async () => {
    if (!runnerModelLoaded) return;
    const response = await postRequest('/start_running');
    showGlobalStatus(response.message, response.success);
    if (response.success) {
        runnerLoopActive = true;
        if (!runningStatusIntervalId && currentOpenTab === 'RunTab') { 
            runningStatusIntervalId = setInterval(fetchRunnerStatus, 300); 
        }
    }
    updateRunnerControlsAvailability();
});

stopRunningLoopBtn.addEventListener('click', async () => {
    if (!runnerLoopActive && !runnerModelLoaded) return; 
    const response = await postRequest('/stop_running');
    showGlobalStatus(response.message, response.success);
    runnerLoopActive = false; 
    // Keep polling for history nav, fetchRunnerStatus will show loop is inactive
    fetchRunnerStatus(); 
    updateRunnerControlsAvailability();
});

resetRunnerStateBtn.addEventListener('click', async () => {
    if (!runnerModelLoaded) return;
    await handleRunnerAction('reset_runner'); // This will also fetch status via its own call
    runnerLoopActive = false; // Reset stops the loop
    // No need to clear interval here, let fetchRunnerStatus manage
    updateRunnerControlsAvailability();
});


rewindBtnRun.addEventListener('click', async () => handleRunnerAction('rewind'));
skipForwardBtnRun.addEventListener('click', async () => handleRunnerAction('skip_forward'));

async function handleRunnerAction(action, params = {}) {
    if (!runnerModelLoaded) return;
    const payload = { action, ...params };
    const response = await postRequest('/runner_action', payload);
    showGlobalStatus(response.message, response.success);
     if(response.success && response.preview_url) {
        previewCanvasRunImgEl.src = `${response.preview_url}?t=${new Date().getTime()}`;
        runStatusDiv.textContent = response.message || `Runner: ${action} at step ${response.history_step}/${response.total_history-1}.`;
    }
    // Loop state (runnerLoopActive) isn't changed by history nav or erase by default.
    // Server response for is_loop_active in fetchRunnerStatus will reflect if loop was running.
    updateRunnerControlsAvailability(); 
}

previewCanvasRunImgEl.addEventListener('click', (event) => {
    if (!runnerModelLoaded || currentOpenTab !== 'RunTab' ) return; 
    if (runnerLoopActive) { // Only allow erase if loop is paused/stopped
        showGlobalStatus("Pause the running loop to erase.", false);
        return;
    }

    const rect = previewCanvasRunImgEl.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return; // Image not rendered yet

    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const normX = Math.max(0, Math.min(1, x / rect.width)); 
    const normY = Math.max(0, Math.min(1, y / rect.height));
    
    const eraserSliderVal = parseInt(eraserSizeSliderRun.value); 
    const normEraserFactor = (eraserSliderVal / 20) * 0.15 + 0.01; // Map 1-20 to ~0.01 - 0.16

    handleRunnerAction('erase', { 
        norm_x: normX, 
        norm_y: normY, 
        eraser_size_norm: normEraserFactor,
        canvas_render_width: rect.width, 
        canvas_render_height: rect.height
    });
});

async function fetchRunnerStatus() {
    if (currentOpenTab !== 'RunTab' && runningStatusIntervalId) {
        // clearInterval(runningStatusIntervalId); runningStatusIntervalId = null;
        return; 
    }
    if (!runnerModelLoaded && runningStatusIntervalId) { 
        clearInterval(runningStatusIntervalId); runningStatusIntervalId = null;
        runStatusDiv.textContent = "Status: Runner: No model loaded.";
        updateRunnerControlsAvailability();
        return;
    }
    
    try {
        const response = await fetch('/get_runner_status');
        if (!response.ok) {
            runStatusDiv.textContent = `Runner status error: ${response.status}`;
            runnerLoopActive = false;
            // clearInterval(runningStatusIntervalId); runningStatusIntervalId = null;
            updateRunnerControlsAvailability();
            return;
        }
        const data = await response.json();
        runStatusDiv.textContent = data.status_message || `Runner Loop: ${data.is_loop_active ? 'Active' : 'Paused'}, Step: ${data.history_step || 0}/${(data.total_history-1) || 0}`;
        
        if (data.preview_url) {
             previewCanvasRunImgEl.src = `${data.preview_url}?t=${new Date().getTime()}`;
        }
        
        runnerLoopActive = data.is_loop_active; 
        
        // If loop is not active (e.g. paused, stopped, or finished an action)
        // we still want to poll, but maybe slower if no user interaction is expected.
        // For now, constant polling interval when tab is open and model loaded.
        updateRunnerControlsAvailability();

    } catch (error) {
        runStatusDiv.textContent = `Runner status fetch error: ${error}. Polling might stop.`;
        runnerLoopActive = false; // Assume error means loop is not active
        // clearInterval(runningStatusIntervalId); runningStatusIntervalId = null; // Option to stop polling on error
        updateRunnerControlsAvailability();
    }
}

// --- Initial Setup ---
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('trainTabButton').click(); // Activate TrainTab initially
    
    const isRegen = experimentTypeSelectTrain.value === 'Regenerating';
    damageNInputTrain.style.display = isRegen ? 'block' : 'none';
    damageNLabelTrain.style.display = isRegen ? 'block' : 'none';
    eraserSizeValueRun.textContent = eraserSizeSliderRun.value;

    const placeholderDim = 256; 
    const placeholderColor = '#f0f0f0';
    const svgPlaceholder = (text) => `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='${placeholderDim}' height='${placeholderDim}' viewBox='0 0 ${placeholderDim} ${placeholderDim}'%3E%3Crect width='100%25' height='100%25' fill='${placeholderColor}'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' font-family='sans-serif' font-size='16' fill='%236c757d'%3E${text}%3C/text%3E%3C/svg%3E`;
    
    previewCanvasTrainImgEl.src = svgPlaceholder('Trainer Preview');
    previewCanvasRunImgEl.src = svgPlaceholder('Runner Preview');

    updateTrainerControlsAvailability();
    updateRunnerControlsAvailability();

    // Initial status fetch for the default active tab
    if (currentOpenTab === 'TrainTab') fetchTrainerStatus();
    else if (currentOpenTab === 'RunTab') fetchRunnerStatus();
});