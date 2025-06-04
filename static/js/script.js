// static/js/script.js
let currentOpenTab = 'TrainTab'; 
let trainingStatusIntervalId = null;
let runningStatusIntervalId = null;

const previewCanvasTrainImgEl = document.getElementById('previewCanvasTrain');
const previewCanvasRunImgEl = document.getElementById('previewCanvasRun');

// ... (Other DOM Elements remain the same) ...
const loadEmojiBtnTrain = document.getElementById('loadEmojiBtnTrain');
const imageFileInputTrain = document.getElementById('imageFileInputTrain');
const loadImageFileBtnTrain = document.getElementById('loadImageFileBtnTrain');
const emojiInputTrain = document.getElementById('emojiInput'); 

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

const modelFileInputRun = document.getElementById('modelFileInputRun');
const loadModelBtnRun = document.getElementById('loadModelBtnRun');
const startRunningLoopBtn = document.getElementById('startRunningLoopBtn');
const stopRunningLoopBtn = document.getElementById('stopRunningLoopBtn');
const resetRunnerStateBtn = document.getElementById('resetRunnerStateBtn');
const rewindBtnRun = document.getElementById('rewindBtnRun');
const skipForwardBtnRun = document.getElementById('skipForwardBtnRun');

const toolModeEraseRadio = document.getElementById('toolModeErase');
const toolModeDrawRadio = document.getElementById('toolModeDraw');
const drawColorPickerRun = document.getElementById('drawColorPickerRun');
const brushSizeSliderRun = document.getElementById('brushSizeSliderRun');
const brushSizeValueRun = document.getElementById('brushSizeValueRun');

const fpsSliderRun = document.getElementById('fpsSliderRun');
const fpsValueRun = document.getElementById('fpsValueRun');

const runStatusDiv = document.getElementById('runStatus');
const runModelParamsText = document.getElementById('runModelParamsText');

const globalStatusMessageEl = document.getElementById('globalStatusMessage');


// --- State Variables ---
let trainerInitialized = false;
let trainingLoopActive = false;
let runnerModelLoaded = false;
let runnerLoopActive = false;
let currentToolMode = 'erase'; 
let isInteractingWithCanvas = false; // Flag for drag interactions

// --- Utility Functions (showGlobalStatus, postRequest, postFormRequest - same as before) ---
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

// --- UI Update Functions (updateTrainerControlsAvailability, updateRunnerControlsAvailability - same as before) ---
function updateTrainerControlsAvailability() {
    loadEmojiBtnTrain.disabled = trainingLoopActive;
    loadImageFileBtnTrain.disabled = trainingLoopActive;
    initTrainerBtn.disabled = trainingLoopActive; 
    startTrainingBtn.disabled = !trainerInitialized || trainingLoopActive;
    stopTrainingBtn.disabled = !trainerInitialized || !trainingLoopActive;
    saveTrainerModelBtn.disabled = !trainerInitialized; 
}
function updateRunnerControlsAvailability() {
    loadModelBtnRun.disabled = runnerLoopActive;
    startRunningLoopBtn.disabled = !runnerModelLoaded || runnerLoopActive;
    stopRunningLoopBtn.disabled = !runnerModelLoaded || !runnerLoopActive; 
    resetRunnerStateBtn.disabled = !runnerModelLoaded; 
    
    rewindBtnRun.disabled = !runnerModelLoaded; 
    skipForwardBtnRun.disabled = !runnerModelLoaded;
    brushSizeSliderRun.disabled = !runnerModelLoaded;
    toolModeEraseRadio.disabled = !runnerModelLoaded;
    toolModeDrawRadio.disabled = !runnerModelLoaded;
    drawColorPickerRun.disabled = !runnerModelLoaded || currentToolMode !== 'draw';
    fpsSliderRun.disabled = !runnerModelLoaded; 

    if (runnerModelLoaded) {
        previewCanvasRunImgEl.classList.toggle('erase-mode', currentToolMode === 'erase');
        previewCanvasRunImgEl.classList.toggle('draw-mode', currentToolMode === 'draw');
    } else {
        previewCanvasRunImgEl.className = ''; 
        previewCanvasRunImgEl.style.cursor = 'default';
    }
}

// --- Tab Management (same as before) ---
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
        if (trainerInitialized && !trainingStatusIntervalId) { 
             // trainingStatusIntervalId = setInterval(fetchTrainerStatus, 1200); 
        }
        fetchTrainerStatus(); 
        updateTrainerControlsAvailability();
    } else if (tabId === 'RunTab') {
        if (trainingStatusIntervalId) clearInterval(trainingStatusIntervalId); trainingStatusIntervalId = null;
        if (runnerModelLoaded && !runningStatusIntervalId) { 
            // runningStatusIntervalId = setInterval(fetchRunnerStatus, 300); 
        }
        fetchRunnerStatus(); 
        updateRunnerControlsAvailability();
    }
}

// --- Training Tab Logic (Same as previous good version) ---
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
        if (trainingStatusIntervalId) clearInterval(trainingStatusIntervalId); 
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
    fetchTrainerStatus(); 
    updateTrainerControlsAvailability();
});
saveTrainerModelBtn.addEventListener('click', async () => {
    if (!trainerInitialized) return; 
    const response = await postRequest('/save_trainer_model');
    showGlobalStatus(response.message, response.success);
});
async function fetchTrainerStatus() {
    if (currentOpenTab !== 'TrainTab' && trainingStatusIntervalId) { return; }
    if (!trainerInitialized && trainingStatusIntervalId) { 
        clearInterval(trainingStatusIntervalId); trainingStatusIntervalId = null;
        trainingStatusDiv.textContent = "Status: Trainer not initialized.";
        updateTrainerControlsAvailability(); return;
    }
    try {
        const response = await fetch('/get_training_status');
        if (!response.ok) { 
            trainingStatusDiv.textContent = `Trainer status error: ${response.status}`;
            trainingLoopActive = false; 
            updateTrainerControlsAvailability(); return; 
        }
        const data = await response.json();
        trainingStatusDiv.textContent = data.status_message || `Step: ${data.step || 0}, Loss: ${data.loss || 'N/A'}, Time: ${data.training_time || 'N/A'}`;
        if (data.preview_url) previewCanvasTrainImgEl.src = `${data.preview_url}?t=${new Date().getTime()}`;
        trainingLoopActive = data.is_training; 
        updateTrainerControlsAvailability();
    } catch (error) { 
        trainingStatusDiv.textContent = `Trainer status fetch error: ${error}. Polling might stop.`;
        trainingLoopActive = false;
        if (trainingStatusIntervalId) clearInterval(trainingStatusIntervalId); trainingStatusIntervalId = null;
        updateTrainerControlsAvailability();
    }
}

// --- Run Tab Logic ---
brushSizeSliderRun.addEventListener('input', () => {
    brushSizeValueRun.textContent = brushSizeSliderRun.value;
});
fpsSliderRun.addEventListener('input', async () => {
    const fps = parseInt(fpsSliderRun.value);
    fpsValueRun.textContent = fps;
    if (runnerModelLoaded) { 
        const response = await postRequest('/set_runner_speed', { fps: fps });
        if (response.success && runnerLoopActive && runningStatusIntervalId) { 
            clearInterval(runningStatusIntervalId);
            runningStatusIntervalId = setInterval(fetchRunnerStatus, Math.max(50, 1000 / fps));
        }
    }
});

toolModeEraseRadio.addEventListener('change', () => { 
    if (toolModeEraseRadio.checked) currentToolMode = 'erase';
    updateRunnerControlsAvailability(); 
});
toolModeDrawRadio.addEventListener('change', () => {
    if (toolModeDrawRadio.checked) currentToolMode = 'draw';
    updateRunnerControlsAvailability(); 
});


loadModelBtnRun.addEventListener('click', async () => {
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
        const initialFps = parseInt(fpsSliderRun.value);
        if (!runningStatusIntervalId && currentOpenTab === 'RunTab') {
             runningStatusIntervalId = setInterval(fetchRunnerStatus, Math.max(50, 1000 / initialFps)); 
        }
        runStatusDiv.textContent = "Status: Runner: Model loaded. FPS: " + initialFps;
        fpsSliderRun.dispatchEvent(new Event('input')); 
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
            runningStatusIntervalId = setInterval(fetchRunnerStatus, Math.max(50, 1000 / parseInt(fpsSliderRun.value))); 
        }
    }
    updateRunnerControlsAvailability();
});

stopRunningLoopBtn.addEventListener('click', async () => {
    if(!runnerModelLoaded) {
        showGlobalStatus("Runner: No model loaded to stop.", false); return;
    }
    console.log("Stop Running Loop button clicked by user. runnerLoopActive (client-side):", runnerLoopActive);
    const response = await postRequest('/stop_running');
    showGlobalStatus(response.message, response.success);
    // runnerLoopActive will be updated by fetchRunnerStatus
    fetchRunnerStatus(); 
    updateRunnerControlsAvailability();
});

resetRunnerStateBtn.addEventListener('click', async () => {
    if (!runnerModelLoaded) return;
    await handleRunnerAction('reset_runner'); 
    runnerLoopActive = false; 
    updateRunnerControlsAvailability();
});


rewindBtnRun.addEventListener('click', async () => handleRunnerAction('rewind'));
skipForwardBtnRun.addEventListener('click', async () => handleRunnerAction('skip_forward'));

async function handleRunnerAction(action, params = {}) {
    if (!runnerModelLoaded) return;
    const payload = { action, ...params };
    // console.log("handleRunnerAction sending:", payload); 
    const response = await postRequest('/runner_action', payload);
    
    if(response.success && response.preview_url) {
        previewCanvasRunImgEl.src = `${response.preview_url}?t=${new Date().getTime()}`; // Update preview
        const currentFPS = parseFloat(fpsSliderRun.value).toFixed(1); // Use client's current FPS for message
        runStatusDiv.textContent = `${response.message} (Target: ${currentFPS} FPS)`;
         if (action !== 'modify_area') { 
            // showGlobalStatus(response.message, response.success); // Can be too verbose
        }
    } else if (!response.success) {
        showGlobalStatus(response.message, false); 
    }
    // No need to call updateRunnerControlsAvailability if only preview/status text changes
    // unless action might change runnerLoopActive (e.g. reset could imply stop)
    if (action === 'reset_runner') {
        runnerLoopActive = false; // Reset implies loop stops
        updateRunnerControlsAvailability();
    }
}

// --- Canvas Interaction for Runner ---
function performCanvasAction(event, isDrag = false) {
    if (!runnerModelLoaded || currentOpenTab !== 'RunTab' ) return; 
    
    const rect = previewCanvasRunImgEl.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return; 

    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Prevent action if click is outside the image bounds (can happen with mouseleave quickly)
    if (x < 0 || x > rect.width || y < 0 || y > rect.height) {
        if (isDrag) isInteractingWithCanvas = false; // Stop drag if mouse leaves
        return;
    }

    const normX = Math.max(0, Math.min(1, x / rect.width)); 
    const normY = Math.max(0, Math.min(1, y / rect.height));
    
    const brushSliderVal = parseInt(brushSizeSliderRun.value); 
    const normBrushFactor = (brushSliderVal / 30) * 0.20 + 0.01; 

    // console.log(`  Tool: ${currentToolMode}, Color: ${drawColorPickerRun.value}, normX: ${normX.toFixed(2)}, normY: ${normY.toFixed(2)}, brushFactor: ${normBrushFactor.toFixed(2)}`);

    // Send the action to the backend.
    // The backend lock handles concurrency with the stepping loop.
    // For drag, we don't want to show a global status for every mousemove event.
    handleRunnerAction('modify_area', { 
        tool_mode: currentToolMode,
        draw_color_hex: drawColorPickerRun.value,
        norm_x: normX, 
        norm_y: normY, 
        brush_size_norm: normBrushFactor,
        canvas_render_width: rect.width, 
        canvas_render_height: rect.height
    });
}

previewCanvasRunImgEl.addEventListener('mousedown', (event) => {
    if (event.button !== 0) return; // Only main (left) click
    if (!runnerModelLoaded || currentOpenTab !== 'RunTab') return;
    isInteractingWithCanvas = true;
    performCanvasAction(event); // Perform action on initial click
    event.preventDefault(); // Prevent text selection or other default drag behaviors
});

previewCanvasRunImgEl.addEventListener('mousemove', (event) => {
    if (!isInteractingWithCanvas) return; // Only if mouse button is down
    performCanvasAction(event, true); // Perform action on drag
    event.preventDefault();
});

// Stop interaction when mouse button is released anywhere or mouse leaves canvas
document.addEventListener('mouseup', (event) => {
    if (event.button !== 0) return;
    if (isInteractingWithCanvas) {
        isInteractingWithCanvas = false;
    }
});
previewCanvasRunImgEl.addEventListener('mouseleave', () => {
    if (isInteractingWithCanvas) {
        isInteractingWithCanvas = false;
    }
});


async function fetchRunnerStatus() {
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
            runnerLoopActive = false;
            updateRunnerControlsAvailability(); return;
        }
        const data = await response.json();
        
        const serverTargetFPS = data.current_fps === "Max" ? "Max" : parseFloat(data.current_fps).toFixed(1);
        runStatusDiv.textContent = `${data.status_message || 'Status unavailable'} (Target: ${serverTargetFPS} FPS)`;
        
        // Only update preview if not currently interacting with canvas (to avoid jitter)
        // or if the loop is active (meaning backend is driving changes)
        if (!isInteractingWithCanvas || data.is_loop_active) {
            if (data.preview_url) previewCanvasRunImgEl.src = `${data.preview_url}?t=${new Date().getTime()}`;
        }
        
        const previousRunnerLoopActiveState = runnerLoopActive;
        runnerLoopActive = data.is_loop_active; 
        
        // Adjust polling interval
        if (runningStatusIntervalId) {
            clearInterval(runningStatusIntervalId); // Clear previous
            let newIntervalTime = 1000; 
            if (runnerLoopActive) {
                const targetFpsNum = parseFloat(data.current_fps);
                newIntervalTime = (targetFpsNum && targetFpsNum > 0) ? Math.max(33, 1000 / targetFpsNum) : 50;
            }
            runningStatusIntervalId = setInterval(fetchRunnerStatus, newIntervalTime);
        } else if (runnerModelLoaded && currentOpenTab === 'RunTab') { // If interval was cleared but should be running
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

// --- Initial Setup (Same as previous good version) ---
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('trainTabButton').click(); 
    
    const isRegen = experimentTypeSelectTrain.value === 'Regenerating';
    damageNInputTrain.style.display = isRegen ? 'block' : 'none';
    damageNLabelTrain.style.display = isRegen ? 'block' : 'none';
    brushSizeValueRun.textContent = brushSizeSliderRun.value;
    fpsValueRun.textContent = fpsSliderRun.value;
    currentToolMode = toolModeEraseRadio.checked ? 'erase' : 'draw';


    const placeholderDim = 256; 
    const placeholderColor = '#f0f0f0';
    const svgPlaceholder = (text) => `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='${placeholderDim}' height='${placeholderDim}' viewBox='0 0 ${placeholderDim} ${placeholderDim}'%3E%3Crect width='100%25' height='100%25' fill='${placeholderColor}'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' font-family='sans-serif' font-size='16' fill='%236c757d'%3E${text}%3C/text%3E%3C/svg%3E`;
    
    previewCanvasTrainImgEl.src = svgPlaceholder('Trainer Preview');
    previewCanvasRunImgEl.src = svgPlaceholder('Runner Preview');

    updateTrainerControlsAvailability();
    updateRunnerControlsAvailability();

    if (currentOpenTab === 'TrainTab') fetchTrainerStatus();
    else if (currentOpenTab === 'RunTab') fetchRunnerStatus();
});