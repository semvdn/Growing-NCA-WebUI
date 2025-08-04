// static/js/ui.js

// --- Global State & Timers ---
let currentOpenTab = 'TrainTab'; 
let trainingStatusIntervalId = null;
let runningStatusIntervalId = null;

// --- Trainer Canvas Elements & State ---
const trainerDrawCanvasEl = document.getElementById('trainerDrawCanvas');
const trainerProgressCanvasEl = document.getElementById('trainerProgressCanvas');
let trainerProgressCtx = null;
if (trainerProgressCanvasEl) {
    trainerProgressCtx = trainerProgressCanvasEl.getContext('2d');
    trainerProgressCtx.imageSmoothingEnabled = false; // Crucial for pixelated scaling
}
const trainDrawColorPicker = document.getElementById('trainDrawColorPicker');
const clearTrainCanvasBtn = document.getElementById('clearTrainCanvasBtn');
const undoTrainCanvasBtn = document.getElementById('undoTrainCanvasBtn');
const redoTrainCanvasBtn = document.getElementById('redoTrainCanvasBtn');
const trainerDrawnImageNameInput = document.getElementById('trainerDrawnImageNameInput');
const confirmDrawingBtnTrain = document.getElementById('confirmDrawingBtnTrain');
const trainBrushSizeSlider = document.getElementById('trainBrushSizeSlider');
const trainBrushSizeValue = document.getElementById('trainBrushSizeValue');
const trainBrushOpacitySlider = document.getElementById('trainBrushOpacitySlider');
const trainBrushOpacityValue = document.getElementById('trainBrushOpacityValue');
let trainerCtx = null;
let isDrawingOnTrainerCanvas = false;
let trainerCanvasHistory = [];
let trainerCanvasHistoryPointer = -1;
let isEraserModeTrain = false;
let lastX = -1;
let lastY = -1;
let tempTrainerCanvas = null;
let tempTrainerCtx = null;
let currentStrokeBaseImageData = null;

// --- Runner Canvas & Interaction ---
const previewCanvasRunEl = document.getElementById('previewCanvasRun');
let previewCanvasRunCtx = null;
if (previewCanvasRunEl) {
    previewCanvasRunCtx = previewCanvasRunEl.getContext('2d');
    previewCanvasRunCtx.imageSmoothingEnabled = false;
}

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
const loadTrainerModelFileInput = document.getElementById('loadTrainerModelFileInput');
const loadTrainerModelBtn = document.getElementById('loadTrainerModelBtn');

// --- DOM Elements (Runner) ---
const modelFileInputRun = document.getElementById('modelFileInputRun');
const loadModelBtnRun = document.getElementById('loadModelBtnRun');
const loadCurrentTrainingModelBtnRun = document.getElementById('loadCurrentTrainingModelBtnRun');
const startRunningLoopBtn = document.getElementById('startRunningLoopBtn');
const stopRunningLoopBtn = document.getElementById('stopRunningLoopBtn');
const resetRunnerStateBtn = document.getElementById('resetRunnerStateBtn');
const runToolModeEraseRadio = document.getElementById('runToolModeErase');
const runToolModeDrawRadio = document.getElementById('runToolModeDraw');
const runDrawColorPicker = document.getElementById('runDrawColorPicker');
const runBrushSizeSlider = document.getElementById('runBrushSizeSlider');
const runBrushSizeValue = document.getElementById('runBrushSizeValue');
const runFpsSlider = document.getElementById('runFpsSlider');
const runFpsValue = document.getElementById('runFpsValue');
const enableEntropyTrainCheckbox = document.getElementById('enableEntropyTrain');
const entropyStrengthTrainSlider = document.getElementById('entropyStrengthTrain');
const entropyStrengthValueTrain = document.getElementById('entropyStrengthValueTrain');
const enableEntropyRunCheckbox = document.getElementById('enableEntropyRun');
const entropyStrengthRunSlider = document.getElementById('entropyStrengthRun');
const entropyStrengthValueRun = document.getElementById('entropyStrengthValueRun');
const runStatusDiv = document.getElementById('runStatus');
const runModelParamsText = document.getElementById('runModelParamsText');
const globalStatusMessageEl = document.getElementById('globalStatusMessage');

// --- Capture Tool Elements ---
const takeScreenshotTrainBtn = document.getElementById('takeScreenshotTrainBtn');
const startRecordingTrainBtn = document.getElementById('startRecordingTrainBtn');
const stopRecordingTrainBtn = document.getElementById('stopRecordingTrainBtn');
const recordingTimerTrain = document.getElementById('recordingTimerTrain');
const takeScreenshotRunBtn = document.getElementById('takeScreenshotRunBtn');
const startRecordingRunBtn = document.getElementById('startRecordingRunBtn');
const stopRecordingRunBtn = document.getElementById('stopRecordingRunBtn');
const recordingTimerRun = document.getElementById('recordingTimerRun');
const whiteBackgroundTrainCheckbox = document.getElementById('whiteBackgroundTrainCheckbox');
const whiteBackgroundRunCheckbox = document.getElementById('whiteBackgroundRunCheckbox');

// --- State Variables ---
let trainerInitialized = false;
let trainerTargetConfirmed = false;
let trainingLoopActive = false;
let runnerModelLoaded = false;
let runnerLoopActive = false;
let currentRunToolMode = 'erase';
let isInteractingWithRunCanvas = false;
let lastXRun = -1;
let lastYRun = -1;

// --- Constants ---
const DRAW_CANVAS_WIDTH = 512;
const DRAW_CANVAS_HEIGHT = 512;
const TARGET_CAPTURE_DIM = 720;


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

function drawUpscaledPixels(targetCtx, targetWidth, targetHeight, rawPixels, rawWidth, rawHeight) {
    if (!targetCtx || !rawPixels || rawWidth === 0 || rawHeight === 0) return;

    const tempLowResCanvas = document.createElement('canvas');
    tempLowResCanvas.width = rawWidth;
    tempLowResCanvas.height = rawHeight;
    const tempLowResCtx = tempLowResCanvas.getContext('2d');

    const imageData = tempLowResCtx.createImageData(rawWidth, rawHeight);
    const pixelDataArray = new Uint8ClampedArray(rawPixels);
    imageData.data.set(pixelDataArray);
    tempLowResCtx.putImageData(imageData, 0, 0);

    targetCtx.imageSmoothingEnabled = false;
    targetCtx.clearRect(0, 0, targetWidth, targetHeight);
    targetCtx.drawImage(tempLowResCanvas, 0, 0, targetWidth, targetHeight);
}

// --- UI Update Functions ---
function updateTrainerControlsAvailability() {
    confirmDrawingBtnTrain.disabled = trainingLoopActive;
    loadImageFileBtnTrain.disabled = trainingLoopActive;
    clearTrainCanvasBtn.disabled = trainingLoopActive;
    trainDrawColorPicker.disabled = trainingLoopActive || isEraserModeTrain;
    trainBrushSizeSlider.disabled = trainingLoopActive;
    trainBrushOpacitySlider.disabled = trainingLoopActive;
    undoTrainCanvasBtn.disabled = trainingLoopActive || trainerCanvasHistoryPointer <= 0;
    redoTrainCanvasBtn.disabled = trainingLoopActive || trainerCanvasHistoryPointer >= trainerCanvasHistory.length - 1;
    document.getElementById('trainEraserModeCheckbox').disabled = trainingLoopActive;
    
    initTrainerBtn.disabled = !trainerTargetConfirmed || trainingLoopActive;
    startTrainingBtn.disabled = !trainerInitialized || trainingLoopActive;
    stopTrainingBtn.disabled = !trainerInitialized || !trainingLoopActive;
    saveTrainerModelBtn.disabled = !trainerInitialized;
    loadCurrentTrainingModelBtnRun.disabled = !trainerInitialized || trainingLoopActive;
    loadTrainerModelBtn.disabled = trainingLoopActive;

    enableEntropyTrainCheckbox.disabled = trainingLoopActive;
    entropyStrengthTrainSlider.disabled = trainingLoopActive || !enableEntropyTrainCheckbox.checked;

    const isTrainCanvasVisible = trainerDrawCanvasEl.style.display !== 'none';
    const isTrainProgressCanvasVisible = trainerProgressCanvasEl.style.display !== 'none';
    takeScreenshotTrainBtn.disabled = !(isTrainCanvasVisible || isTrainProgressCanvasVisible);
    startRecordingTrainBtn.disabled = isRecording || !trainingLoopActive || !(isTrainCanvasVisible || isTrainProgressCanvasVisible);
    stopRecordingTrainBtn.disabled = !isRecording;
    recordingTimerTrain.style.display = isRecording ? 'inline' : 'none';
}

function updateRunnerControlsAvailability() {
    loadModelBtnRun.disabled = runnerLoopActive;
    loadCurrentTrainingModelBtnRun.disabled = runnerLoopActive || !trainerInitialized;
    startRunningLoopBtn.disabled = !runnerModelLoaded || runnerLoopActive;
    stopRunningLoopBtn.disabled = !runnerModelLoaded || !runnerLoopActive;
    resetRunnerStateBtn.disabled = !runnerModelLoaded;
    
    runBrushSizeSlider.disabled = !runnerModelLoaded;
    runToolModeEraseRadio.disabled = !runnerModelLoaded;
    runToolModeDrawRadio.disabled = !runnerModelLoaded;
    runDrawColorPicker.disabled = !runnerModelLoaded || currentRunToolMode !== 'draw';
    runFpsSlider.disabled = !runnerModelLoaded;

    enableEntropyRunCheckbox.disabled = !runnerModelLoaded;
    entropyStrengthRunSlider.disabled = !runnerModelLoaded || !enableEntropyRunCheckbox.checked;

    if (runnerModelLoaded) {
        previewCanvasRunEl.classList.toggle('erase-mode', currentRunToolMode === 'erase');
        previewCanvasRunEl.classList.toggle('draw-mode', currentRunToolMode === 'draw');
    } else {
        previewCanvasRunEl.className = '';
        previewCanvasRunEl.style.cursor = 'default';
    }

    takeScreenshotRunBtn.disabled = !runnerModelLoaded;
    startRecordingRunBtn.disabled = isRecording || !runnerLoopActive || !runnerModelLoaded;
    stopRecordingRunBtn.disabled = !isRecording;
    recordingTimerRun.style.display = isRecording ? 'inline' : 'none';
}