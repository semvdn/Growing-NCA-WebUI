// static/js/script.js
let currentOpenTab = 'TrainTab'; 
let trainingStatusIntervalId = null;
let runningStatusIntervalId = null;

// --- Trainer Canvas Elements & State ---
const trainerDrawCanvasEl = document.getElementById('trainerDrawCanvas');
const trainerProgressCanvasEl = document.getElementById('trainerProgressCanvas'); // NEW
let trainerProgressCtx = null; // NEW
if (trainerProgressCanvasEl) { // NEW
    trainerProgressCtx = trainerProgressCanvasEl.getContext('2d'); // NEW
    trainerProgressCtx.imageSmoothingEnabled = false; // Crucial for pixelated scaling // NEW
} // NEW
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
let isEraserModeTrain = false; // New state variable for eraser mode
let lastX = -1; // To store the last X coordinate for continuous drawing
let lastY = -1; // To store the last Y coordinate for continuous drawing
let tempTrainerCanvas = null; // Temporary canvas for current stroke
let tempTrainerCtx = null; // Context for temporary canvas
let currentStrokeBaseImageData = null; // Stores the canvas state before the current stroke started

// --- Runner Canvas & Interaction ---
const previewCanvasRunEl = document.getElementById('previewCanvasRun'); // NEW
let previewCanvasRunCtx = null;
if (previewCanvasRunEl) {
    previewCanvasRunCtx = previewCanvasRunEl.getContext('2d');
    // For image-rendering: pixelated to work well when scaling,
    // ensure the context's image smoothing is off.
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

// New: Load Trainer Model elements
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
let lastXRun = -1; // New: To store the last X coordinate for continuous drawing on runner canvas
let lastYRun = -1; // New: To store the last Y coordinate for continuous drawing on runner canvas

// --- Capture State Variables ---
let mediaRecorder = null;
let recordedChunks = [];
let recordingTimerIntervalId = null;
let recordingStartTime = 0;
let pausedRecordingDuration = 0; // To accumulate time when NCA is paused
let isRecording = false;

// --- Constants ---
const DRAW_CANVAS_WIDTH = 512;
const DRAW_CANVAS_HEIGHT = 512;
const TARGET_CAPTURE_DIM = 720; // 10x upscale from 72x72 original resolution

// --- High-Res Capture Canvas (hidden) ---
let highResCaptureCanvas = null;
let highResCaptureCtx = null;


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

// --- Utility for Upscaled Pixel Drawing ---
function drawUpscaledPixels(targetCtx, targetWidth, targetHeight, rawPixels, rawWidth, rawHeight) {
    if (!targetCtx || !rawPixels || rawWidth === 0 || rawHeight === 0) return;

    // Create a temporary canvas for the original low-res image
    const tempLowResCanvas = document.createElement('canvas');
    tempLowResCanvas.width = rawWidth;
    tempLowResCanvas.height = rawHeight;
    const tempLowResCtx = tempLowResCanvas.getContext('2d');

    // Put the raw pixel data onto the temporary low-res canvas
    const imageData = tempLowResCtx.createImageData(rawWidth, rawHeight);
    const pixelDataArray = new Uint8ClampedArray(rawPixels);
    imageData.data.set(pixelDataArray);
    tempLowResCtx.putImageData(imageData, 0, 0);

    // Draw the low-res canvas onto the target canvas, scaled up
    targetCtx.imageSmoothingEnabled = false; // Crucial for pixelated scaling
    targetCtx.clearRect(0, 0, targetWidth, targetHeight);
    targetCtx.drawImage(tempLowResCanvas, 0, 0, targetWidth, targetHeight);
}

// --- Capture Logic ---
function captureCanvasAsImage(sourceElement, filenamePrefix) {
    let canvasToProcess = sourceElement; // The canvas we will potentially draw a background on
    let useWhiteBackground = false;

    // Determine which checkbox to check based on the sourceElement's ID
    if (sourceElement.id === 'trainerDrawCanvas' || sourceElement.id === 'trainerProgressCanvas') {
        useWhiteBackground = whiteBackgroundTrainCheckbox.checked;
    } else if (sourceElement.id === 'previewCanvasRun') {
        useWhiteBackground = whiteBackgroundRunCheckbox.checked;
    }

    // If a white background is requested, create a temporary canvas for processing
    if (useWhiteBackground) {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = sourceElement.naturalWidth || sourceElement.width;
        tempCanvas.height = sourceElement.naturalHeight || sourceElement.height;
        const tempCtx = tempCanvas.getContext('2d');

        // Fill with white background
        tempCtx.fillStyle = 'white';
        tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);

        // Draw the original source element onto this temporary canvas
        if (sourceElement.tagName === 'IMG') {
            tempCtx.drawImage(sourceElement, 0, 0, tempCanvas.width, tempCanvas.height);
        } else if (sourceElement.tagName === 'CANVAS') {
            // For canvas, if it's the runner preview, use the high-res capture canvas as source
            if (sourceElement.id === 'previewCanvasRun') {
                tempCtx.drawImage(highResCaptureCanvas, 0, 0, tempCanvas.width, tempCanvas.height);
            } else {
                tempCtx.drawImage(sourceElement, 0, 0);
            }
        } else {
            console.error('Unsupported element for capture:', sourceElement);
            return;
        }
        canvasToProcess = tempCanvas; // Use the temporary canvas for dataURL conversion
    } else {
        // If no white background, and it's the runner canvas, use the high-res capture canvas directly
        if (sourceElement.id === 'previewCanvasRun') {
            canvasToProcess = highResCaptureCanvas;
        } else {
            // For other canvases (trainer) without white background, create a temporary canvas and draw
            canvasToProcess = document.createElement('canvas');
            let ctx = canvasToProcess.getContext('2d');

            // Set canvas dimensions to match the source
            canvasToProcess.width = sourceElement.naturalWidth || sourceElement.width;
            canvasToProcess.height = sourceElement.naturalHeight || sourceElement.height;

            // Draw the image or canvas content onto the temporary canvas
            if (sourceElement.tagName === 'IMG') {
                ctx.drawImage(sourceElement, 0, 0, canvasToProcess.width, canvasToProcess.height);
            } else if (sourceElement.tagName === 'CANVAS') {
                ctx.drawImage(sourceElement, 0, 0);
            } else {
                console.error('Unsupported element for capture:', sourceElement);
                return;
            }
        }
    }

    const dataURL = canvasToProcess.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = dataURL;
    a.download = `${filenamePrefix}_${new Date().toISOString().slice(0,19).replace(/[-T:]/g, '')}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    showGlobalStatus('Screenshot captured!', true);
}

// --- Video Recording Logic ---
async function startRecording(canvasElement, tabName) {
    if (isRecording) {
        showGlobalStatus('Already recording.', false);
        return;
    }

    recordedChunks = [];
    let streamSourceCanvas = canvasElement;
    let useWhiteBackground = false;

    // Determine which checkbox to check based on the canvasElement's ID
    if (canvasElement.id === 'trainerDrawCanvas' || canvasElement.id === 'trainerProgressCanvas') {
        useWhiteBackground = whiteBackgroundTrainCheckbox.checked;
    } else if (canvasElement.id === 'previewCanvasRun') {
        useWhiteBackground = whiteBackgroundRunCheckbox.checked;
    }

    let stream;
    if (useWhiteBackground) {
        // Create a temporary canvas for drawing frames with a white background
        const tempRecordingCanvas = document.createElement('canvas');
        tempRecordingCanvas.width = canvasElement.naturalWidth || canvasElement.width;
        tempRecordingCanvas.height = canvasElement.naturalHeight || canvasElement.height;
        const tempRecordingCtx = tempRecordingCanvas.getContext('2d');

        // Function to draw a frame with white background
        const drawFrameWithWhiteBackground = () => {
            tempRecordingCtx.fillStyle = 'white';
            tempRecordingCtx.fillRect(0, 0, tempRecordingCanvas.width, tempRecordingCanvas.height);
            
            // Draw the original source element onto this temporary canvas
            if (canvasElement.tagName === 'IMG') {
                tempRecordingCtx.drawImage(canvasElement, 0, 0, tempRecordingCanvas.width, tempRecordingCanvas.height);
            } else if (canvasElement.tagName === 'CANVAS') {
                // For canvas, if it's the runner preview, use the high-res capture canvas as source
                if (canvasElement.id === 'previewCanvasRun') {
                    tempRecordingCtx.drawImage(highResCaptureCanvas, 0, 0, tempRecordingCanvas.width, tempRecordingCanvas.height);
                } else {
                    tempRecordingCtx.drawImage(canvasElement, 0, 0);
                }
            }
        };

        // Create a stream from the temporary canvas, drawing frames to it
        stream = tempRecordingCanvas.captureStream(60); // Capture at 60 FPS
        // Override the default draw function to include white background
        const originalDraw = stream.getVideoTracks()[0].requestFrame; // This might not be the correct way to hook into frame drawing for MediaRecorder
        
        // A more robust way is to continuously draw to the temp canvas and capture its stream
        let animationFrameId;
        const captureLoop = () => {
            drawFrameWithWhiteBackground();
            animationFrameId = requestAnimationFrame(captureLoop);
        };
        animationFrameId = requestAnimationFrame(captureLoop); // Start the drawing loop

        // When recording stops, stop the animation frame loop
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/mp4' });
        mediaRecorder.onstop = async () => {
            cancelAnimationFrame(animationFrameId); // Stop the drawing loop
            showGlobalStatus('Saving video...', true);
            const webmBlob = new Blob(recordedChunks, { type: 'video/webm' });
            const outputFilename = `NCA_${tabName}_Recording_${new Date().toISOString().slice(0,19).replace(/[-T:]/g, '')}.mp4`;

            // Create download link
            const a = document.createElement('a');
            a.href = URL.createObjectURL(webmBlob);
            a.download = outputFilename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(a.href); // Clean up URL object

            showGlobalStatus('Video recorded and downloaded!', true);
            recordedChunks = [];
            isRecording = false;
            updateTrainerControlsAvailability(); // Re-enable buttons
            updateRunnerControlsAvailability();
        };

    } else {
        // Original logic: capture stream directly from the source canvas
        if (canvasElement.id === 'previewCanvasRun') {
            streamSourceCanvas = highResCaptureCanvas;
        }
        stream = streamSourceCanvas.captureStream(60); // Capture at 60 FPS
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/mp4' });
        mediaRecorder.onstop = async () => {
            showGlobalStatus('Saving video...', true);
            const webmBlob = new Blob(recordedChunks, { type: 'video/webm' });
            const outputFilename = `NCA_${tabName}_Recording_${new Date().toISOString().slice(0,19).replace(/[-T:]/g, '')}.mp4`;

            // Create download link
            const a = document.createElement('a');
            a.href = URL.createObjectURL(webmBlob);
            a.download = outputFilename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(a.href); // Clean up URL object

            showGlobalStatus('Video recorded and downloaded!', true);
            recordedChunks = [];
            isRecording = false;
            updateTrainerControlsAvailability(); // Re-enable buttons
            updateRunnerControlsAvailability();
        };
    }

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };

    mediaRecorder.start();
    isRecording = true;
    recordingStartTime = Date.now();
    pausedRecordingDuration = 0; // Reset paused duration
    startRecordingTimer(tabName); // Start the timer
    showGlobalStatus('Recording started...', true);
    updateTrainerControlsAvailability(); // Disable start button
    updateRunnerControlsAvailability();
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        clearInterval(recordingTimerIntervalId);
        recordingTimerIntervalId = null;
        // UI updates will happen in mediaRecorder.onstop
    }
}

// --- Recording Timer Logic ---
function updateRecordingTimer(tabName) {
    let timerSpan = tabName === 'Train' ? recordingTimerTrain : recordingTimerRun;
    if (!isRecording) {
        timerSpan.textContent = '00:00';
        return;
    }

    let currentElapsed = (Date.now() - recordingStartTime) + pausedRecordingDuration;
    const minutes = Math.floor(currentElapsed / 60000);
    const seconds = Math.floor((currentElapsed % 60000) / 1000);
    timerSpan.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

function startRecordingTimer(tabName) {
    if (recordingTimerIntervalId) clearInterval(recordingTimerIntervalId);
    recordingTimerIntervalId = setInterval(() => updateRecordingTimer(tabName), 1000);
}

// --- UI Update Functions ---
function updateTrainerControlsAvailability() {
    confirmDrawingBtnTrain.disabled = trainingLoopActive;
    loadImageFileBtnTrain.disabled = trainingLoopActive;
    clearTrainCanvasBtn.disabled = trainingLoopActive;
    trainDrawColorPicker.disabled = trainingLoopActive || isEraserModeTrain; // Disable color picker in eraser mode
    trainBrushSizeSlider.disabled = trainingLoopActive;
    trainBrushOpacitySlider.disabled = trainingLoopActive;
    undoTrainCanvasBtn.disabled = trainingLoopActive || trainerCanvasHistoryPointer <= 0;
    redoTrainCanvasBtn.disabled = trainingLoopActive || trainerCanvasHistoryPointer >= trainerCanvasHistory.length - 1;
    document.getElementById('trainEraserModeCheckbox').disabled = trainingLoopActive; // Disable eraser checkbox during training
    
    initTrainerBtn.disabled = !trainerTargetConfirmed || trainingLoopActive;
    startTrainingBtn.disabled = !trainerInitialized || trainingLoopActive;
    stopTrainingBtn.disabled = !trainerInitialized || !trainingLoopActive;
    saveTrainerModelBtn.disabled = !trainerInitialized;
    loadCurrentTrainingModelBtnRun.disabled = !trainerInitialized || trainingLoopActive;
    loadTrainerModelBtn.disabled = trainingLoopActive; // New: Disable if training is active

    // Capture Tools
    const isTrainCanvasVisible = trainerDrawCanvasEl.style.display !== 'none';
    const isTrainProgressCanvasVisible = trainerProgressCanvasEl.style.display !== 'none'; // NEW
    takeScreenshotTrainBtn.disabled = !(isTrainCanvasVisible || isTrainProgressCanvasVisible); // MODIFIED
    startRecordingTrainBtn.disabled = isRecording || !trainingLoopActive || !(isTrainCanvasVisible || isTrainProgressCanvasVisible); // MODIFIED
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

    if (runnerModelLoaded) {
        previewCanvasRunEl.classList.toggle('erase-mode', currentRunToolMode === 'erase');
        previewCanvasRunEl.classList.toggle('draw-mode', currentRunToolMode === 'draw');
    } else {
        previewCanvasRunEl.className = '';
        previewCanvasRunEl.style.cursor = 'default';
    }

    // Capture Tools
    takeScreenshotRunBtn.disabled = !runnerModelLoaded; // Always available if model loaded
    startRecordingRunBtn.disabled = isRecording || !runnerLoopActive || !runnerModelLoaded;
    stopRecordingRunBtn.disabled = !isRecording;
    recordingTimerRun.style.display = isRecording ? 'inline' : 'none';
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
        if (runningStatusIntervalId) {
            clearInterval(runningStatusIntervalId);
            runningStatusIntervalId = null;
        }
        // Only start training status polling if trainer is initialized or training is active
        if ((trainerInitialized || trainingLoopActive) && !trainingStatusIntervalId) {
            trainingStatusIntervalId = setInterval(fetchTrainerStatus, 1200);
        }
        fetchTrainerStatus();
        updateTrainerControlsAvailability();
    } else if (tabId === 'RunTab') {
        if (trainingStatusIntervalId) { // Always clear training interval when leaving TrainTab
            clearInterval(trainingStatusIntervalId);
            trainingStatusIntervalId = null;
        }
        // Only start runner status polling if a model is loaded
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
    trainerCtx.clearRect(0, 0, DRAW_CANVAS_WIDTH, DRAW_CANVAS_HEIGHT);
    trainerDrawCanvasEl.classList.add('active-draw');
    trainerProgressCanvasEl.style.display = 'none'; // MODIFIED
    trainerDrawCanvasEl.style.display = 'inline-block';
    trainerTargetConfirmed = false;
    initTrainerBtn.disabled = true;
    updateTrainerControlsAvailability();
    trainingStatusDiv.textContent = "Status: Draw a pattern on the canvas or load an image file.";
    saveTrainerCanvasState(); // Save initial blank state

    // Initialize temporary canvas for drawing strokes
    if (!tempTrainerCanvas) {
        tempTrainerCanvas = document.createElement('canvas');
        tempTrainerCanvas.width = DRAW_CANVAS_WIDTH;
        tempTrainerCanvas.height = DRAW_CANVAS_HEIGHT;
        tempTrainerCtx = tempTrainerCanvas.getContext('2d');
    }
    tempTrainerCtx.clearRect(0, 0, DRAW_CANVAS_WIDTH, DRAW_CANVAS_HEIGHT); // Clear temp canvas
}

function clearTrainerDrawCanvas() {
    if (trainerCtx) {
        trainerCtx.clearRect(0, 0, trainerDrawCanvasEl.width, trainerDrawCanvasEl.height);
        trainerTargetConfirmed = false; 
        initTrainerBtn.disabled = true; 
        updateTrainerControlsAvailability();
        trainingStatusDiv.textContent = "Status: Drawing cleared. Draw a pattern or load a file.";
    }
}

function drawOnTrainerCanvas(event) {
    if (!isDrawingOnTrainerCanvas || trainingLoopActive || !tempTrainerCtx || !trainerCtx) return;

    const rect = trainerDrawCanvasEl.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const brushSize = parseInt(trainBrushSizeSlider.value);
    const brushOpacity = parseInt(trainBrushOpacitySlider.value) / 100;
    const color = trainDrawColorPicker.value;

    // The temporary canvas always draws opaque shapes.
    // The compositing and opacity are applied when drawing from temp to main canvas.
    tempTrainerCtx.globalCompositeOperation = 'source-over'; // Always draw opaque on temp canvas
    tempTrainerCtx.lineWidth = brushSize * 2;
    tempTrainerCtx.lineCap = 'round';
    tempTrainerCtx.lineJoin = 'round';

    const hexToRgb = (hex) => {
        const bigint = parseInt(hex.slice(1), 16);
        const r = (bigint >> 16) & 255;
        const g = (bigint >> 8) & 255;
        const b = bigint & 255;
        return `${r},${g},${b}`;
    };
    // Draw with full opacity on the temporary canvas, color doesn't matter for eraser
    tempTrainerCtx.strokeStyle = `rgb(${hexToRgb(color)})`;
    tempTrainerCtx.fillStyle = `rgb(${hexToRgb(color)})`;

    if (lastX === -1 || lastY === -1) { // First point of a new stroke
        tempTrainerCtx.beginPath();
        tempTrainerCtx.arc(x, y, brushSize, 0, Math.PI * 2);
        tempTrainerCtx.fill();
    } else { // Subsequent points, draw a line segment
        tempTrainerCtx.beginPath();
        tempTrainerCtx.moveTo(lastX, lastY);
        tempTrainerCtx.lineTo(x, y);
        tempTrainerCtx.stroke();
    }

    // Update last coordinates for the next segment
    lastX = x;
    lastY = y;

    // Restore the main canvas to its state *before* this stroke started
    trainerCtx.clearRect(0, 0, trainerDrawCanvasEl.width, trainerDrawCanvasEl.height);
    if (currentStrokeBaseImageData) {
        trainerCtx.putImageData(currentStrokeBaseImageData, 0, 0); // Draw base
    }

    // Apply global composite operation and opacity to the main canvas
    if (isEraserModeTrain) {
        trainerCtx.globalCompositeOperation = 'destination-out'; // Erase mode on main canvas
    } else {
        trainerCtx.globalCompositeOperation = 'source-over'; // Draw mode on main canvas
    }
    trainerCtx.globalAlpha = brushOpacity; // Apply overall opacity for the stroke

    // Draw the temporary canvas onto the main canvas
    trainerCtx.drawImage(tempTrainerCanvas, 0, 0);

    // Reset globalAlpha and globalCompositeOperation for future operations
    trainerCtx.globalAlpha = 1;
    trainerCtx.globalCompositeOperation = 'source-over';

    if (trainerTargetConfirmed) {
        trainingStatusDiv.textContent = "Status: Drawing modified. Confirm again to use as target.";
    }
    trainerTargetConfirmed = false;
    initTrainerBtn.disabled = true;
    updateTrainerControlsAvailability();
}

function saveTrainerCanvasState() {
    if (!trainerCtx) return;
    const imageData = trainerCtx.getImageData(0, 0, trainerDrawCanvasEl.width, trainerDrawCanvasEl.height);
    // If we are not at the end of history, clear future states
    if (trainerCanvasHistoryPointer < trainerCanvasHistory.length - 1) {
        trainerCanvasHistory = trainerCanvasHistory.slice(0, trainerCanvasHistoryPointer + 1);
    }
    trainerCanvasHistory.push(imageData);
    trainerCanvasHistoryPointer = trainerCanvasHistory.length - 1;
    updateTrainerControlsAvailability();
}

function restoreTrainerCanvasState(imageData) {
    if (!trainerCtx || !imageData) return;
    trainerCtx.putImageData(imageData, 0, 0);
    trainerTargetConfirmed = false;
    initTrainerBtn.disabled = true;
    updateTrainerControlsAvailability();
    trainingStatusDiv.textContent = "Status: Canvas state restored. Confirm again to use as target.";
}

undoTrainCanvasBtn.addEventListener('click', () => {
    if (trainingLoopActive || trainerCanvasHistoryPointer <= 0) return;
    trainerCanvasHistoryPointer--;
    restoreTrainerCanvasState(trainerCanvasHistory[trainerCanvasHistoryPointer]);
});

redoTrainCanvasBtn.addEventListener('click', () => {
    if (trainingLoopActive || trainerCanvasHistoryPointer >= trainerCanvasHistory.length - 1) return;
    trainerCanvasHistoryPointer++;
    restoreTrainerCanvasState(trainerCanvasHistory[trainerCanvasHistoryPointer]);
});

trainerDrawCanvasEl.addEventListener('mousedown', (e) => {
    if (e.button !== 0 || trainingLoopActive) return;
    isDrawingOnTrainerCanvas = true;
    const rect = trainerDrawCanvasEl.getBoundingClientRect();
    lastX = e.clientX - rect.left;
    lastY = e.clientY - rect.top;

    // Save the current state of the main canvas before starting the new stroke
    currentStrokeBaseImageData = trainerCtx.getImageData(0, 0, trainerDrawCanvasEl.width, trainerDrawCanvasEl.height);
    tempTrainerCtx.clearRect(0, 0, tempTrainerCanvas.width, tempTrainerCanvas.height); // Clear temp canvas for new stroke

    drawOnTrainerCanvas(e); // Draw the first point (dot)
    e.preventDefault();
});
trainerDrawCanvasEl.addEventListener('mousemove', (e) => {
    if (trainingLoopActive) return;
    if (isDrawingOnTrainerCanvas) {
        drawOnTrainerCanvas(e); // Draw continuous line segments
    }
    e.preventDefault();
});
document.addEventListener('mouseup', (e) => {
    if (e.button !== 0) return;
    if (isDrawingOnTrainerCanvas) {
        isDrawingOnTrainerCanvas = false;
        lastX = -1; // Reset last coordinates
        lastY = -1;
        currentStrokeBaseImageData = null; // Clear base image data
        saveTrainerCanvasState(); // Save the final state of trainerCtx
    }
});
trainerDrawCanvasEl.addEventListener('mouseleave', () => {
    if (isDrawingOnTrainerCanvas) {
        isDrawingOnTrainerCanvas = false;
        lastX = -1; // Reset last coordinates
        lastY = -1;
        currentStrokeBaseImageData = null; // Clear base image data
        saveTrainerCanvasState(); // Save state if mouse leaves while drawing
    }
});


clearTrainCanvasBtn.addEventListener('click', () => {
    if (trainingLoopActive) {
        showGlobalStatus("Cannot clear drawing while training is active.", false);
        return;
    }
    initializeTrainerDrawCanvas(); 
});

confirmDrawingBtnTrain.addEventListener('click', async () => {
    if (trainingLoopActive || !trainerCtx) return;
    const drawnImageName = trainerDrawnImageNameInput.value.trim();
    if (!drawnImageName) {
        showGlobalStatus("Please enter a name for your drawn image.", false);
        return;
    }

    const imageDataUrl = trainerDrawCanvasEl.toDataURL('image/png');
    const response = await postRequest('/upload_drawn_pattern_target', {
        image_data_url: imageDataUrl,
        drawn_image_name: drawnImageName
    });
    showGlobalStatus(response.message, response.success);
    if (response.success) {
        trainerTargetConfirmed = true;
        trainerDrawCanvasEl.style.display = 'none';
        // trainerProgressImgEl.src = `/get_trainer_target_preview?t=${new Date().getTime()}`; // REMOVED
        // trainerProgressImgEl.style.display = 'inline-block'; // REMOVED
        trainerProgressCanvasEl.style.display = 'inline-block'; // NEW
        // Fetch raw pixel data for the confirmed target and draw it
        try { // NEW
            const rawPreviewResponse = await fetch(`/get_trainer_target_raw_preview_data?t=${new Date().getTime()}`); // NEW
            if (rawPreviewResponse.ok) { // NEW
                const rawData = await rawPreviewResponse.json(); // NEW
                if (rawData.success && rawData.pixels && rawData.height > 0 && rawData.width > 0) { // NEW
                    drawUpscaledPixels(trainerProgressCtx, DRAW_CANVAS_WIDTH, DRAW_CANVAS_HEIGHT, rawData.pixels, rawData.width, rawData.height); // NEW
                } else { // NEW
                    console.error("Failed to get raw target preview data:", rawData.message); // NEW
                    trainerProgressCtx.clearRect(0, 0, trainerProgressCanvasEl.width, trainerProgressCanvasEl.height); // NEW
                    trainerProgressCtx.fillStyle = '#e0e0e0'; // NEW
                    trainerProgressCtx.fillRect(0, 0, trainerProgressCanvasEl.width, trainerProgressCanvasEl.height); // NEW
                    trainerProgressCtx.fillStyle = '#555'; // NEW
                    trainerProgressCtx.font = '20px sans-serif'; // NEW
                    trainerProgressCtx.textAlign = 'center'; // NEW
                    trainerProgressCtx.textBaseline = 'middle'; // NEW
                    trainerProgressCtx.fillText('Target Preview Unavailable', trainerProgressCanvasEl.width / 2, trainerProgressCanvasEl.height / 2); // NEW
                } // NEW
            } else { // NEW
                console.error("Failed to fetch raw target preview data. Response not OK:", rawPreviewResponse.status); // NEW
            } // NEW
        } catch (pixelError) { // NEW
            console.error("Error fetching or drawing raw target pixel data:", pixelError); // NEW
        } // NEW
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
        // trainerProgressImgEl.src = `/get_trainer_target_preview?t=${new Date().getTime()}`; // REMOVED
        // trainerProgressImgEl.style.display = 'inline-block'; // REMOVED
        trainerProgressCanvasEl.style.display = 'inline-block'; // NEW
        // Fetch raw pixel data for the confirmed target and draw it
        try { // NEW
            const rawPreviewResponse = await fetch(`/get_trainer_target_raw_preview_data?t=${new Date().getTime()}`); // NEW
            if (rawPreviewResponse.ok) { // NEW
                const rawData = await rawPreviewResponse.json(); // NEW
                if (rawData.success && rawData.pixels && rawData.height > 0 && rawData.width > 0) { // NEW
                    drawUpscaledPixels(trainerProgressCtx, DRAW_CANVAS_WIDTH, DRAW_CANVAS_HEIGHT, rawData.pixels, rawData.width, rawData.height); // NEW
                } else { // NEW
                    console.error("Failed to get raw target preview data:", rawData.message); // NEW
                    trainerProgressCtx.clearRect(0, 0, trainerProgressCanvasEl.width, trainerProgressCanvasEl.height); // NEW
                    trainerProgressCtx.fillStyle = '#e0e0e0'; // NEW
                    trainerProgressCtx.fillRect(0, 0, trainerProgressCanvasEl.width, trainerProgressCanvasEl.height); // NEW
                    trainerProgressCtx.fillStyle = '#555'; // NEW
                    trainerProgressCtx.font = '20px sans-serif'; // NEW
                    trainerProgressCtx.textAlign = 'center'; // NEW
                    trainerProgressCtx.textBaseline = 'middle'; // NEW
                    trainerProgressCtx.fillText('Target Preview Unavailable', trainerProgressCanvasEl.width / 2, trainerProgressCanvasEl.height / 2); // NEW
                } // NEW
            } else { // NEW
                console.error("Failed to fetch raw target preview data. Response not OK:", rawPreviewResponse.status); // NEW
            } // NEW
        } catch (pixelError) { // NEW
            console.error("Error fetching or drawing raw target pixel data:", pixelError); // NEW
        } // NEW
        
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
    formData.append('image_filename', imageFileInputTrain.files[0].name); // Pass filename explicitly
    handleLoadTargetForTrainerFromFile(formData);
});

initTrainerBtn.addEventListener('click', async () => {
    if (!trainerTargetConfirmed) {
        showGlobalStatus("Please confirm a drawn target or load a file first.", false); return;
    }
    trainerDrawCanvasEl.style.display = 'none';
    // trainerProgressImgEl.style.display = 'inline-block'; // REMOVED
    trainerProgressCanvasEl.style.display = 'inline-block'; // NEW

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
        // trainerProgressImgEl.src = `${response.initial_state_preview_url}?t=${new Date().getTime()}`; // REMOVED
        trainerProgressCanvasEl.style.display = 'inline-block'; // NEW
        trainerDrawCanvasEl.style.display = 'none'; // NEW
        // Fetch raw pixel data for the initial state and draw it
        try { // NEW
            const rawPreviewResponse = await fetch(`/get_live_trainer_raw_preview_data?t=${new Date().getTime()}`); // NEW
            if (rawPreviewResponse.ok) { // NEW
                const rawData = await rawPreviewResponse.json(); // NEW
                if (rawData.success && rawData.pixels && rawData.height > 0 && rawData.width > 0) { // NEW
                    drawUpscaledPixels(trainerProgressCtx, DRAW_CANVAS_WIDTH, DRAW_CANVAS_HEIGHT, rawData.pixels, rawData.width, rawData.height); // NEW
                } else { // NEW
                    console.error("Failed to get raw initial state preview data:", rawData.message); // NEW
                    trainerProgressCtx.clearRect(0, 0, trainerProgressCanvasEl.width, trainerProgressCanvasEl.height); // NEW
                    trainerProgressCtx.fillStyle = '#e0e0e0'; // NEW
                    trainerProgressCtx.fillRect(0, 0, trainerProgressCanvasEl.width, trainerProgressCanvasEl.height); // NEW
                    trainerProgressCtx.fillStyle = '#555'; // NEW
                    trainerProgressCtx.font = '20px sans-serif'; // NEW
                    trainerProgressCtx.textAlign = 'center'; // NEW
                    trainerProgressCtx.textBaseline = 'middle'; // NEW
                    trainerProgressCtx.fillText('Initial State Unavailable', trainerProgressCanvasEl.width / 2, trainerProgressCanvasEl.height / 2); // NEW
                } // NEW
            } else { // NEW
                console.error("Failed to fetch raw initial state preview data. Response not OK:", rawPreviewResponse.status); // NEW
            } // NEW
        } catch (pixelError) { // NEW
            console.error("Error fetching or drawing raw initial state pixel data:", pixelError); // NEW
        } // NEW
        trainerInitialized = true;
        trainingLoopActive = false;
        if (trainingStatusIntervalId) clearInterval(trainingStatusIntervalId);
        if (!trainingStatusIntervalId && currentOpenTab === 'TrainTab') {
             trainingStatusIntervalId = setInterval(fetchTrainerStatus, 1200);
        }
        trainingStatusDiv.textContent = "Status: Trainer Initialized.";
    } else {
        trainerInitialized = false;
        // Revert to drawing if init failed and drawn pattern was the source
        // initializeTrainerDrawCanvas(); // This might be too aggressive, user might want to see the failed target
        trainingStatusDiv.textContent = "Status: Trainer initialization failed. Check target or settings.";

    }
    updateTrainerControlsAvailability();
});

startTrainingBtn.addEventListener('click', async () => {
    if (!trainerInitialized) return;
    const response = await postRequest('/start_training');
    showGlobalStatus(response.message, response.success);
    if (response.success) {
        trainingLoopActive = true;
        trainerDrawCanvasEl.style.display = 'none';
        trainerProgressCanvasEl.style.display = 'inline-block'; // MODIFIED
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
    if (trainingStatusIntervalId) { // Explicitly clear interval when training stops
        clearInterval(trainingStatusIntervalId);
        trainingStatusIntervalId = null;
    }
    fetchTrainerStatus();
    updateTrainerControlsAvailability();
});

saveTrainerModelBtn.addEventListener('click', async () => {
    if (!trainerInitialized) return;
    const response = await postRequest('/save_trainer_model');
    // Update message to reflect checkpointing
    showGlobalStatus(response.message || "Checkpoint saved!", response.success);
});

loadTrainerModelBtn.addEventListener('click', async () => {
    if (trainingLoopActive) {
        showGlobalStatus("Cannot load model while training is active. Please stop training first.", false);
        return;
    }
    if (!loadTrainerModelFileInput.files.length) {
        showGlobalStatus('Please select a .weights.h5 model file to load.', false);
        return;
    }
    const formData = new FormData();
    formData.append('model_file', loadTrainerModelFileInput.files[0]);

    const response = await postFormRequest('/load_trainer_model', formData);
    showGlobalStatus(response.message, response.success);
    if (response.success) {
        trainerInitialized = true;
        trainingLoopActive = false; // Ensure training loop is not active after loading
        if (trainingStatusIntervalId) clearInterval(trainingStatusIntervalId);
        if (!trainingStatusIntervalId && currentOpenTab === 'TrainTab') {
            trainingStatusIntervalId = setInterval(fetchTrainerStatus, 1200);
        }
        trainModelParamsText.textContent = response.model_summary || 'N/A';
        trainingStatusDiv.textContent = "Status: Trainer model loaded. Ready to continue training.";
        // Optionally, update other UI elements based on loaded metadata if needed
        // e.g., experimentTypeSelectTrain.value = response.metadata.experiment_type;
    } else {
        trainerInitialized = false;
        trainingStatusDiv.textContent = "Status: Failed to load trainer model.";
    }
    updateTrainerControlsAvailability();
});

async function fetchTrainerStatus() {
    if (currentOpenTab !== 'TrainTab' && trainingStatusIntervalId) { return; }
    
    if (!trainerInitialized && !trainerTargetConfirmed ) { 
        if (trainingStatusIntervalId) clearInterval(trainingStatusIntervalId); 
        trainingStatusIntervalId = null;
        // initializeTrainerDrawCanvas(); // Ensure drawing canvas is shown if completely reset
        // trainingStatusDiv.textContent = "Status: Define a target pattern.";
        // updateTrainerControlsAvailability(); 
        return; // Don't poll if nothing is set up
    }

    try {
        const response = await fetch('/get_training_status');
        if (!response.ok) { 
            trainingStatusDiv.textContent = `Trainer status error: ${response.status}`;
            updateTrainerControlsAvailability(); return; 
        }
        const data = await response.json();
        // Include elapsed time in the status message
        trainingStatusDiv.textContent = data.status_message || `Step: ${data.step || 0}, Loss: ${data.loss || 'N/A'}, Time: ${data.training_time || 'N/A'}`;
        
        if (data.is_training || (trainerInitialized && trainerTargetConfirmed)) {
            trainerDrawCanvasEl.style.display = 'none';
            trainerProgressCanvasEl.style.display = 'inline-block'; // MODIFIED
            // if (data.preview_url) trainerProgressImgEl.src = `${data.preview_url}?t=${new Date().getTime()}`; // REMOVED
            // === Phase 2 Change: Add raw data fetching here === // NEW
            try { // NEW
                const rawPreviewResponse = await fetch(`/get_live_trainer_raw_preview_data?t=${new Date().getTime()}`); // NEW
                if (rawPreviewResponse.ok) { // NEW
                    const rawData = await rawPreviewResponse.json(); // NEW
                    if (rawData.success && rawData.pixels && rawData.height > 0 && rawData.width > 0) { // NEW
                        drawUpscaledPixels(trainerProgressCtx, DRAW_CANVAS_WIDTH, DRAW_CANVAS_HEIGHT, rawData.pixels, rawData.width, rawData.height); // NEW
                    } else if (!rawData.success && trainerProgressCtx) { // NEW
                        // Draw placeholder if trainer is not ready but model is loaded
                        trainerProgressCtx.fillStyle = '#e0e0e0'; // NEW
                        trainerProgressCtx.fillRect(0, 0, trainerProgressCanvasEl.width, trainerProgressCanvasEl.height); // NEW
                        trainerProgressCtx.fillStyle = '#555'; // NEW
                        trainerProgressCtx.font = '20px sans-serif'; // NEW
                        trainerProgressCtx.textAlign = 'center'; // NEW
                        trainerProgressCtx.textBaseline = 'middle'; // NEW
                        trainerProgressCtx.fillText('Training State Unavailable', trainerProgressCanvasEl.width / 2, trainerProgressCanvasEl.height / 2); // NEW
                    } // NEW
                } else { // NEW
                    console.error("Failed to fetch raw trainer preview data. Response not OK:", rawPreviewResponse.status); // NEW
                } // NEW
            } catch (pixelError) { // NEW
                console.error("Error fetching or drawing raw trainer pixel data:", pixelError); // NEW
            } // NEW
            // === End of Phase 2 Change === // NEW
        } else if (trainerTargetConfirmed && !trainerInitialized) { // Target confirmed, but not initialized yet
            trainerDrawCanvasEl.style.display = 'none';
            trainerProgressCanvasEl.style.display = 'inline-block'; // MODIFIED
            // trainerProgressImgEl.src = `/get_trainer_target_preview?t=${new Date().getTime()}`; // Show confirmed target // REMOVED
            // Fetch raw pixel data for the confirmed target and draw it // NEW
            try { // NEW
                const rawPreviewResponse = await fetch(`/get_trainer_target_raw_preview_data?t=${new Date().getTime()}`); // NEW
                if (rawPreviewResponse.ok) { // NEW
                    const rawData = await rawPreviewResponse.json(); // NEW
                    if (rawData.success && rawData.pixels && rawData.height > 0 && rawData.width > 0) { // NEW
                        drawUpscaledPixels(trainerProgressCtx, DRAW_CANVAS_WIDTH, DRAW_CANVAS_HEIGHT, rawData.pixels, rawData.width, rawData.height); // NEW
                    } else { // NEW
                        console.error("Failed to get raw target preview data:", rawData.message); // NEW
                        trainerProgressCtx.clearRect(0, 0, trainerProgressCanvasEl.width, trainerProgressCanvasEl.height); // NEW
                        trainerProgressCtx.fillStyle = '#e0e0e0'; // NEW
                        trainerProgressCtx.fillRect(0, 0, trainerProgressCanvasEl.width, trainerProgressCanvasEl.height); // NEW
                        trainerProgressCtx.fillStyle = '#555'; // NEW
                        trainerProgressCtx.font = '20px sans-serif'; // NEW
                        trainerProgressCtx.textAlign = 'center'; // NEW
                        trainerProgressCtx.textBaseline = 'middle'; // NEW
                        trainerProgressCtx.fillText('Target Preview Unavailable', trainerProgressCanvasEl.width / 2, trainerProgressCanvasEl.height / 2); // NEW
                    } // NEW
                } else { // NEW
                    console.error("Failed to fetch raw target preview data. Response not OK:", rawPreviewResponse.status); // NEW
                } // NEW
            } catch (pixelError) { // NEW
                console.error("Error fetching or drawing raw target pixel data:", pixelError); // NEW
            } // NEW
        } else { // No target confirmed, not initialized
            initializeTrainerDrawCanvas();
        }

        const prevTrainingLoopActive = trainingLoopActive;
        trainingLoopActive = data.is_training;
        updateTrainerControlsAvailability();

        // Handle recording timer based on training loop activity
        if (isRecording) {
            if (trainingLoopActive && !prevTrainingLoopActive) { // Loop just became active, resume timer
                recordingStartTime = Date.now(); // Reset start time to calculate from now
                startRecordingTimer('Train');
            } else if (!trainingLoopActive && prevTrainingLoopActive) { // Loop just became inactive, pause timer
                clearInterval(recordingTimerIntervalId);
                pausedRecordingDuration += (Date.now() - recordingStartTime); // Accumulate paused time
            }
        }

        // If training just stopped, ensure interval is cleared
        if (prevTrainingLoopActive && !trainingLoopActive && trainingStatusIntervalId) {
            clearInterval(trainingStatusIntervalId);
            trainingStatusIntervalId = null;
        } else if (trainingLoopActive && !trainingStatusIntervalId && currentOpenTab === 'TrainTab'){
            // Restart polling if it somehow stopped but should be active
            trainingStatusIntervalId = setInterval(fetchTrainerStatus, 1200);
        }
        // If training is not active and not initialized, ensure polling stops
        if (!data.is_training && !trainerInitialized && trainingStatusIntervalId) {
            clearInterval(trainingStatusIntervalId);
            trainingStatusIntervalId = null;
        }


    } catch (error) { 
        trainingStatusDiv.textContent = `Trainer status fetch error: ${error}.`;
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
        if (runnerLoopActive && runningStatusIntervalId) { 
            clearInterval(runningStatusIntervalId);
            runningStatusIntervalId = setInterval(fetchRunnerStatus, Math.max(33, 1000 / fps)); // Min 33ms interval
        } else if (!runnerLoopActive && runningStatusIntervalId) { // If paused, also adjust potential slower poll
            clearInterval(runningStatusIntervalId);
            runningStatusIntervalId = setInterval(fetchRunnerStatus, 1000); // Slower poll for paused
        }
    }
});
runToolModeEraseRadio.addEventListener('change', () => { if (runToolModeEraseRadio.checked) currentRunToolMode = 'erase'; updateRunnerControlsAvailability(); });
runToolModeDrawRadio.addEventListener('change', () => { if (runToolModeDrawRadio.checked) currentRunToolMode = 'draw'; updateRunnerControlsAvailability(); });

loadCurrentTrainingModelBtnRun.addEventListener('click', async () => {
    if (!trainerInitialized) {
        showGlobalStatus("Trainer model not available (trainer not initialized).", false); return;
    }
    // No need to check trainingLoopActive on client, backend can handle copying weights from non-stepping trainer.
    const response = await postRequest('/load_current_training_model_for_runner');
    showGlobalStatus(response.message, response.success);
    if (response.success) {
        runModelParamsText.textContent = response.model_summary || 'N/A';
        runnerModelLoaded = true;
        runnerLoopActive = false; 
        if (runningStatusIntervalId) clearInterval(runningStatusIntervalId); 
        const initialFps = parseInt(runFpsSlider.value);
        if (!runningStatusIntervalId && currentOpenTab === 'RunTab') {
             runningStatusIntervalId = setInterval(fetchRunnerStatus, Math.max(50, 1000 / initialFps)); 
        }
        runStatusDiv.textContent = "Status: Runner: Loaded current training model. FPS: " + initialFps;
        runFpsSlider.dispatchEvent(new Event('input')); 
    }
    updateRunnerControlsAvailability();
});

loadModelBtnRun.addEventListener('click', async () => {
    const formData = new FormData();
    if (modelFileInputRun.files.length > 0) formData.append('model_file', modelFileInputRun.files[0]);
    const response = await postFormRequest('/load_model_for_runner', formData);
    showGlobalStatus(response.message, response.success);
    if (response.success) {
        let modelInfoText = response.model_summary || 'N/A';
        if (response.metadata) {
            modelInfoText += `\n--- Metadata ---\n`;
            modelInfoText += `Trained on: ${response.metadata.trained_on_image || 'N/A'}\n`;
            modelInfoText += `Steps: ${response.metadata.training_steps || 'N/A'}\n`;
            modelInfoText += `Experiment: ${response.metadata.experiment_type || 'N/A'}\n`;
            modelInfoText += `Saved: ${response.metadata.save_datetime || 'N/A'}\n`;
        }
        runModelParamsText.textContent = modelInfoText; // Updated to display metadata
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
startRunningLoopBtn.addEventListener('click', async () => {
    if (!runnerModelLoaded) return;
    const response = await postRequest('/start_running');
    showGlobalStatus(response.message, response.success);
    if (response.success) {
        runnerLoopActive = true;
        if (!runningStatusIntervalId && currentOpenTab === 'RunTab') { 
             const fps = parseInt(runFpsSlider.value);
            runningStatusIntervalId = setInterval(fetchRunnerStatus, Math.max(33, 1000 / fps)); 
        }
    }
    updateRunnerControlsAvailability();
});
stopRunningLoopBtn.addEventListener('click', async () => {
    if(!runnerModelLoaded) {
        showGlobalStatus("Runner: No model loaded to stop.", false); return;
    }
    const response = await postRequest('/stop_running');
    showGlobalStatus(response.message, response.success);
    // Server confirms loop stop. fetchRunnerStatus will update runnerLoopActive.
    fetchRunnerStatus(); 
    updateRunnerControlsAvailability();
});
resetRunnerStateBtn.addEventListener('click', async () => {
    if (!runnerModelLoaded) return;
    await handleRunnerAction('reset_runner'); 
    runnerLoopActive = false; 
    updateRunnerControlsAvailability();
});
async function handleRunnerAction(action, params = {}) {
    if (!runnerModelLoaded) return;
    const payload = { action, ...params };
    const response = await postRequest('/runner_action', payload);
    if(response.success && response.preview_url) {
        const currentFPS = parseFloat(runFpsSlider.value).toFixed(1);
        runStatusDiv.textContent = `${response.message} (Target: ${currentFPS} FPS)`;
         if (action !== 'modify_area') { /* showGlobalStatus for non-drag actions */ }
    } else if (!response.success) { showGlobalStatus(response.message, false); }
    if (action === 'reset_runner') { runnerLoopActive = false; } 
    updateRunnerControlsAvailability(); 
}
function performCanvasAction(event) { // Removed isDrag parameter
    if (!runnerModelLoaded || currentOpenTab !== 'RunTab') return;

    const rect = previewCanvasRunEl.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Check if mouse is within canvas bounds
    if (x < 0 || x > rect.width || y < 0 || y > rect.height) {
        // If mouse leaves canvas during drag, stop interaction and reset last coordinates
        if (isInteractingWithRunCanvas) {
            isInteractingWithRunCanvas = false;
            lastXRun = -1;
            lastYRun = -1;
        }
        return;
    }

    const brushSize = parseInt(runBrushSizeSlider.value);
    const drawColor = runDrawColorPicker.value;

    const normX = Math.max(0, Math.min(1, x / rect.width));
    const normY = Math.max(0, Math.min(1, y / rect.height));
    const normBrushFactor = (brushSize / 30) * 0.20 + 0.01;
    handleRunnerAction('modify_area', {
        tool_mode: currentRunToolMode,
        draw_color_hex: drawColor,
        norm_x: normX, norm_y: normY, brush_size_norm: normBrushFactor,
        canvas_render_width: rect.width, canvas_render_height: rect.height
    });
}

previewCanvasRunEl.addEventListener('mousedown', (event) => {
    if (event.button !== 0) return;
    if (!runnerModelLoaded || currentOpenTab !== 'RunTab') return;
    isInteractingWithRunCanvas = true;
    // Set initial lastXRun, lastYRun for the first point of the stroke
    const rect = previewCanvasRunEl.getBoundingClientRect();
    lastXRun = event.clientX - rect.left;
    lastYRun = event.clientY - rect.top;
    performCanvasAction(event); // Draw the first point (dot)
    event.preventDefault();
});
previewCanvasRunEl.addEventListener('mousemove', (event) => {
    if (!isInteractingWithRunCanvas) return;
    performCanvasAction(event); // Draw continuous line segments
    event.preventDefault();
});
document.addEventListener('mouseup', (event) => {
    if (event.button !== 0) return;
    if (isInteractingWithRunCanvas) {
        isInteractingWithRunCanvas = false;
        lastXRun = -1; // Reset last coordinates
        lastYRun = -1;
    }
});
previewCanvasRunEl.addEventListener('mouseleave', () => {
    if (isInteractingWithRunCanvas) {
        isInteractingWithRunCanvas = false;
        lastXRun = -1; // Reset last coordinates
        lastYRun = -1;
    }
});

async function fetchRunnerStatus() {
    if (currentOpenTab !== 'RunTab' && runningStatusIntervalId) { return; }
    if (!runnerModelLoaded && runningStatusIntervalId) {
        clearInterval(runningStatusIntervalId); runningStatusIntervalId = null;
        runStatusDiv.textContent = "Status: Runner: No model loaded.";
        if (previewCanvasRunCtx) { // Clear canvas if no model
            previewCanvasRunCtx.clearRect(0, 0, previewCanvasRunEl.width, previewCanvasRunEl.height);
            // Optionally draw a placeholder here too
        }
        updateRunnerControlsAvailability(); return;
    }
    try {
        const response = await fetch('/get_runner_status');
        if (!response.ok) {
            runStatusDiv.textContent = `Runner status error: ${response.status}`;
            runnerLoopActive = false; updateRunnerControlsAvailability(); return;
        }
        const data = await response.json();
        
        let statusMessage = data.status_message || 'Status unavailable';
        // Remove "Step: X/Y, " or "Step: X/Y" from the beginning of the status message
        statusMessage = statusMessage.replace(/^Step: \d+\/\d+,?\s*/, '');

        const targetFPSDisplay = data.current_fps === "Max" ? "Max" : parseFloat(data.current_fps).toFixed(1);
        const actualFPSDisplay = data.actual_fps || "N/A"; // Get actual_fps from data
        runStatusDiv.textContent = `${statusMessage} (Target: ${targetFPSDisplay} FPS, Actual: ${actualFPSDisplay} FPS)`;
        runStatusDiv.textContent = `${data.status_message || 'Status unavailable'} (Target: ${targetFPSDisplay} FPS, Actual: ${actualFPSDisplay} FPS)`;

        // Preview update logic will be changed in Phase 2
        // For now, the old logic would be:
        // if (!isInteractingWithRunCanvas || data.is_loop_active) {
        //     if (data.preview_url) previewCanvasRunImgEl.src = `${data.preview_url}?t=${new Date().getTime()}`;
        // }
        // This will be replaced by raw data fetching below.

        const prevRunnerLoopActive = runnerLoopActive;
        runnerLoopActive = data.is_loop_active;

        // Handle recording timer (remains the same)
        if (isRecording) {
            if (runnerLoopActive && !prevRunnerLoopActive) {
                recordingStartTime = Date.now();
                startRecordingTimer('Run');
            } else if (!runnerLoopActive && prevRunnerLoopActive) {
                clearInterval(recordingTimerIntervalId);
                pausedRecordingDuration += (Date.now() - recordingStartTime);
            }
        }

        if (runningStatusIntervalId) { // Only adjust if an interval is already supposed to be running
            clearInterval(runningStatusIntervalId);
            let newIntervalTime = 1000;
            if (runnerLoopActive) {
                const targetFpsNum = parseFloat(data.current_fps); // Use current_fps from status data
                newIntervalTime = (targetFpsNum && targetFpsNum > 0) ? Math.max(33, 1000 / targetFpsNum) : 50;
            } // else if paused/stopped, it remains 1000ms
            runningStatusIntervalId = setInterval(fetchRunnerStatus, newIntervalTime);
        } else if (runnerModelLoaded && currentOpenTab === 'RunTab' && !runningStatusIntervalId) {
             // If polling stopped but should be active (e.g. after tab switch)
             const targetFpsNum = parseFloat(data.current_fps);
             const intervalTime = (runnerLoopActive && targetFpsNum && targetFpsNum > 0) ? Math.max(33, 1000/targetFpsNum) : 1000;
             runningStatusIntervalId = setInterval(fetchRunnerStatus, intervalTime);
        }
        updateRunnerControlsAvailability();

        // === Phase 2 Change: Add raw data fetching here ===
        if (runnerModelLoaded && (!isInteractingWithRunCanvas || data.is_loop_active)) {
            try {
                const rawPreviewResponse = await fetch(`/get_live_runner_raw_preview_data?t=${new Date().getTime()}`);
                if (rawPreviewResponse.ok) {
                    const rawData = await rawPreviewResponse.json();
                    if (rawData.success && rawData.pixels && rawData.height > 0 && rawData.width > 0) {
                        if (previewCanvasRunCtx) {
                            // Ensure the display canvas is always DRAW_CANVAS_WIDTH x DRAW_CANVAS_HEIGHT for consistent UI
                            if (previewCanvasRunEl.width !== DRAW_CANVAS_WIDTH || previewCanvasRunEl.height !== DRAW_CANVAS_HEIGHT) {
                                previewCanvasRunEl.width = DRAW_CANVAS_WIDTH;
                                previewCanvasRunEl.height = DRAW_CANVAS_HEIGHT;
                            }
                            // Draw the raw pixels upscaled to the display canvas (512x512)
                            drawUpscaledPixels(previewCanvasRunCtx, DRAW_CANVAS_WIDTH, DRAW_CANVAS_HEIGHT, rawData.pixels, rawData.width, rawData.height);
                        }

                        // Draw to the high-res capture canvas (720x720)
                        if (highResCaptureCtx) {
                            drawUpscaledPixels(highResCaptureCtx, TARGET_CAPTURE_DIM, TARGET_CAPTURE_DIM, rawData.pixels, rawData.width, rawData.height);
                        }
                    } else if (!rawData.success && previewCanvasRunCtx) {
                        // Draw placeholder if runner is not ready but model is loaded
                        previewCanvasRunCtx.fillStyle = '#e0e0e0';
                        previewCanvasRunCtx.fillRect(0, 0, previewCanvasRunEl.width, previewCanvasRunEl.height);
                        previewCanvasRunCtx.fillStyle = '#555';
                        previewCanvasRunCtx.font = '20px sans-serif';
                        previewCanvasRunCtx.textAlign = 'center';
                        previewCanvasRunCtx.textBaseline = 'middle';
                        previewCanvasRunCtx.fillText('Runner State Unavailable', previewCanvasRunEl.width / 2, previewCanvasRunEl.height / 2);
                    }
                } else {
                    console.error("Failed to fetch raw preview data. Response not OK:", rawPreviewResponse.status);
                }
            } catch (pixelError) {
                console.error("Error fetching or drawing raw pixel data:", pixelError);
            }
        }
        // === End of Phase 2 Change ===


    } catch (error) {
        runStatusDiv.textContent = `Runner status fetch error: ${error}. Polling might stop.`;
        runnerLoopActive = false;
        if (runningStatusIntervalId) clearInterval(runningStatusIntervalId); runningStatusIntervalId = null;
        updateRunnerControlsAvailability();
    }
}

// --- Initial Setup ---
document.addEventListener('DOMContentLoaded', () => {
    initializeTrainerDrawCanvas(); // Set up trainer drawing canvas first
    document.getElementById('trainTabButton').click(); // Activate TrainTab which will also call updateTrainerControlsAvailability
    
    trainBrushSizeValue.textContent = trainBrushSizeSlider.value;
    trainBrushOpacityValue.textContent = trainBrushOpacitySlider.value + '%';

    // Initialize hidden high-res capture canvas
    highResCaptureCanvas = document.createElement('canvas');
    highResCaptureCanvas.width = TARGET_CAPTURE_DIM;
    highResCaptureCanvas.height = TARGET_CAPTURE_DIM;
    highResCaptureCtx = highResCaptureCanvas.getContext('2d');
    highResCaptureCtx.imageSmoothingEnabled = false; // Crucial for pixelated scaling

    // Initialize trainerProgressCanvas with a placeholder
    if (trainerProgressCtx) { // NEW
        const dimW = parseInt(trainerProgressCanvasEl.getAttribute('width')) || 512; // NEW
        const dimH = parseInt(trainerProgressCanvasEl.getAttribute('height')) || 512; // NEW
        trainerProgressCtx.fillStyle = '#f0f0f0'; // NEW
        trainerProgressCtx.fillRect(0, 0, dimW, dimH); // NEW
        trainerProgressCtx.fillStyle = '#6c757d'; // NEW
        trainerProgressCtx.font = '32px sans-serif'; // NEW
        trainerProgressCtx.textAlign = 'center'; // NEW
        trainerProgressCtx.textBaseline = 'middle'; // NEW
        trainerProgressCtx.fillText('Target / Training Progress', dimW / 2, dimH / 2); // NEW
    } // NEW

    trainBrushSizeSlider.addEventListener('input', () => {
        trainBrushSizeValue.textContent = trainBrushSizeSlider.value;
    });

    trainBrushOpacitySlider.addEventListener('input', () => {
        trainBrushOpacityValue.textContent = trainBrushOpacitySlider.value + '%';
    });

    const trainEraserModeCheckbox = document.getElementById('trainEraserModeCheckbox');
    trainEraserModeCheckbox.addEventListener('change', () => {
        isEraserModeTrain = trainEraserModeCheckbox.checked;
        // Optionally, change cursor or provide visual feedback
        trainerDrawCanvasEl.style.cursor = isEraserModeTrain ? 'url("data:image/svg+xml;utf8,<svg xmlns=\'http://www.w3.org/2000/svg\' width=\'24\' height=\'24\' viewBox=\'0 0 24 24\'><path fill=\'black\' d=\'M5.414 20.586L19.556 6.444l-1.414-1.414L4 19.172zM4 20a1 1 0 001.414.086L20.586 4.914A1 1 0 0020 4H5a1 1 0 00-1 1v15z\'/><rect x=\'3\' y=\'18\' width=\'18\' height=\'3\' fill=\'grey\'/></svg>") 12 12, auto' : 'crosshair';
    });

    experimentTypeSelectTrain.addEventListener('change', () => {
        const isRegen = experimentTypeSelectTrain.value === 'Regenerating';
        damageNInputTrain.style.display = isRegen ? 'block' : 'none';
        damageNLabelTrain.style.display = isRegen ? 'block' : 'none';
    });
    experimentTypeSelectTrain.dispatchEvent(new Event('change'));

    runBrushSizeValue.textContent = runBrushSizeSlider.value;
    runFpsValue.textContent = runFpsSlider.value;
    currentRunToolMode = runToolModeEraseRadio.checked ? 'erase' : 'draw';

    // --- Capture Tool Event Listeners ---
    takeScreenshotTrainBtn.addEventListener('click', () => {
        if (trainerDrawCanvasEl.style.display !== 'none') {
            captureCanvasAsImage(trainerDrawCanvasEl, 'NCA_Train_Drawing');
        } else if (trainerProgressCanvasEl.style.display !== 'none') { // MODIFIED
            captureCanvasAsImage(trainerProgressCanvasEl, 'NCA_Train_Progress'); // MODIFIED
        }
    });
    takeScreenshotRunBtn.addEventListener('click', () => {
        captureCanvasAsImage(previewCanvasRunEl, 'NCA_Run_Preview'); // Pass the canvas element
    });

    startRecordingTrainBtn.addEventListener('click', () => {
        if (trainingLoopActive) {
            if (trainerDrawCanvasEl.style.display !== 'none') {
                startRecording(trainerDrawCanvasEl, 'Train_Drawing');
            } else if (trainerProgressCanvasEl.style.display !== 'none') { // MODIFIED
                startRecording(trainerProgressCanvasEl, 'Train_Progress'); // MODIFIED
            }
        } else {
            showGlobalStatus('Training loop must be active to record video.', false);
        }
    });
    stopRecordingTrainBtn.addEventListener('click', stopRecording);

    startRecordingRunBtn.addEventListener('click', () => {
        if (runnerLoopActive) {
            startRecording(previewCanvasRunEl, 'Run_Preview'); // Pass the canvas element
        } else {
            showGlobalStatus('Runner loop must be active to record video.', false);
        }
    });
    stopRecordingRunBtn.addEventListener('click', stopRecording);

    const placeholderDim = DRAW_CANVAS_WIDTH;
    const placeholderColor = '#f0f0f0';
    // const svgPlaceholder = (text) => `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='${placeholderDim}' height='${placeholderDim}' viewBox='0 0 ${placeholderDim} ${placeholderDim}'%3E%3Crect width='100%25' height='100%25' fill='${placeholderColor}'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' font-family='sans-serif' font-size='32' fill='%236c757d'%3E${text}%3C/text%3E%3C/svg%3E`; // REMOVED
    
    // trainerProgressImgEl.src = svgPlaceholder('Target / Training Progress'); // REMOVED

    // Initial calls after DOM is ready and tab is set
    // updateTrainerControlsAvailability(); // Called by openTab
    // updateRunnerControlsAvailability();   // Called by openTab
    // if (currentOpenTab === 'TrainTab') fetchTrainerStatus(); // Called by openTab
    // else if (currentOpenTab === 'RunTab') fetchRunnerStatus(); // Called by openTab

    // Collapsible Fieldset Logic
    document.querySelectorAll('.collapsible-fieldset .collapsible-header').forEach(header => {
        header.addEventListener('click', function() {
            const fieldset = this.closest('.collapsible-fieldset');
            fieldset.classList.toggle('collapsed');
        });
    });

    // Initially collapse all fieldsets except the first one in each tab
    document.querySelectorAll('.tab-content').forEach(tabContent => {
        const fieldsets = tabContent.querySelectorAll('.collapsible-fieldset');
        if (fieldsets.length > 0) {
            fieldsets.forEach((fieldset, index) => {
                if (index > 0) { // Collapse all except the first one
                    fieldset.classList.add('collapsed');
                }
            });
        }
    });
});

if (previewCanvasRunCtx) { // New placeholder drawing for runner canvas
    const dimW = parseInt(previewCanvasRunEl.getAttribute('width')) || 512;
    const dimH = parseInt(previewCanvasRunEl.getAttribute('height')) || 512;
    previewCanvasRunCtx.fillStyle = '#f0f0f0';
    previewCanvasRunCtx.fillRect(0, 0, dimW, dimH);
    previewCanvasRunCtx.fillStyle = '#6c757d';
    previewCanvasRunCtx.font = '32px sans-serif';
    previewCanvasRunCtx.textAlign = 'center';
    previewCanvasRunCtx.textBaseline = 'middle';
    previewCanvasRunCtx.fillText('Runner Preview', dimW / 2, dimH / 2);
}