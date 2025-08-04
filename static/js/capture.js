// static/js/capture.js

// --- Capture State Variables ---
let mediaRecorder = null;
let recordedChunks = [];
let recordingTimerIntervalId = null;
let recordingStartTime = 0;
let pausedRecordingDuration = 0;
let isRecording = false;

// --- High-Res Capture Canvas (hidden) ---
let highResCaptureCanvas = null;
let highResCaptureCtx = null;

function initializeCapture() {
    // Initialize hidden high-res capture canvas
    highResCaptureCanvas = document.createElement('canvas');
    highResCaptureCanvas.width = TARGET_CAPTURE_DIM;
    highResCaptureCanvas.height = TARGET_CAPTURE_DIM;
    highResCaptureCtx = highResCaptureCanvas.getContext('2d');
    highResCaptureCtx.imageSmoothingEnabled = false; // Crucial for pixelated scaling

    // --- Capture Tool Event Listeners ---
    takeScreenshotTrainBtn.addEventListener('click', () => {
        if (trainerDrawCanvasEl.style.display !== 'none') {
            captureCanvasAsImage(trainerDrawCanvasEl, 'NCA_Train_Drawing');
        } else if (trainerProgressCanvasEl.style.display !== 'none') {
            captureCanvasAsImage(trainerProgressCanvasEl, 'NCA_Train_Progress');
        }
    });
    takeScreenshotRunBtn.addEventListener('click', () => {
        captureCanvasAsImage(previewCanvasRunEl, 'NCA_Run_Preview');
    });

    startRecordingTrainBtn.addEventListener('click', () => {
        if (trainingLoopActive) {
            if (trainerDrawCanvasEl.style.display !== 'none') {
                startRecording(trainerDrawCanvasEl, 'Train_Drawing');
            } else if (trainerProgressCanvasEl.style.display !== 'none') {
                startRecording(trainerProgressCanvasEl, 'Train_Progress');
            }
        } else {
            showGlobalStatus('Training loop must be active to record video.', false);
        }
    });
    stopRecordingTrainBtn.addEventListener('click', stopRecording);

    startRecordingRunBtn.addEventListener('click', () => {
        if (runnerLoopActive) {
            startRecording(previewCanvasRunEl, 'Run_Preview');
        } else {
            showGlobalStatus('Runner loop must be active to record video.', false);
        }
    });
    stopRecordingRunBtn.addEventListener('click', stopRecording);
}

function captureCanvasAsImage(sourceElement, filenamePrefix) {
    let canvasToProcess = sourceElement;
    let useWhiteBackground = false;

    if (sourceElement.id === 'trainerDrawCanvas' || sourceElement.id === 'trainerProgressCanvas') {
        useWhiteBackground = whiteBackgroundTrainCheckbox.checked;
    } else if (sourceElement.id === 'previewCanvasRun') {
        useWhiteBackground = whiteBackgroundRunCheckbox.checked;
    }

    if (useWhiteBackground) {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = sourceElement.naturalWidth || sourceElement.width;
        tempCanvas.height = sourceElement.naturalHeight || sourceElement.height;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.fillStyle = 'white';
        tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
        
        if (sourceElement.tagName === 'IMG' || sourceElement.tagName === 'CANVAS') {
             if (sourceElement.id === 'previewCanvasRun') {
                tempCtx.drawImage(highResCaptureCanvas, 0, 0, tempCanvas.width, tempCanvas.height);
            } else {
                tempCtx.drawImage(sourceElement, 0, 0, tempCanvas.width, tempCanvas.height);
            }
        } else {
            console.error('Unsupported element for capture:', sourceElement);
            return;
        }
        canvasToProcess = tempCanvas;
    } else {
        if (sourceElement.id === 'previewCanvasRun') {
            canvasToProcess = highResCaptureCanvas;
        } else {
            canvasToProcess = document.createElement('canvas');
            let ctx = canvasToProcess.getContext('2d');
            canvasToProcess.width = sourceElement.naturalWidth || sourceElement.width;
            canvasToProcess.height = sourceElement.naturalHeight || sourceElement.height;
            if (sourceElement.tagName === 'IMG' || sourceElement.tagName === 'CANVAS') {
                ctx.drawImage(sourceElement, 0, 0, canvasToProcess.width, canvasToProcess.height);
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

async function startRecording(canvasElement, tabName) {
    if (isRecording) {
        showGlobalStatus('Already recording.', false);
        return;
    }

    recordedChunks = [];
    let streamSourceCanvas = canvasElement;
    let useWhiteBackground = (canvasElement.id.includes('Train') ? whiteBackgroundTrainCheckbox : whiteBackgroundRunCheckbox).checked;
    
    let stream;
    let animationFrameId;

    const processAndDownload = async () => {
        showGlobalStatus('Saving video...', true);
        const webmBlob = new Blob(recordedChunks, { type: 'video/webm' });
        const outputFilename = `NCA_${tabName}_Recording_${new Date().toISOString().slice(0,19).replace(/[-T:]/g, '')}.mp4`;
        const a = document.createElement('a');
        a.href = URL.createObjectURL(webmBlob);
        a.download = outputFilename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(a.href);
        showGlobalStatus('Video recorded and downloaded!', true);
        recordedChunks = [];
        isRecording = false;
        updateTrainerControlsAvailability();
        updateRunnerControlsAvailability();
    };

    if (useWhiteBackground) {
        const tempRecordingCanvas = document.createElement('canvas');
        tempRecordingCanvas.width = canvasElement.naturalWidth || canvasElement.width;
        tempRecordingCanvas.height = canvasElement.naturalHeight || canvasElement.height;
        const tempRecordingCtx = tempRecordingCanvas.getContext('2d');

        const drawFrameWithWhiteBackground = () => {
            tempRecordingCtx.fillStyle = 'white';
            tempRecordingCtx.fillRect(0, 0, tempRecordingCanvas.width, tempRecordingCanvas.height);
            let source = (canvasElement.id === 'previewCanvasRun') ? highResCaptureCanvas : canvasElement;
            tempRecordingCtx.drawImage(source, 0, 0, tempRecordingCanvas.width, tempRecordingCanvas.height);
        };

        stream = tempRecordingCanvas.captureStream(60);
        const captureLoop = () => {
            if (!isRecording) {
                cancelAnimationFrame(animationFrameId);
                return;
            }
            drawFrameWithWhiteBackground();
            animationFrameId = requestAnimationFrame(captureLoop);
        };
        animationFrameId = requestAnimationFrame(captureLoop);

        mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
        mediaRecorder.onstop = () => {
            cancelAnimationFrame(animationFrameId);
            processAndDownload();
        };

    } else {
        streamSourceCanvas = (canvasElement.id === 'previewCanvasRun') ? highResCaptureCanvas : canvasElement;
        stream = streamSourceCanvas.captureStream(60);
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
        mediaRecorder.onstop = processAndDownload;
    }

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) recordedChunks.push(event.data);
    };

    mediaRecorder.start();
    isRecording = true;
    recordingStartTime = Date.now();
    pausedRecordingDuration = 0;
    startRecordingTimer(tabName);
    showGlobalStatus('Recording started...', true);
    updateTrainerControlsAvailability();
    updateRunnerControlsAvailability();
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        clearInterval(recordingTimerIntervalId);
        recordingTimerIntervalId = null;
    }
}

function updateRecordingTimer(tabName) {
    let timerSpan = tabName.includes('Train') ? recordingTimerTrain : recordingTimerRun;
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