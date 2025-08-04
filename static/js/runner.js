// static/js/runner.js

function initializeRunner() {
    initializeRunnerEventListeners();
    initializeRunnerCanvasPlaceholder();
}

function initializeRunnerCanvasPlaceholder() {
    if (previewCanvasRunCtx) {
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
}

async function handleRunnerAction(action, params = {}) {
    if (!runnerModelLoaded) return;
    const payload = { action, ...params };
    const response = await postRequest('/runner_action', payload);
    if (!response.success) {
        showGlobalStatus(response.message, false);
    } else {
         if (action !== 'modify_area') showGlobalStatus(response.message, true);
    }
    fetchRunnerStatus(); // Refresh status and preview after action
}

function performCanvasAction(event) {
    if (!runnerModelLoaded || currentOpenTab !== 'RunTab') return;

    const rect = previewCanvasRunEl.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    if (x < 0 || x > rect.width || y < 0 || y > rect.height) {
        if (isInteractingWithRunCanvas) {
            isInteractingWithRunCanvas = false;
            lastXRun = -1; lastYRun = -1;
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

async function fetchRunnerStatus() {
    if (currentOpenTab !== 'RunTab') return;
    if (!runnerModelLoaded) {
        if (runningStatusIntervalId) {
            clearInterval(runningStatusIntervalId);
            runningStatusIntervalId = null;
        }
        return;
    }

    try {
        const response = await fetch('/get_runner_status');
        if (!response.ok) {
            runStatusDiv.textContent = `Runner status error: ${response.status}`;
            runnerLoopActive = false;
            updateRunnerControlsAvailability();
            return;
        }
        const data = await response.json();
        
        const targetFPSDisplay = data.current_fps === "Max" ? "Max" : parseFloat(data.current_fps).toFixed(1);
        const actualFPSDisplay = data.actual_fps || "N/A";
        runStatusDiv.textContent = `${data.status_message || 'Status unavailable'} (Target: ${targetFPSDisplay} FPS, Actual: ${actualFPSDisplay} FPS)`;

        const prevRunnerLoopActive = runnerLoopActive;
        runnerLoopActive = data.is_loop_active;

        if (isRecording) {
            if (runnerLoopActive && !prevRunnerLoopActive) {
                recordingStartTime = Date.now();
                startRecordingTimer('Run');
            } else if (!runnerLoopActive && prevRunnerLoopActive) {
                clearInterval(recordingTimerIntervalId);
                pausedRecordingDuration += (Date.now() - recordingStartTime);
            }
        }
        
        if (runningStatusIntervalId) {
            clearInterval(runningStatusIntervalId);
            let newIntervalTime = runnerLoopActive ? Math.max(33, 1000 / (parseFloat(data.current_fps) || 20)) : 1000;
            runningStatusIntervalId = setInterval(fetchRunnerStatus, newIntervalTime);
        }

        if (runnerModelLoaded && (!isInteractingWithRunCanvas || data.is_loop_active)) {
            const rawPreviewResponse = await fetch(`/get_live_runner_raw_preview_data?t=${new Date().getTime()}`);
            if (rawPreviewResponse.ok) {
                const rawData = await rawPreviewResponse.json();
                if (rawData.success && rawData.pixels && rawData.height > 0 && rawData.width > 0) {
                    if (previewCanvasRunCtx) {
                        drawUpscaledPixels(previewCanvasRunCtx, DRAW_CANVAS_WIDTH, DRAW_CANVAS_HEIGHT, rawData.pixels, rawData.width, rawData.height);
                    }
                    if (highResCaptureCtx) {
                        drawUpscaledPixels(highResCaptureCtx, TARGET_CAPTURE_DIM, TARGET_CAPTURE_DIM, rawData.pixels, rawData.width, rawData.height);
                    }
                }
            }
        }

        updateRunnerControlsAvailability();
    } catch (error) {
        runStatusDiv.textContent = `Runner status fetch error: ${error}.`;
        runnerLoopActive = false;
        if (runningStatusIntervalId) clearInterval(runningStatusIntervalId);
        runningStatusIntervalId = null;
        updateRunnerControlsAvailability();
    }
}


function initializeRunnerEventListeners() {
    // Buttons
    loadCurrentTrainingModelBtnRun.addEventListener('click', async () => {
        if (!trainerInitialized) return showGlobalStatus("Trainer not initialized.", false);
        const response = await postRequest('/load_current_training_model_for_runner');
        showGlobalStatus(response.message, response.success);
        if (response.success) {
            runModelParamsText.textContent = response.model_summary || 'N/A';
            runnerModelLoaded = true;
            runnerLoopActive = false; 
            if (runningStatusIntervalId) clearInterval(runningStatusIntervalId);
            runningStatusIntervalId = setInterval(fetchRunnerStatus, 1000);
            if (response.metadata) {
                enableEntropyRunCheckbox.checked = response.metadata.enable_entropy || false;
                entropyStrengthRunSlider.value = response.metadata.entropy_strength || 0.0;
                entropyStrengthValueRun.textContent = parseFloat(entropyStrengthRunSlider.value).toFixed(3);
                entropyStrengthRunSlider.disabled = !enableEntropyRunCheckbox.checked;
            }
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
                modelInfoText += `\n--- Metadata ---\nTrained on: ${response.metadata.trained_on_image || 'N/A'}\nSteps: ${response.metadata.training_steps || 'N/A'}`;
            }
            runModelParamsText.textContent = modelInfoText;
            runnerModelLoaded = true;
            runnerLoopActive = false;
            if (runningStatusIntervalId) clearInterval(runningStatusIntervalId);
            runningStatusIntervalId = setInterval(fetchRunnerStatus, 1000);
             if (response.metadata) {
                enableEntropyRunCheckbox.checked = response.metadata.enable_entropy || false;
                entropyStrengthRunSlider.value = response.metadata.entropy_strength || 0.0;
                entropyStrengthValueRun.textContent = parseFloat(entropyStrengthRunSlider.value).toFixed(3);
                entropyStrengthRunSlider.disabled = !enableEntropyRunCheckbox.checked;
            }
        }
        updateRunnerControlsAvailability();
    });
    startRunningLoopBtn.addEventListener('click', async () => {
        if (!runnerModelLoaded) return;
        const response = await postRequest('/start_running');
        showGlobalStatus(response.message, response.success);
        if (response.success) {
            runnerLoopActive = true;
            const fps = parseInt(runFpsSlider.value);
            if (runningStatusIntervalId) clearInterval(runningStatusIntervalId);
            runningStatusIntervalId = setInterval(fetchRunnerStatus, Math.max(33, 1000 / fps));
        }
        updateRunnerControlsAvailability();
    });
    stopRunningLoopBtn.addEventListener('click', async () => {
        if (!runnerModelLoaded) return;
        const response = await postRequest('/stop_running');
        showGlobalStatus(response.message, response.success);
        fetchRunnerStatus();
        updateRunnerControlsAvailability();
    });
    resetRunnerStateBtn.addEventListener('click', () => handleRunnerAction('reset_runner'));

    // Canvas Interaction
    previewCanvasRunEl.addEventListener('mousedown', (event) => {
        if (event.button !== 0 || !runnerModelLoaded) return;
        isInteractingWithRunCanvas = true;
        performCanvasAction(event);
        event.preventDefault();
    });
    previewCanvasRunEl.addEventListener('mousemove', (event) => {
        if (!isInteractingWithRunCanvas) return;
        performCanvasAction(event);
        event.preventDefault();
    });
    document.addEventListener('mouseup', (event) => {
        if (event.button !== 0 || !isInteractingWithRunCanvas) return;
        isInteractingWithRunCanvas = false;
        lastXRun = -1; lastYRun = -1;
    });
    previewCanvasRunEl.addEventListener('mouseleave', () => {
        if (isInteractingWithRunCanvas) {
            isInteractingWithRunCanvas = false;
            lastXRun = -1; lastYRun = -1;
        }
    });

    // Inputs
    runBrushSizeSlider.addEventListener('input', () => { runBrushSizeValue.textContent = runBrushSizeSlider.value; });
    runFpsSlider.addEventListener('input', async () => {
        const fps = parseInt(runFpsSlider.value);
        runFpsValue.textContent = fps;
        if (runnerModelLoaded) await postRequest('/set_runner_speed', { fps: fps });
    });
    runToolModeEraseRadio.addEventListener('change', () => { if (runToolModeEraseRadio.checked) currentRunToolMode = 'erase'; updateRunnerControlsAvailability(); });
    runToolModeDrawRadio.addEventListener('change', () => { if (runToolModeDrawRadio.checked) currentRunToolMode = 'draw'; updateRunnerControlsAvailability(); });
    enableEntropyRunCheckbox.addEventListener('change', async () => {
        entropyStrengthRunSlider.disabled = !enableEntropyRunCheckbox.checked;
        await postRequest('/set_runner_entropy', { enable_entropy: enableEntropyRunCheckbox.checked, entropy_strength: parseFloat(entropyStrengthRunSlider.value) });
    });
    entropyStrengthRunSlider.addEventListener('input', async () => {
        entropyStrengthValueRun.textContent = parseFloat(entropyStrengthRunSlider.value).toFixed(3);
        await postRequest('/set_runner_entropy', { enable_entropy: enableEntropyRunCheckbox.checked, entropy_strength: parseFloat(entropyStrengthRunSlider.value) });
    });
}