// static/js/trainer.js

function initializeTrainer() {
    initializeTrainerDrawCanvas();
    initializeTrainerEventListeners();
}

function initializeTrainerDrawCanvas() {
    trainerDrawCanvasEl.width = DRAW_CANVAS_WIDTH;
    trainerDrawCanvasEl.height = DRAW_CANVAS_HEIGHT;
    trainerCtx = trainerDrawCanvasEl.getContext('2d');
    trainerCtx.clearRect(0, 0, DRAW_CANVAS_WIDTH, DRAW_CANVAS_HEIGHT);
    trainerDrawCanvasEl.classList.add('active-draw');
    trainerProgressCanvasEl.style.display = 'none';
    trainerDrawCanvasEl.style.display = 'inline-block';
    trainerTargetConfirmed = false;
    initTrainerBtn.disabled = true;
    updateTrainerControlsAvailability();
    trainingStatusDiv.textContent = "Status: Draw a pattern on the canvas or load an image file.";
    saveTrainerCanvasState();

    if (!tempTrainerCanvas) {
        tempTrainerCanvas = document.createElement('canvas');
        tempTrainerCanvas.width = DRAW_CANVAS_WIDTH;
        tempTrainerCanvas.height = DRAW_CANVAS_HEIGHT;
        tempTrainerCtx = tempTrainerCanvas.getContext('2d');
    }
    tempTrainerCtx.clearRect(0, 0, DRAW_CANVAS_WIDTH, DRAW_CANVAS_HEIGHT);
}

function clearTrainerDrawCanvas() {
    if (trainerCtx) {
        trainerCtx.clearRect(0, 0, trainerDrawCanvasEl.width, trainerDrawCanvasEl.height);
        trainerTargetConfirmed = false;
        initTrainerBtn.disabled = true;
        updateTrainerControlsAvailability();
        trainingStatusDiv.textContent = "Status: Drawing cleared. Draw a pattern or load a file.";
        saveTrainerCanvasState(); // Save the cleared state for undo
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

    tempTrainerCtx.globalCompositeOperation = 'source-over';
    tempTrainerCtx.lineWidth = brushSize * 2;
    tempTrainerCtx.lineCap = 'round';
    tempTrainerCtx.lineJoin = 'round';
    tempTrainerCtx.strokeStyle = color;
    tempTrainerCtx.fillStyle = color;

    if (lastX === -1 || lastY === -1) {
        tempTrainerCtx.beginPath();
        tempTrainerCtx.arc(x, y, brushSize, 0, Math.PI * 2);
        tempTrainerCtx.fill();
    } else {
        tempTrainerCtx.beginPath();
        tempTrainerCtx.moveTo(lastX, lastY);
        tempTrainerCtx.lineTo(x, y);
        tempTrainerCtx.stroke();
    }
    lastX = x;
    lastY = y;

    trainerCtx.clearRect(0, 0, trainerDrawCanvasEl.width, trainerDrawCanvasEl.height);
    if (currentStrokeBaseImageData) trainerCtx.putImageData(currentStrokeBaseImageData, 0, 0);

    trainerCtx.globalCompositeOperation = isEraserModeTrain ? 'destination-out' : 'source-over';
    trainerCtx.globalAlpha = brushOpacity;
    trainerCtx.drawImage(tempTrainerCanvas, 0, 0);
    trainerCtx.globalAlpha = 1;
    trainerCtx.globalCompositeOperation = 'source-over';

    if (trainerTargetConfirmed) trainingStatusDiv.textContent = "Status: Drawing modified. Confirm again to use as target.";
    trainerTargetConfirmed = false;
    initTrainerBtn.disabled = true;
    updateTrainerControlsAvailability();
}

function saveTrainerCanvasState() {
    if (!trainerCtx) return;
    const imageData = trainerCtx.getImageData(0, 0, trainerDrawCanvasEl.width, trainerDrawCanvasEl.height);
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

async function handleLoadTargetForTrainerFromFile(formData) {
    const response = await postFormRequest('/load_target_from_file', formData);
    showGlobalStatus(response.message, response.success);
    if (response.success) {
        trainerDrawCanvasEl.style.display = 'none';
        trainerProgressCanvasEl.style.display = 'inline-block';
        try {
            const rawPreviewResponse = await fetch(`/get_trainer_target_raw_preview_data?t=${new Date().getTime()}`);
            if (rawPreviewResponse.ok) {
                const rawData = await rawPreviewResponse.json();
                if (rawData.success && rawData.pixels && rawData.height > 0 && rawData.width > 0) {
                    drawUpscaledPixels(trainerProgressCtx, DRAW_CANVAS_WIDTH, DRAW_CANVAS_HEIGHT, rawData.pixels, rawData.width, rawData.height);
                } else {
                    console.error("Failed to get raw target preview data:", rawData.message);
                }
            }
        } catch (pixelError) {
            console.error("Error fetching or drawing raw target pixel data:", pixelError);
        }
        
        trainerTargetConfirmed = true;
        trainerInitialized = false;
        trainingLoopActive = false;
        if (trainingStatusIntervalId) clearInterval(trainingStatusIntervalId); trainingStatusIntervalId = null;
        trainModelParamsText.textContent = "N/A (File target loaded, initialize Trainer)";
        trainingStatusDiv.textContent = "Status: File target loaded. Initialize Trainer.";
    }
    updateTrainerControlsAvailability();
}

async function fetchTrainerStatus() {
    if (currentOpenTab !== 'TrainTab') return;
    
    if (!trainerInitialized && !trainerTargetConfirmed) { 
        if (trainingStatusIntervalId) {
             clearInterval(trainingStatusIntervalId); 
             trainingStatusIntervalId = null;
        }
        return;
    }

    try {
        const response = await fetch('/get_training_status');
        if (!response.ok) { 
            trainingStatusDiv.textContent = `Trainer status error: ${response.status}`;
            return; 
        }
        const data = await response.json();
        trainingStatusDiv.textContent = data.status_message || `Step: ${data.step || 0}, Loss: ${data.loss || 'N/A'}, Time: ${data.training_time || 'N/A'}`;
        
        if (data.is_training || (trainerInitialized && trainerTargetConfirmed)) {
            trainerDrawCanvasEl.style.display = 'none';
            trainerProgressCanvasEl.style.display = 'inline-block';
            try {
                const rawPreviewResponse = await fetch(`/get_live_trainer_raw_preview_data?t=${new Date().getTime()}`);
                if (rawPreviewResponse.ok) {
                    const rawData = await rawPreviewResponse.json();
                    if (rawData.success && rawData.pixels && rawData.height > 0 && rawData.width > 0) {
                        drawUpscaledPixels(trainerProgressCtx, DRAW_CANVAS_WIDTH, DRAW_CANVAS_HEIGHT, rawData.pixels, rawData.width, rawData.height);
                    }
                }
            } catch (pixelError) {
                console.error("Error fetching or drawing raw trainer pixel data:", pixelError);
            }
        } else if (trainerTargetConfirmed && !trainerInitialized) {
            trainerDrawCanvasEl.style.display = 'none';
            trainerProgressCanvasEl.style.display = 'inline-block';
             try {
                const rawPreviewResponse = await fetch(`/get_trainer_target_raw_preview_data?t=${new Date().getTime()}`);
                if (rawPreviewResponse.ok) {
                    const rawData = await rawPreviewResponse.json();
                    if (rawData.success && rawData.pixels && rawData.height > 0 && rawData.width > 0) {
                        drawUpscaledPixels(trainerProgressCtx, DRAW_CANVAS_WIDTH, DRAW_CANVAS_HEIGHT, rawData.pixels, rawData.width, rawData.height);
                    }
                }
            } catch (pixelError) {
                console.error("Error fetching or drawing raw target pixel data:", pixelError);
            }
        } else {
            initializeTrainerDrawCanvas();
        }

        const prevTrainingLoopActive = trainingLoopActive;
        trainingLoopActive = data.is_training;
        
        if (isRecording) {
            if (trainingLoopActive && !prevTrainingLoopActive) {
                recordingStartTime = Date.now();
                startRecordingTimer('Train');
            } else if (!trainingLoopActive && prevTrainingLoopActive) {
                clearInterval(recordingTimerIntervalId);
                pausedRecordingDuration += (Date.now() - recordingStartTime);
            }
        }
        
        if (prevTrainingLoopActive && !trainingLoopActive && trainingStatusIntervalId) {
            clearInterval(trainingStatusIntervalId);
            trainingStatusIntervalId = null;
        } else if (trainingLoopActive && !trainingStatusIntervalId && currentOpenTab === 'TrainTab'){
            trainingStatusIntervalId = setInterval(fetchTrainerStatus, 1200);
        }
        
        updateTrainerControlsAvailability();

    } catch (error) { 
        trainingStatusDiv.textContent = `Trainer status fetch error: ${error}.`;
        updateTrainerControlsAvailability();
    }
}

function initializeTrainerEventListeners() {
    // Canvas interaction
    trainerDrawCanvasEl.addEventListener('mousedown', (e) => {
        if (e.button !== 0 || trainingLoopActive) return;
        isDrawingOnTrainerCanvas = true;
        const rect = trainerDrawCanvasEl.getBoundingClientRect();
        lastX = e.clientX - rect.left;
        lastY = e.clientY - rect.top;
        currentStrokeBaseImageData = trainerCtx.getImageData(0, 0, trainerDrawCanvasEl.width, trainerDrawCanvasEl.height);
        tempTrainerCtx.clearRect(0, 0, tempTrainerCanvas.width, tempTrainerCanvas.height);
        drawOnTrainerCanvas(e);
        e.preventDefault();
    });
    trainerDrawCanvasEl.addEventListener('mousemove', (e) => {
        if (trainingLoopActive || !isDrawingOnTrainerCanvas) return;
        drawOnTrainerCanvas(e);
        e.preventDefault();
    });
    document.addEventListener('mouseup', (e) => {
        if (e.button !== 0 || !isDrawingOnTrainerCanvas) return;
        isDrawingOnTrainerCanvas = false;
        lastX = -1; lastY = -1;
        currentStrokeBaseImageData = null;
        saveTrainerCanvasState();
    });
    trainerDrawCanvasEl.addEventListener('mouseleave', () => {
        if (!isDrawingOnTrainerCanvas) return;
        isDrawingOnTrainerCanvas = false;
        lastX = -1; lastY = -1;
        currentStrokeBaseImageData = null;
        saveTrainerCanvasState();
    });

    // Buttons
    clearTrainCanvasBtn.addEventListener('click', () => {
        if (trainingLoopActive) return showGlobalStatus("Cannot clear drawing while training.", false);
        clearTrainerDrawCanvas();
    });
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
    confirmDrawingBtnTrain.addEventListener('click', async () => {
        if (trainingLoopActive || !trainerCtx) return;
        const drawnImageName = trainerDrawnImageNameInput.value.trim();
        if (!drawnImageName) return showGlobalStatus("Please enter a name for your drawn image.", false);

        const imageDataUrl = trainerDrawCanvasEl.toDataURL('image/png');
        const response = await postRequest('/upload_drawn_pattern_target', { image_data_url: imageDataUrl, drawn_image_name: drawnImageName });
        showGlobalStatus(response.message, response.success);
        if (response.success) {
            trainerTargetConfirmed = true;
            trainingStatusDiv.textContent = "Status: Drawn pattern confirmed. Initialize Trainer.";
            fetchTrainerStatus(); // To show the target preview
        }
        updateTrainerControlsAvailability();
    });
    loadImageFileBtnTrain.addEventListener('click', () => {
        if (trainingLoopActive) return showGlobalStatus("Cannot load file while training.", false);
        if (!imageFileInputTrain.files.length) return showGlobalStatus('Please select an image file.', false);
        const formData = new FormData();
        formData.append('image_file', imageFileInputTrain.files[0]);
        handleLoadTargetForTrainerFromFile(formData);
    });
    initTrainerBtn.addEventListener('click', async () => {
        if (!trainerTargetConfirmed) return showGlobalStatus("Please confirm a target first.", false);
        const payload = {
            experiment_type: experimentTypeSelectTrain.value,
            fire_rate: parseFloat(fireRateInputTrain.value),
            damage_n: parseInt(damageNInputTrain.value),
            batch_size: parseInt(batchSizeInputTrain.value),
            pool_size: parseInt(poolSizeInputTrain.value),
            enable_entropy: enableEntropyTrainCheckbox.checked,
            entropy_strength: parseFloat(entropyStrengthTrainSlider.value)
        };
        const response = await postRequest('/initialize_trainer', payload);
        showGlobalStatus(response.message, response.success);
        if (response.success) {
            trainModelParamsText.textContent = response.model_summary || 'N/A';
            trainerInitialized = true;
            trainingLoopActive = false;
            if (trainingStatusIntervalId) clearInterval(trainingStatusIntervalId);
            trainingStatusIntervalId = setInterval(fetchTrainerStatus, 1200);
            trainingStatusDiv.textContent = "Status: Trainer Initialized.";
        } else {
            trainerInitialized = false;
            trainingStatusDiv.textContent = "Status: Trainer initialization failed.";
        }
        updateTrainerControlsAvailability();
    });
    startTrainingBtn.addEventListener('click', async () => {
        if (!trainerInitialized) return;
        const response = await postRequest('/start_training');
        showGlobalStatus(response.message, response.success);
        if (response.success) {
            trainingLoopActive = true;
            if (!trainingStatusIntervalId) trainingStatusIntervalId = setInterval(fetchTrainerStatus, 1200);
        }
        updateTrainerControlsAvailability();
    });
    stopTrainingBtn.addEventListener('click', async () => {
        if (!trainerInitialized || !trainingLoopActive) return;
        const response = await postRequest('/stop_training');
        showGlobalStatus(response.message, response.success);
        trainingLoopActive = false;
        if (trainingStatusIntervalId) {
            clearInterval(trainingStatusIntervalId);
            trainingStatusIntervalId = null;
        }
        fetchTrainerStatus();
        updateTrainerControlsAvailability();
    });
    saveTrainerModelBtn.addEventListener('click', async () => {
        if (!trainerInitialized) return;
        const response = await postRequest('/save_trainer_model');
        showGlobalStatus(response.message, response.success);
    });
    loadTrainerModelBtn.addEventListener('click', async () => {
        if (trainingLoopActive) return showGlobalStatus("Stop training before loading a model.", false);
        if (!loadTrainerModelFileInput.files.length) return showGlobalStatus('Please select a model file.', false);
        const formData = new FormData();
        formData.append('model_file', loadTrainerModelFileInput.files[0]);
        const response = await postFormRequest('/load_trainer_model', formData);
        showGlobalStatus(response.message, response.success);
        if (response.success) {
            trainerInitialized = true;
            trainingLoopActive = false;
            if (trainingStatusIntervalId) clearInterval(trainingStatusIntervalId);
            trainingStatusIntervalId = setInterval(fetchTrainerStatus, 1200);
            trainModelParamsText.textContent = response.model_summary || 'N/A';
            trainingStatusDiv.textContent = "Status: Trainer model loaded.";
            if (response.metadata) {
                experimentTypeSelectTrain.value = response.metadata.experiment_type || 'Growing';
                fireRateInputTrain.value = response.metadata.fire_rate || 0.5;
                //... update other controls ...//
            }
        } else {
            trainerInitialized = false;
        }
        updateTrainerControlsAvailability();
    });

    // Inputs
    document.getElementById('trainEraserModeCheckbox').addEventListener('change', (e) => {
        isEraserModeTrain = e.target.checked;
        trainerDrawCanvasEl.style.cursor = isEraserModeTrain ? 'url("data:image/svg+xml;utf8,<svg xmlns=\'http://www.w3.org/2000/svg\' width=\'24\' height=\'24\' viewBox=\'0 0 24 24\'><path fill=\'black\' d=\'M5.414 20.586L19.556 6.444l-1.414-1.414L4 19.172zM4 20a1 1 0 001.414.086L20.586 4.914A1 1 0 0020 4H5a1 1 0 00-1 1v15z\'/><rect x=\'3\' y=\'18\' width=\'18\' height=\'3\' fill=\'grey\'/></svg>") 12 12, auto' : 'crosshair';
    });
    experimentTypeSelectTrain.addEventListener('change', () => {
        const isRegen = experimentTypeSelectTrain.value === 'Regenerating';
        damageNInputTrain.style.display = isRegen ? 'block' : 'none';
        damageNLabelTrain.style.display = isRegen ? 'block' : 'none';
    });
    enableEntropyTrainCheckbox.addEventListener('change', () => {
        entropyStrengthTrainSlider.disabled = !enableEntropyTrainCheckbox.checked;
        updateTrainerControlsAvailability();
    });
    entropyStrengthTrainSlider.addEventListener('input', () => {
        entropyStrengthValueTrain.textContent = parseFloat(entropyStrengthTrainSlider.value).toFixed(3);
    });
    trainBrushSizeSlider.addEventListener('input', () => { trainBrushSizeValue.textContent = trainBrushSizeSlider.value; });
    trainBrushOpacitySlider.addEventListener('input', () => { trainBrushOpacityValue.textContent = trainBrushOpacitySlider.value + '%'; });
}