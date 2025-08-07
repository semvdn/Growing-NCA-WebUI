import { createCA } from './ca.js';

document.addEventListener("DOMContentLoaded", () => {
    // --- Global State ---
    let manifest = null;
    let ca = null;
    let paused = false;
    let currentCheckpoints = [];
    
    const speedLevels = [0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10];
    let toolMode = 'erase';
    let brushSize = 10;
    let simSpeed = 1;
    let stepAccumulator = 0.0;
    let drawColor = '#ff0000';
    // New state for entropy
    let entropyEnabled = false;
    let entropyStrength = 0.02;

    // --- DOM Elements ---
    const modelSelect = document.getElementById('model-select');
    const regimenGroup = document.getElementById('regimen-radio-group');
    const checkpointSlider = document.getElementById('checkpoint-slider');
    const sliderValue = document.getElementById('slider-value');
    const resetButton = document.getElementById('reset-button');
    const playPauseButton = document.getElementById('play-pause-button');
    const stepCountEl = document.getElementById('stepCount');
    const ipsEl = document.getElementById('ips');
    const canvasTitle = document.getElementById('canvas-title');
    const speedSlider = document.getElementById('speed-slider');
    const speedValue = document.getElementById('speed-value');
    const toolModeRadios = document.querySelectorAll('input[name="tool-mode"]');
    const brushSizeSlider = document.getElementById('brush-size-slider');
    const brushSizeValue = document.getElementById('brush-size-value');
    const drawColorPicker = document.getElementById('draw-color-picker');
    // New entropy elements
    const entropyEnableCheckbox = document.getElementById('entropy-enable-checkbox');
    const entropyStrengthSlider = document.getElementById('entropy-strength-slider');
    const entropyStrengthValue = document.getElementById('entropy-strength-value');

    const canvas = document.getElementById('demo-canvas');
    const gl = canvas.getContext("webgl");
    const W = 96, H = 96, SCALE = 5;
    canvas.width = W * SCALE;
    canvas.height = H * SCALE;

    // --- Main Functions ---

    async function init() {
        try {
            const response = await fetch('models.json');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            manifest = await response.json();
            
            // Set initial state from HTML defaults
            simSpeed = speedLevels[parseInt(speedSlider.value)];
            speedValue.textContent = `${simSpeed}x`;
            drawColor = drawColorPicker.value;
            entropyEnabled = entropyEnableCheckbox.checked;
            entropyStrength = parseFloat(entropyStrengthSlider.value);

            populateModelSelect();
            setupEventListeners();
            await updateModel();
            requestAnimationFrame(render);
        } catch (e) {
            console.error("Failed to load model manifest:", e);
            canvasTitle.textContent = "Error: Could not load models.json";
        }
    }

    function populateModelSelect() {
        for (const modelName in manifest) {
            const option = document.createElement('option');
            option.value = modelName;
            option.textContent = modelName;
            modelSelect.appendChild(option);
        }
        updateRegimenAndSlider();
    }

    function updateRegimenAndSlider() {
        const modelName = modelSelect.value;
        const modelData = manifest[modelName];

        regimenGroup.innerHTML = '';
        let firstRegimen = true;
        for (const regimenName in modelData.regimens) {
            const input = document.createElement('input');
            input.type = 'radio';
            input.name = 'regimen';
            input.value = regimenName;
            input.id = `radio-${regimenName}`;
            if (firstRegimen) {
                input.checked = true;
                firstRegimen = false;
            }
            const label = document.createElement('label');
            label.htmlFor = `radio-${regimenName}`;
            label.textContent = regimenName;
            regimenGroup.appendChild(input);
            regimenGroup.appendChild(label);
        }
        updateSlider();
    }
    
    function updateSlider() {
        const modelName = modelSelect.value;
        const selectedRegimen = document.querySelector('input[name="regimen"]:checked').value;
        currentCheckpoints = manifest[modelName].regimens[selectedRegimen].checkpoints;

        checkpointSlider.min = 0;
        checkpointSlider.max = currentCheckpoints.length - 1;
        checkpointSlider.value = checkpointSlider.max;
        updateSliderValue();
    }

    function updateSliderValue() {
        const sliderIndex = parseInt(checkpointSlider.value);
        const steps = currentCheckpoints[sliderIndex];
        sliderValue.textContent = `${steps} steps`;
    }

    async function updateModel() {
        const modelName = modelSelect.value;
        const selectedRegimen = document.querySelector('input[name="regimen"]:checked').value;
        const sliderIndex = parseInt(checkpointSlider.value);
        const steps = currentCheckpoints[sliderIndex];
        
        const regimen_str = selectedRegimen.toLowerCase();
        const modelPath = `webgl_models/${modelName}_${regimen_str}_${steps}.json`;
        
        canvasTitle.textContent = `Loading ${modelName} (${regimen_str})...`;
        
        try {
            const response = await fetch(modelPath);
            if (!response.ok) throw new Error(`Could not fetch ${modelPath}`);
            const modelWeights = await response.json();

            if (!ca) {
                ca = createCA(gl, modelWeights, [W, H]);
                // Set initial entropy settings on the new CA instance
                ca.setEntropy(entropyEnabled, entropyStrength);
            } else {
                ca.setWeights(modelWeights);
                ca.reset();
            }
            canvasTitle.textContent = `${modelName}`;
        } catch (e) {
            console.error("Error loading model weights:", e);
            canvasTitle.textContent = `Error loading ${modelName}`;
        }
    }

    // --- Event Listeners ---

    function setupEventListeners() {
        modelSelect.addEventListener('change', () => {
            updateRegimenAndSlider();
            updateModel();
        });

        regimenGroup.addEventListener('change', (e) => {
            if(e.target.name === 'regimen') {
                updateSlider();
                updateModel();
            }
        });
        
        checkpointSlider.addEventListener('input', updateSliderValue);
        checkpointSlider.addEventListener('change', updateModel);

        resetButton.addEventListener('click', () => ca && ca.reset());

        playPauseButton.addEventListener('click', () => {
            paused = !paused;
            playPauseButton.textContent = paused ? 'Play' : 'Pause';
        });
        
        speedSlider.addEventListener('input', (e) => {
            const speedIndex = parseInt(e.target.value);
            simSpeed = speedLevels[speedIndex];
            speedValue.textContent = `${simSpeed}x`;
        });

        brushSizeSlider.addEventListener('input', (e) => {
            brushSize = parseInt(e.target.value);
            brushSizeValue.textContent = brushSize;
        });

        toolModeRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                toolMode = e.target.value;
                drawColorPicker.disabled = (toolMode !== 'draw');
            });
        });

        drawColorPicker.addEventListener('input', (e) => {
            drawColor = e.target.value;
        });
        
        // New Entropy Listeners
        entropyEnableCheckbox.addEventListener('change', (e) => {
            entropyEnabled = e.target.checked;
            entropyStrengthSlider.disabled = !entropyEnabled;
            // Immediately update the CA instance
            if (ca) ca.setEntropy(entropyEnabled, entropyStrength);
        });

        entropyStrengthSlider.addEventListener('input', (e) => {
            entropyStrength = parseFloat(e.target.value);
            entropyStrengthValue.textContent = entropyStrength.toFixed(3);
            if (ca) ca.setEntropy(entropyEnabled, entropyStrength);
        });

        function canvasToGrid(x, y) {
            const [w, h] = ca.gridSize;
            const gridX = Math.floor(x / canvas.clientWidth * w);
            const gridY = Math.floor(y / canvas.clientHeight * h);
            return [gridX, gridY];
        }

        function handleInteraction(e, isMove = false) {
            e.preventDefault();
            if (!ca || (isMove && e.buttons !== 1)) return;

            const rect = canvas.getBoundingClientRect();
            const [x, y] = canvasToGrid(e.clientX - rect.left, e.clientY - rect.top);
            
            if (toolMode === 'draw') {
                ca.paint(x, y, brushSize, 'color', drawColor);
            } else {
                ca.paint(x, y, brushSize, 'clear');
            }
        }

        canvas.addEventListener('mousedown', (e) => handleInteraction(e, false));
        canvas.addEventListener('mousemove', (e) => handleInteraction(e, true));
    }


    // --- Render Loop ---

    function render() {
        if (ca && !paused) {
            stepAccumulator += simSpeed;
            while (stepAccumulator >= 1.0) {
                ca.step();
                stepAccumulator -= 1.0;
            }
            
            stepCountEl.textContent = ca.getStepCount();
            ipsEl.textContent = ca.fps();
        }

        if (ca) {
            twgl.bindFramebufferInfo(gl);
            ca.draw();
        }
        
        requestAnimationFrame(render);
    }

    init();
});