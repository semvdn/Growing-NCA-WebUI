// static/js/main.js

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

    // Clear the other tab's interval and start the current one's if needed
    if (tabId === 'TrainTab') {
        if (runningStatusIntervalId) {
            clearInterval(runningStatusIntervalId);
            runningStatusIntervalId = null;
        }
        if ((trainerInitialized || trainingLoopActive) && !trainingStatusIntervalId) {
            trainingStatusIntervalId = setInterval(fetchTrainerStatus, 1200);
        }
        fetchTrainerStatus();
    } else if (tabId === 'RunTab') {
        if (trainingStatusIntervalId) {
            clearInterval(trainingStatusIntervalId);
            trainingStatusIntervalId = null;
        }
        if (runnerModelLoaded && !runningStatusIntervalId) {
            const fps = parseInt(runFpsSlider.value);
            let interval = runnerLoopActive ? Math.max(50, 1000 / fps) : 1000;
            runningStatusIntervalId = setInterval(fetchRunnerStatus, interval);
        }
        fetchRunnerStatus();
    }
    updateTrainerControlsAvailability();
    updateRunnerControlsAvailability();
}


// --- Initial Setup ---
document.addEventListener('DOMContentLoaded', () => {
    // Initialize modules
    initializeTrainer();
    initializeRunner();
    initializeCapture();

    // Set initial UI states
    document.getElementById('trainTabButton').click();
    trainBrushSizeValue.textContent = trainBrushSizeSlider.value;
    trainBrushOpacityValue.textContent = trainBrushOpacitySlider.value + '%';
    runBrushSizeValue.textContent = runBrushSizeSlider.value;
    runFpsValue.textContent = runFpsSlider.value;
    currentRunToolMode = runToolModeEraseRadio.checked ? 'erase' : 'draw';
    
    // Initial placeholder for trainer progress
    if (trainerProgressCtx) {
        const dimW = parseInt(trainerProgressCanvasEl.getAttribute('width')) || 512;
        const dimH = parseInt(trainerProgressCanvasEl.getAttribute('height')) || 512;
        trainerProgressCtx.fillStyle = '#f0f0f0';
        trainerProgressCtx.fillRect(0, 0, dimW, dimH);
        trainerProgressCtx.fillStyle = '#6c757d';
        trainerProgressCtx.font = '32px sans-serif';
        trainerProgressCtx.textAlign = 'center';
        trainerProgressCtx.textBaseline = 'middle';
        trainerProgressCtx.fillText('Target / Training Progress', dimW / 2, dimH / 2);
    }
    
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
        fieldsets.forEach((fieldset, index) => {
            if (index > 0) {
                fieldset.classList.add('collapsed');
            }
        });
    });
});