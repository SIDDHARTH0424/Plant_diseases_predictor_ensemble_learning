const tabs = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadPreviewArea = document.getElementById('upload-preview-area');
const uploadPreview = document.getElementById('upload-preview');
const resetUploadBtn = document.getElementById('reset-upload-btn');
const analyzeUploadBtn = document.getElementById('analyze-upload-btn');

const webcamVideo = document.getElementById('webcam-video');
const webcamCanvas = document.getElementById('webcam-canvas');
const startCamBtn = document.getElementById('start-cam-btn');
const captureBtn = document.getElementById('capture-btn');

const loadingIndicator = document.getElementById('loading-indicator');
const resultsCard = document.getElementById('results-card');

let currentStream = null;
let selectedFile = null;

// Tab Switching
tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        
        tab.classList.add('active');
        document.getElementById(`${tab.dataset.tab}-tab`).classList.add('active');

        if (tab.dataset.tab === 'upload') {
            stopWebcam();
        }
    });
});

// --- UPLOAD LOGIC ---
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
        handleFileSelection(e.dataTransfer.files[0]);
    }
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFileSelection(e.target.files[0]);
    }
});

function handleFileSelection(file) {
    if (!file.type.startsWith('image/')) return;
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        uploadPreview.src = e.target.result;
        dropZone.style.display = 'none';
        uploadPreviewArea.style.display = 'flex';
        resultsCard.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

resetUploadBtn.addEventListener('click', () => {
    selectedFile = null;
    uploadPreviewArea.style.display = 'none';
    dropZone.style.display = 'block';
    resultsCard.style.display = 'none';
    fileInput.value = "";
});

// --- WEBCAM LOGIC ---
startCamBtn.addEventListener('click', async () => {
    try {
        currentStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        webcamVideo.srcObject = currentStream;
        startCamBtn.style.display = 'none';
        captureBtn.style.display = 'block';
        resultsCard.style.display = 'none';
    } catch (err) {
        alert("Unable to access camera: " + err.message);
    }
});

function stopWebcam() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
        webcamVideo.srcObject = null;
        startCamBtn.style.display = 'block';
        captureBtn.style.display = 'none';
    }
}

// --- ANALYSIS LOGIC ---
analyzeUploadBtn.addEventListener('click', () => {
    if (selectedFile) {
        submitImageForAnalysis(selectedFile);
    }
});

captureBtn.addEventListener('click', () => {
    // Capture frame from video
    const context = webcamCanvas.getContext('2d');
    webcamCanvas.width = webcamVideo.videoWidth;
    webcamCanvas.height = webcamVideo.videoHeight;
    context.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);
    
    // To maintain fluid UI, proceed without stopping stream
    webcamCanvas.toBlob((blob) => {
        submitImageForAnalysis(blob, "webcam_capture.jpg");
    }, 'image/jpeg', 0.95);
});

async function submitImageForAnalysis(fileOrBlob, filename = "upload.jpg") {
    loadingIndicator.style.display = 'block';
    resultsCard.style.display = 'none';
    document.querySelector('.interface-card').style.opacity = '0.5';
    document.querySelector('.interface-card').style.pointerEvents = 'none';

    const formData = new FormData();
    formData.append('file', fileOrBlob, filename);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        renderResults(result);
    } catch (error) {
        resultsCard.innerHTML = `<div class="error-msg"><i class="fa-solid fa-triangle-exclamation"></i> Network error. Ensure the server is running. (${error.message})</div>`;
        resultsCard.style.display = 'block';
    } finally {
        loadingIndicator.style.display = 'none';
        document.querySelector('.interface-card').style.opacity = '1';
        document.querySelector('.interface-card').style.pointerEvents = 'auto';
    }
}

function renderResults(data) {
    if (data.detail) {
        resultsCard.innerHTML = `<div class="error-msg"><i class="fa-solid fa-triangle-exclamation"></i> Backend Error: ${data.detail}</div>`;
        resultsCard.style.display = 'block';
        return;
    }
    if (data.status === "rejected") {
        resultsCard.innerHTML = `<div class="error-msg"><i class="fa-solid fa-circle-xmark"></i> ${data.message}</div>`;
        resultsCard.style.display = 'block';
        return;
    }

    const isHealthy = data.disease.toLowerCase().includes("healthy");
    let healthColor = isHealthy ? "var(--success)" : "var(--danger)";
    
    // Handle split for "None" vs "High" etc.
    let severityTag = data.severity.split(' ')[0];
    const severityBadge = isHealthy ? 
        `<span class="severity-badge status-None"><i class="fa-solid fa-check"></i> Healthy</span>` : 
        `<span class="severity-badge status-${severityTag}"><i class="fa-solid fa-shield-virus"></i> ${data.severity} Severity</span>`;

    let html = `
        <div class="results-header">
            <div>
                <h2 style="color: ${healthColor}; font-size: 2rem; margin-bottom: 0.25rem;">${data.disease}</h2>
                <div style="color: var(--text-muted); font-size: 0.9rem;">Pathogen: ${data.pathogen || 'N/A'} • Confidence: ${(data.confidence * 100).toFixed(1)}%</div>
                ${data.low_confidence_warning ? '<div style="color: var(--warning); font-size: 0.8rem; margin-top: 5px;"><i class="fa-solid fa-triangle-exclamation"></i> Low confidence prediction</div>' : ''}
            </div>
            ${severityBadge}
        </div>
    `;

    if (!isHealthy) {
        html += `
            <div class="detail-section">
                <h4><i class="fa-solid fa-bolt"></i> Immediate Action</h4>
                <ul class="detail-list">
                    ${data.immediate_action && data.immediate_action.length ? data.immediate_action.map(a => `<li>${a}</li>`).join('') : '<li>Examine affected leaves and isolate plant.</li>'}
                </ul>
            </div>
            
            <div class="detail-section">
                <h4><i class="fa-solid fa-spray-can"></i> Recommended Products</h4>
                <ul class="detail-list">
                    ${data.recommended_products && data.recommended_products.length ? data.recommended_products.map(p => `<li>${p}</li>`).join('') : '<li>Consult local nursery for specific fungicide/bactericide.</li>'}
                </ul>
            </div>

            <div class="detail-section">
                <h4><i class="fa-brands fa-pagelines"></i> Organic Options</h4>
                <ul class="detail-list">
                    ${data.organic_options && data.organic_options.length ? data.organic_options.map(o => `<li>${o}</li>`).join('') : '<li>No specific organic options identified.</li>'}
                </ul>
            </div>
            
            <div class="detail-section" style="border-top: 1px solid var(--card-border); padding-top: 1rem; margin-top: 1rem;">
                <h4><i class="fa-solid fa-shield-halved"></i> Long-Term Prevention</h4>
                <ul class="detail-list" style="font-size: 0.9rem;">
                    ${data.prevention && data.prevention.length ? data.prevention.map(p => `<li>${p}</li>`).join('') : '<li>Ensure optimal growing conditions appropriately.</li>'}
                </ul>
            </div>
        `;
    } else {
         html += `
            <div class="detail-section" style="text-align:center; padding: 2rem 0;">
                <i class="fa-solid fa-seedling" style="font-size: 4rem; color: var(--success); margin-bottom: 1rem;"></i>
                <h3 style="color: var(--text-main); margin-bottom: 1rem;">Your plant looks great!</h3>
                <p style="color: var(--text-muted); max-width: 400px; margin: 0 auto;">Keep maintaining your current watering and lighting schedule. Prevention is always the best medicine.</p>
            </div>
         `;
    }

    if (data.notes) {
        html += `<div style="font-size: 0.8rem; color: var(--text-muted); font-style: italic; margin-top: 1rem;">Note: ${data.notes}</div>`;
    }

    resultsCard.innerHTML = html;
    resultsCard.style.display = 'block';
    
    // Smooth scroll to results
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'end' });
}
