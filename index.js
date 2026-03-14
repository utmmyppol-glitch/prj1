const API_BASE_URL = 'http://localhost:8000';

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initDropZones();
    initForms();
});

// Tab Logic
function initTabs() {
    const btns = document.querySelectorAll('.tab-btn');
    const contents = document.querySelectorAll('.tab-content');

    btns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            
            btns.forEach(b => b.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        });
    });
}

// Drop Zone Logic
function initDropZones() {
    document.querySelectorAll('.drop-zone').forEach(zone => {
        const input = zone.querySelector('.drop-zone__input');

        zone.addEventListener('click', () => input.click());

        input.addEventListener('change', () => {
            if (input.files.length) {
                updateThumbnail(zone, input.files[0]);
            }
        });

        zone.addEventListener('dragover', e => {
            e.preventDefault();
            zone.classList.add('drop-zone--over');
        });

        ['dragleave', 'dragend'].forEach(type => {
            zone.addEventListener(type, () => {
                zone.classList.remove('drop-zone--over');
            });
        });

        zone.addEventListener('drop', e => {
            e.preventDefault();
            if (e.dataTransfer.files.length) {
                input.files = e.dataTransfer.files;
                updateThumbnail(zone, e.dataTransfer.files[0]);
            }
            zone.classList.remove('drop-zone--over');
        });
    });
}

function updateThumbnail(zone, file) {
    let thumbnailElement = zone.querySelector('.drop-zone__thumb');

    if (zone.querySelector('.drop-zone__prompt')) {
        zone.querySelector('.drop-zone__prompt').style.display = 'none';
    }

    if (!thumbnailElement) {
        thumbnailElement = document.createElement('div');
        thumbnailElement.classList.add('drop-zone__thumb');
        zone.appendChild(thumbnailElement);
    }

    thumbnailElement.dataset.label = file.name;

    if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => {
            thumbnailElement.style.backgroundImage = `url('${reader.result}')`;
            thumbnailElement.style.width = '100%';
            thumbnailElement.style.height = '100%';
            thumbnailElement.style.position = 'absolute';
            thumbnailElement.style.top = '0';
            thumbnailElement.style.left = '0';
            thumbnailElement.style.borderRadius = '14px';
            thumbnailElement.style.backgroundSize = 'contain';
            thumbnailElement.style.backgroundRepeat = 'no-repeat';
            thumbnailElement.style.backgroundPosition = 'center';
            thumbnailElement.style.backgroundColor = '#000';
        };
    }
}

// Global Loader
const toggleLoader = (show) => {
    const loader = document.getElementById('loader');
    if (show) loader.classList.remove('hidden');
    else loader.classList.add('hidden');
};

// Form Interaction Logic
function initForms() {
    // Range value display
    document.querySelectorAll('input[type="range"]').forEach(input => {
        input.addEventListener('input', (e) => {
            e.target.nextElementSibling.textContent = e.target.value;
        });
    });

    // Handle all forms
    document.querySelectorAll('.upload-form, .text-form').forEach(form => {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const type = form.getAttribute('data-type');
            const resultArea = form.closest('.card').querySelector('.result-area');
            const dataDisplay = resultArea.querySelector('.data-display');
            const previewImg = resultArea.querySelector('.image-preview');

            toggleLoader(true);
            resultArea.classList.add('hidden');

            try {
                const formData = new FormData(form);
                let response;

                // Handle image preview for file uploads
                const fileInput = form.querySelector('input[type="file"]');
                if (fileInput && fileInput.files[0] && previewImg) {
                    const reader = new FileReader();
                    reader.onload = (e) => previewImg.src = e.target.result;
                    reader.readAsDataURL(fileInput.files[0]);
                }

                if (type === 'sentiment') {
                    // Sentiment uses Form Data but special handling for text-only
                    response = await fetch(`${API_BASE_URL}/predict/sentiment`, {
                        method: 'POST',
                        body: formData
                    });
                } else {
                    // All other vision APIs
                    response = await fetch(`${API_BASE_URL}/predict/${type}`, {
                        method: 'POST',
                        body: formData
                    });
                }

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'API request failed');
                }

                const result = await response.json();
                
                // Show result
                dataDisplay.textContent = JSON.stringify(result, null, 2);
                resultArea.classList.remove('hidden');

            } catch (err) {
                alert(`Error: ${err.message}`);
            } finally {
                toggleLoader(false);
            }
        });
    });
}
