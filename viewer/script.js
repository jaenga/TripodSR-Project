// GLTF Viewer using Three.js

let scene, camera, renderer, controls;
let currentModel = null;
const modelPath = '../outputs/gltf_models/';

// Initialize Three.js scene
function initScene() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x2a2a2a);
    
    // Camera
    camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
    );
    camera.position.set(0, 0, 5);
    
    // Renderer
    const container = document.getElementById('canvas-container');
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);
    
    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 1;
    controls.maxDistance = 50;
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    
    const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight1.position.set(5, 10, 5);
    directionalLight1.castShadow = true;
    scene.add(directionalLight1);
    
    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    directionalLight2.position.set(-5, -5, -5);
    scene.add(directionalLight2);
    
    // Grid helper
    const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
    scene.add(gridHelper);
    
    // Axes helper
    const axesHelper = new THREE.AxesHelper(2);
    scene.add(axesHelper);
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize);
    
    // Start animation loop
    animate();
}

// Load GLTF model from file object or filepath
function loadModel(fileOrPath) {
    const loadingEl = document.getElementById('loading');
    const errorEl = document.getElementById('error');
    loadingEl.classList.remove('hidden');
    errorEl.classList.add('hidden');
    
    // Remove previous model
    if (currentModel) {
        scene.remove(currentModel);
        currentModel = null;
    }
    
    const loader = new THREE.GLTFLoader();
    
    // Handle File object (from drag & drop or file input)
    if (fileOrPath instanceof File) {
        const fileUrl = URL.createObjectURL(fileOrPath);
        loader.load(
            fileUrl,
            // onLoad
            (gltf) => {
                loadingEl.classList.add('hidden');
                URL.revokeObjectURL(fileUrl); // Clean up
                setupModel(gltf.scene, fileOrPath.name);
            },
            // onProgress
            (progress) => {
                if (progress.lengthComputable) {
                    const percentComplete = (progress.loaded / progress.total) * 100;
                    loadingEl.textContent = `모델 로딩 중... ${Math.round(percentComplete)}%`;
                }
            },
            // onError
            (error) => {
                loadingEl.classList.add('hidden');
                URL.revokeObjectURL(fileUrl); // Clean up
                showError(`모델 로드 실패: ${error.message || error}`);
                console.error('Error loading model:', error);
            }
        );
        return;
    }
    
    // Handle filepath string (legacy support)
    if (typeof fileOrPath === 'string') {
        const filepath = modelPath + fileOrPath;
        loader.load(
            filepath,
            // onLoad
            (gltf) => {
                loadingEl.classList.add('hidden');
                setupModel(gltf.scene, fileOrPath);
            },
            // onProgress
            (progress) => {
                if (progress.lengthComputable) {
                    const percentComplete = (progress.loaded / progress.total) * 100;
                    loadingEl.textContent = `모델 로딩 중... ${Math.round(percentComplete)}%`;
                }
            },
            // onError
            (error) => {
                loadingEl.classList.add('hidden');
                showError(`모델 로드 실패: ${error.message || error}`);
                console.error('Error loading model:', error);
            }
        );
    }
}

// Setup loaded model in scene
function setupModel(model, filename) {
    // Center and scale model
    const box = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    
    // Scale to fit in view
    const maxDim = Math.max(size.x, size.y, size.z);
    if (maxDim > 0) {
        const scale = 3 / maxDim;
        model.scale.multiplyScalar(scale);
    }
    
    // Center the model
    box.setFromObject(model);
    box.getCenter(center);
    model.position.sub(center);
    
    // Enable shadows
    model.traverse((child) => {
        if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
        }
    });
    
    scene.add(model);
    currentModel = model;
    
    // Reset camera position
    camera.position.set(0, 0, 5);
    controls.target.set(0, 0, 0);
    controls.update();
    
    console.log('Model loaded:', filename);
}

// Show error message
function showError(message) {
    const errorEl = document.getElementById('error');
    errorEl.textContent = message;
    errorEl.classList.remove('hidden');
}

// Handle window resize
function onWindowResize() {
    const container = document.getElementById('canvas-container');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// Setup drag and drop
function setupDragAndDrop() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const viewerContainer = document.getElementById('viewer-container');
    
    // Click to select file
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    });
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        viewerContainer.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('drag-over');
        }, false);
        viewerContainer.addEventListener(eventName, () => {
            viewerContainer.classList.add('drag-over');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('drag-over');
        }, false);
        viewerContainer.addEventListener(eventName, () => {
            viewerContainer.classList.remove('drag-over');
        }, false);
    });
    
    // Handle dropped files
    ['drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, handleDrop, false);
        viewerContainer.addEventListener(eventName, handleDrop, false);
    });
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    // Check if file is GLTF/GLB
    const validExtensions = ['.gltf', '.glb'];
    const fileName = file.name.toLowerCase();
    const isValid = validExtensions.some(ext => fileName.endsWith(ext));
    
    if (!isValid) {
        showError('GLTF 또는 GLB 파일만 지원됩니다.');
        return;
    }
    
    loadModel(file);
}

// Initialize when page loads
window.addEventListener('DOMContentLoaded', () => {
    initScene();
    setupDragAndDrop();
});


