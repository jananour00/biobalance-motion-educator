import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export function initThreeJSModel() {
    // Check if container exists
    const container = document.querySelector('.model-container');
    if (!container) return;
    
    // Clear previous content
    container.innerHTML = '';
    
    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);
    
    // Camera
    const camera = new THREE.PerspectiveCamera(
        75, 
        container.clientWidth / container.clientHeight, 
        0.1, 
        1000
    );
    camera.position.z = 5;
    
    // Renderer
    const renderer = new THREE.WebGLRenderer({ 
        antialias: true,
        alpha: true
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);
    
    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(0, 1, 1);
    scene.add(directionalLight);
    
    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    
    // Model Loader
    const loader = new GLTFLoader();
    loader.load(
        'assets/3d-models/human-anatomy.glb',
        (gltf) => {
            scene.add(gltf.scene);
            // Add any model-specific setup here
        },
        undefined,
        (error) => {
            console.error('Error loading model:', error);
            showModelError();
        }
    );
    
    // Animation Loop
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();
    
    // Handle Resize
    window.addEventListener('resize', () => {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });
}