
/*
 * DIGITAL MIRROR — LEFT CANVAS
 * Camera + FaceMesh + Generative Needle Agents
 *
 * Temporary test:
 * Trust controls visual intensity.
 * This is NOT the real Distortion Index.
 */

let faceMesh;
let video;
let faces = [];

const options = {
    maxFaces: 1,
    refineLandmarks: false,
    flipHorizontal: false
};

let erosionBuffer;
let vignetteLayer;
let agents = [];

// Temporary visual intensity for testing
let previewIntensity = 0.5;

function preload() {
    faceMesh = ml5.faceMesh(options);
}

function setup() {
    const canvas = createCanvas(640, 480);
    canvas.parent("video-canvas-container");

    pixelDensity(1);

    // Generative layers
    erosionBuffer = createErosionLayer(width, height);
    vignetteLayer = createVignetteLayer(width, height, 70);

    // Temporary HTML controller
    initMockControls({
        onTrustChange: (trust) => {
            updateTrust(trust);

            // Preview only — not ML distortion
            previewIntensity = trust;
        },

        onBack: () => {
            changeArticle(-1);
            resetVisualEffect();
            resetTrustSlider();
        },

        onSelect: (trust) => {
            updateTrust(trust);
            confirmTrust();
        },

        onNext: () => {
            changeArticle(1);
            resetVisualEffect();
            resetTrustSlider();
        }
    });

    // Existing camera
    video = createCapture(VIDEO);
    video.size(width, height);
    video.hide();

    // Existing FaceMesh
    faceMesh.detectStart(video, gotFaces);

    console.log("Generative Digital Mirror initialized");
}

function draw() {
    background(0);

    if (!video) return;

    image(video, 0, 0, width, height);

    if (vignetteLayer) {
        image(vignetteLayer, 0, 0);
    }

    // Read camera pixels for NeedleAgent
    video.loadPixels();

    const hasPixels =
        video.pixels &&
        video.pixels.length >= width * height * 4;

    if (hasPixels) {

        // Background spores
        spawnBorderSpores(agents, 20);

        // Face / eye spores
        spawnEyeSpores(
            faces,
            previewIntensity,
            agents,
            true
        );

        // Update and draw all agents
        for (let i = agents.length - 1; i >= 0; i--) {
            const agent = agents[i];

            agent.update(
                previewIntensity,
                video.pixels
            );

            agent.render(
                erosionBuffer,
                previewIntensity
            );

            if (agent.isDead()) {
                agents.splice(i, 1);
            }
        }
    }

    // Original gradual clearing behavior
    if (previewIntensity < 0.01) {
        erosionBuffer.push();
        erosionBuffer.erase(12, 12);
        erosionBuffer.rect(0, 0, width, height);
        erosionBuffer.noErase();
        erosionBuffer.pop();
    }

    // Composite generative layer over camera
    image(erosionBuffer, 0, 0);

    // Preview indicator
    push();
    noStroke();
    fill(0, 255, 200);
    textFont("Courier New");
    textSize(13);
    text(
        "VISUAL PREVIEW: " +
        previewIntensity.toFixed(2),
        20,
        height - 20
    );
    pop();
}

function gotFaces(results) {
    faces = results;
}

function resetVisualEffect() {
    agents.length = 0;

    if (erosionBuffer) {
        erosionBuffer.clear();
    }

    previewIntensity = 0.5;
}

function resetTrustSlider() {
    const slider = document.getElementById("trust-slider");
    const output = document.getElementById("trust-value");

    if (slider) slider.value = "0.5";
    if (output) output.textContent = "0.50";

    updateTrust(0.5);
}
