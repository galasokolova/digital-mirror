/**
 * Human Mirror
 * Mirrored webcam + FaceMesh + eye-centered needles.
 * Distortion is activated only after CONFIRM.
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
let humanNeedles;

let humanDistortionConfirmed = false;

function preload() {
  faceMesh = ml5.faceMesh(options);
}

function setup() {
  const canvas = createCanvas(640, 480);
  canvas.parent("human-canvas-container");

  pixelDensity(1);
  frameRate(30);

  erosionBuffer = createErosionLayer(width, height);
  vignetteLayer = createVignetteLayer(width, height, 70);

  humanNeedles = new NeedleSystem({
    maxAgents: 30,
    spawnRate: 1,
    agentOptions: {
      life: 200
    }
  });

  video = createCapture(VIDEO);
  video.size(width, height);
  video.hide();

  faceMesh.detectStart(video, gotFaces);
}

function draw() {
  background(0);

  if (!video) return;

  // Mirror the camera horizontally.
  push();
  translate(width, 0);
  scale(-1, 1);
  image(video, 0, 0, width, height);
  pop();

  if (vignetteLayer) {
    image(vignetteLayer, 0, 0);
  }

  // No distortion before CONFIRM.
  if (!humanDistortionConfirmed) {
    return;
  }

  video.loadPixels();

  const hasPixels =
      video.pixels &&
      video.pixels.length >= width * height * 4;

  if (hasPixels && humanNeedles) {

    // Convert FaceMesh coordinates to mirrored coordinates.
    const mirroredFaces = faces.map(face => ({
      ...face,
      keypoints: face.keypoints?.map(point => ({
        ...point,
        x: width - point.x
      })) ?? []
    }));

    humanNeedles.spawnEyes(mirroredFaces);

    humanNeedles.update(
        width,
        height,
        video.pixels
    );

    humanNeedles.renderP5(erosionBuffer);
  }

  // Slowly fade old marks.
  erosionBuffer.push();
  erosionBuffer.erase(3, 3);
  erosionBuffer.rect(0, 0, width, height);
  erosionBuffer.noErase();
  erosionBuffer.pop();

  image(erosionBuffer, 0, 0);

  // Display confirmed distortion.
  push();
  noStroke();
  fill(0, 255, 200);
  textFont("Courier New");
  textSize(13);

  text(
      `HUMAN DISTORTION: ${humanNeedles.intensity.toFixed(2)}`,
      20,
      height - 20
  );

  pop();
}

function gotFaces(results) {
  faces = results;
}

/**
 * Called only after CONFIRM.
 */
window.setHumanDistortion = function(value) {
  if (!humanNeedles) return;

  humanDistortionConfirmed = true;
  humanNeedles.setDistortion(value);
};

/**
 * Called when NEXT loads a new claim.
 */
window.resetHumanMirror = function() {
  humanDistortionConfirmed = false;

  humanNeedles?.reset();
  erosionBuffer?.clear();
};
