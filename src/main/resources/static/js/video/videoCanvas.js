let faceMesh;
let video;
let faces = [];
let options = { maxFaces: 1, refineLandmarks: false, flipHorizontal: false };

let sliderTime;
let sliderLogic;
let sliderFact;
let sliderFake;

function preload() {
    faceMesh = ml5.faceMesh(options);
}

function setup() {
    let mainCanvas = createCanvas(640, 480);
    mainCanvas.parent('video-canvas-container');

    createControlSliders();

    video = createCapture(VIDEO);
    video.size(640, 480);
    video.hide();

    faceMesh.detectStart(video, gotFaces);

    fetchRandomHeadline();
}

function draw() {
    // Render webcam feed
    image(video, 0, 0, width, height);

    let rawTime = sliderTime.value();
    let rawLogic = sliderLogic.value();
    let rawFact = sliderFact.value();
    let rawFake = sliderFake.value();

    // Face distortion logic
    let effectiveFake = applyFaceDistortions(faces, rawLogic, rawFact, rawFake);

    // Bottom parameters HUD
    drawHUD(rawTime, rawLogic, rawFact, rawFake, effectiveFake);
}

function keyPressed() {
    if (key === ' ') {
        let year = Math.round(map(sliderTime.value(), 0, 1023, 1960, 2026));
        sendAnalysisRequest(year, sliderLogic.value(), sliderFact.value(), sliderFake.value());
    }
    if (key === 'n' || key === 'N') {
        fetchRandomHeadline();
    }
}

function gotFaces(results) {
    faces = results;
}