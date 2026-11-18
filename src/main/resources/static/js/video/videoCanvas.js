// js/video/videoCanvas.js

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

    // running request of backend
    setInterval(pollHardwareState, 60);
}

function draw() {
    image(video, 0, 0, width, height);

    // data from the HARDWARE
    let rawTime = hardwareState.timeline;
    let rawLogic = hardwareState.logic;
    let rawFact = hardwareState.factCheck;
    let rawFake = hardwareState.fakeGenerator;

    let effectiveFake = applyFaceDistortions(faces, rawLogic, rawFact, rawFake);

    // lines of sliders
    drawHUD(rawTime, rawLogic, rawFact, rawFake, effectiveFake);
}

function keyPressed() {
    if (key === ' ') {
        let year = Math.round(map(hardwareState.timeline, 0, 1023, 1960, 2026));
        sendAnalysisRequest(year, hardwareState.logic, hardwareState.factCheck, hardwareState.fakeGenerator);
    }
    if (key === 'n' || key === 'N') {
        fetchRandomHeadline();
    }
}

function gotFaces(results) {
    faces = results;
}