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
  createCanvas(640, 480);
  createControlSliders();

  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();

  faceMesh.detectStart(video, gotFaces);

  // First require a headline from api.js
  fetchRandomHeadline();
}

function draw() {
  image(video, 0, 0, width, height);

  let rawTime = sliderTime.value();
  let rawLogic = sliderLogic.value();
  let rawFact = sliderFact.value();
  let rawFake = sliderFake.value();

  let effectiveFake = applyFaceDistortions(faces, rawLogic, rawFact, rawFake);

  drawTimeline(rawTime);
  drawHUD(rawTime, rawLogic, rawFact, rawFake, effectiveFake);

  // Drawing answer in the upper part of canvas
  drawHeadlineBanner();
}

function drawHeadlineBanner() {
  push();
  fill(0, 180);
  noStroke();
  rect(0, 0, width, 40);

  fill(255);
  textSize(12);
  textAlign(CENTER, CENTER);
  text(currentHeadline, width / 2, 20);

  // If AI answered, show it
  if (aiAnalysisResult) {
    fill(255, 200, 50);
    textSize(11);
    text(aiAnalysisResult, width / 2, height - 50);
  }
  pop();
}

function keyPressed() {
  // Space key  — send parameters to Java
  if (key === ' ') {
    let year = Math.round(map(sliderTime.value(), 0, 1023, 1960, 2026));
    sendAnalysisRequest(year, sliderLogic.value(), sliderFact.value(), sliderFake.value());
  }
  // Key 'N' (Next) — get a new headline
  if (key === 'n' || key === 'N') {
    fetchRandomHeadline();
  }
}

function gotFaces(results) {
  faces = results;
}