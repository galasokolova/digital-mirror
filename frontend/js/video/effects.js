function drawEyeBlur(leftX, leftY, rightX, rightY, spotWidth, spotHeight, blurSettings) {
  if (blurSettings.blurAmount <= 0.5) return;

  push();
  drawingContext.filter = `blur(${blurSettings.blurAmount}px)`;
  fill(20, 20, 20, blurSettings.alphaVal);
  noStroke();

  ellipse(leftX, leftY, spotWidth, spotHeight);
  ellipse(rightX, rightY, spotWidth, spotHeight);
  pop();
}

// Buffer to select an eye
let eyeBuffer = null;

function drawFakeDistortions(leftX, leftY, rightX, rightY, spotWidth, spotHeight, effectiveFake) {
  if (effectiveFake <= 40) return;

  // Vertical offset (forehead)
  let yOffset = map(effectiveFake, 40, 1023, 8, 30);

  // Rightward offset (adjusting)
  let xOffset = 35;

  let cloneAlpha = map(effectiveFake, 40, 1023, 140, 245);

  let cropW = Math.round(spotWidth * 1.5);
  let cropH = Math.round(spotHeight * 1.5);

  if (!eyeBuffer) {
    eyeBuffer = createGraphics(160, 120);
    eyeBuffer.pixelDensity(1);
  }

  push();
  imageMode(CENTER);
  tint(255, cloneAlpha);

  // 1. Left eye
  renderSoftEye(leftX, leftY, cropW, cropH);
  image(eyeBuffer, leftX + xOffset, leftY - yOffset);

  // 2. Right eye
  renderSoftEye(rightX, rightY, cropW, cropH);
  image(eyeBuffer, rightX + xOffset, rightY - yOffset);

  pop();
}

// Takes selected fragment and cuts softly
function renderSoftEye(cx, cy, w, h) {
  eyeBuffer.clear();

  // Top-left corner to capture video
  let sx = Math.max(0, Math.round(cx - w / 2));
  let sy = Math.max(0, Math.round(cy - h / 2));

  // Draw video rectangle to buffer
  eyeBuffer.image(video, 0, 0, w, h, sx, sy, w, h);

  // Mask mode: keep only what is drawn next
  eyeBuffer.drawingContext.save();
  eyeBuffer.drawingContext.globalCompositeOperation = 'destination-in';

  // Create radial gradient: solid center fading to edges
  let grad = eyeBuffer.drawingContext.createRadialGradient(
      w / 2, h / 2, Math.min(w, h) * 0.15,
      w / 2, h / 2, Math.min(w, h) * 0.50
  );
  grad.addColorStop(0, 'rgba(0, 0, 0, 1)');
  grad.addColorStop(0.7, 'rgba(0, 0, 0, 0.8)');
  grad.addColorStop(1, 'rgba(0, 0, 0, 0)');

  eyeBuffer.drawingContext.fillStyle = grad;
  eyeBuffer.drawingContext.fillRect(0, 0, w, h);
  eyeBuffer.drawingContext.restore();
}

function applyFaceDistortions(faces, valLogic, valFact, valFake) {
  let effectiveFake = calculateEffectiveFake(valFake, valFact);
  let blurSettings = calculateBlurSettings(valLogic, valFact);

  for (let i = 0; i < faces.length; i++) {
    let face = faces[i];

    if (!face.keypoints || face.keypoints.length <= 386) continue;

    let eyes = getEyeCenters(face.keypoints);
    let dims = getSpotDimensions(eyes.leftX, eyes.leftY, eyes.rightX, eyes.rightY);

    drawEyeBlur(eyes.leftX, eyes.leftY, eyes.rightX, eyes.rightY, dims.spotWidth, dims.spotHeight, blurSettings);
    drawFakeDistortions(eyes.leftX, eyes.leftY, eyes.rightX, eyes.rightY, dims.spotWidth, dims.spotHeight, effectiveFake);
  }

  return effectiveFake;
}