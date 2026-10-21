function applyFaceDistortions(faces, valLogic, valFact, valFake) {
  // Ruducing fact-checking logic
  let effectiveFake = max(0, valFake - valFact);

  // blur radius (20px)
  let blurAmount = 20 - map(valLogic, 0, 1023, 0, 8) - map(valFact, 0, 1023, 0, 12);
  blurAmount = max(0, blurAmount);
  let alphaVal = map(blurAmount, 0, 20, 0, 230);

  for (let i = 0; i < faces.length; i++) {
    let face = faces[i];

    if (face.keypoints && face.keypoints.length > 386) {
      let leftX = (face.keypoints[159].x + face.keypoints[145].x) / 2;
      let leftY = (face.keypoints[159].y + face.keypoints[145].y) / 2;
      let rightX = (face.keypoints[386].x + face.keypoints[374].x) / 2;
      let rightY = (face.keypoints[386].y + face.keypoints[374].y) / 2;

      let eyeDist = dist(leftX, leftY, rightX, rightY);
      let spotWidth = eyeDist * 0.7;
      let spotHeight = eyeDist * 0.5;

      // 1. blurring
      if (blurAmount > 0.5) {
        push();
        drawingContext.filter = `blur(${blurAmount}px)`;
        fill(20, 20, 20, alphaVal);
        noStroke();
        ellipse(leftX, leftY, spotWidth, spotHeight);
        ellipse(rightX, rightY, spotWidth, spotHeight);
        pop();
      }

      // 2. Distortion face with "Fake Generation"
      if (effectiveFake > 40) {
        let distortion = map(effectiveFake, 40, 1023, 5, 50);
        push();
        fill(200, 30, 30, 170);
        stroke(255);
        strokeWeight(1);
        ellipse(leftX - distortion, leftY - distortion, spotWidth * 0.6, spotHeight * 0.6);
        ellipse(rightX + distortion, rightY - distortion, spotWidth * 0.6, spotHeight * 0.6);

        stroke(255, 0, 0, 180);
        line(leftX, leftY, rightX + distortion, rightY - distortion);
        pop();
      }
    }
  }

  return effectiveFake;
}