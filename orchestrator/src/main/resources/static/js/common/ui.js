function createControlSliders() {
  let container = document.getElementById('video-canvas-container');

  // Slider "Time" (0..1023)
  sliderTime = createSlider(0, 1023, 0, 1);
  sliderTime.parent(container);
  sliderTime.style('width', '130px');

  // Slider "Logic" (A1)
  sliderLogic = createSlider(0, 1023, 0, 1);
  sliderLogic.parent(container);
  sliderLogic.style('width', '130px');

  // Slider "Fact-checking" (A2)
  sliderFact = createSlider(0, 1023, 0, 1);
  sliderFact.parent(container);
  sliderFact.style('width', '130px');

  // Slider "Fake Generation" (A3)
  sliderFake = createSlider(0, 1023, 0, 1);
  sliderFake.parent(container);
  sliderFake.style('width', '130px');
}

function drawHUD(t, l, f, fk, effFk) {
  push();
  fill(0, 190);
  noStroke();
  rect(0, height - 32, width, 32);

  fill(255);
  textSize(11);
  textAlign(LEFT, CENTER);
  text(`Time (0..1023): ${t}`, 15, height - 16);
  text(`Logic: ${l}`, 165, height - 16);
  text(`Fact-Check: ${f}`, 315, height - 16);
  text(`Fake: ${fk} [Act: ${Math.round(effFk)}]`, 465, height - 16);
  pop();
}