function createControlSliders() {
  // Time (0..1023)
  sliderTime = createSlider(0, 1023, 0, 1);
  sliderTime.position(20, height + 15);
  sliderTime.style('width', '130px');

  // Slider "Logic" (A1)
  sliderLogic = createSlider(0, 1023, 0, 1);
  sliderLogic.position(170, height + 15);
  sliderLogic.style('width', '130px');

  // Slider "Fact-checking" (A2)
  sliderFact = createSlider(0, 1023, 0, 1);
  sliderFact.position(320, height + 15);
  sliderFact.style('width', '130px');

  // Slider "Fake Generation" (A3)
  sliderFake = createSlider(0, 1023, 0, 1);
  sliderFake.position(470, height + 15);
  sliderFake.style('width', '130px');
}

function drawTimeline(rawTime) {
  push();
  fill(50, 200);
  noStroke();
  rect(50, 50, 540, 10, 5);

  let knobX = map(rawTime, 0, 1023, 50, 590);
  let currentYear = Math.round(map(rawTime, 0, 1023, 1960, 2026));

  fill(255);
  circle(knobX, 55, 16);

  fill(220);
  textSize(13);
  textAlign(CENTER, BOTTOM);
  text("1960", 50, 42);
  text("1980", 158, 42);
  text("1990", 266, 42);
  text("2000", 374, 42);
  text("2010", 482, 42);
  text("2026", 590, 42);

  fill(0, 255, 200);
  textSize(14);
  textStyle(BOLD);
  textAlign(CENTER, TOP);
  text(currentYear, knobX, 70);
  pop();
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