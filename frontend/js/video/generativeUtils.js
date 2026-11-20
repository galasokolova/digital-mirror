function createErosionLayer(w, h) {
    const layer = createGraphics(w, h);
    layer.clear();
    return layer;
}

function showLoadingScreen(ready, message = "Initializing Digital Mirror...") {
    if (ready) return false;

    push();
    // dark padding
    noStroke();
    fill(0, 180);
    rect(20, 20, 260, 36, 4);

    // Status text
    fill(0, 255, 180);
    textSize(14);
    textAlign(LEFT, CENTER);
    text(message, 32, 38);
    pop();

    return true;
}

function createVignetteLayer(w, h, depth = 60) {
    let layer = createGraphics(w, h);
    layer.noFill();
    for (let r = 0; r < depth; r += 5) {
        layer.stroke(0, map(r, 0, depth, 220, 0));
        layer.strokeWeight(6);
        layer.rect(r, r, w - r * 2, h - r * 2);
    }
    return layer;
}