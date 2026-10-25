// Pre-rendered buffer for newsprint texture
let paperTexture = null;

// Generates static grain once at setup
function initPaperTexture(p, width, height) {
    paperTexture = p.createGraphics(width, height);
    paperTexture.pixelDensity(1);
    paperTexture.loadPixels();

    for (let i = 0; i < paperTexture.pixels.length; i += 4) {
        let noiseVal = Math.floor(Math.random() * 45);
        paperTexture.pixels[i] = 40;            // R
        paperTexture.pixels[i + 1] = 30;        // G
        paperTexture.pixels[i + 2] = 20;        // B
        paperTexture.pixels[i + 3] = noiseVal;  // Alpha
    }

    paperTexture.updatePixels();
}

// Blends grain onto paper background
function applyPaperGrain(p) {
    if (!paperTexture) return;

    p.push();
    p.blendMode(p.MULTIPLY);
    p.image(paperTexture, 0, 0);
    p.pop();
}