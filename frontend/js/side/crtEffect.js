// Pre-rendered buffer for static scanlines and vignette
let crtTexture = null;

// Generates scanlines and radial vignette once at setup
function initCrtTexture(p, width, height) {
    crtTexture = p.createGraphics(width, height);
    crtTexture.pixelDensity(1);

    // 1. Horizontal scanlines
    crtTexture.stroke(0, 0, 0, 75);
    crtTexture.strokeWeight(1);
    for (let y = 0; y < height; y += 3) {
        crtTexture.line(0, y, width, y);
    }

    // 2. Radial vignette for curved CRT screen illusion
    let ctx = crtTexture.drawingContext;
    let radGrad = ctx.createRadialGradient(
        width / 2, height / 2, width * 0.35,
        width / 2, height / 2, width * 0.72
    );
    radGrad.addColorStop(0, 'rgba(0, 0, 0, 0)');
    radGrad.addColorStop(1, 'rgba(0, 0, 0, 0.65)');

    ctx.fillStyle = radGrad;
    ctx.fillRect(0, 0, width, height);
}

// Subtle coordinate background grid behind text
function drawCyberBackgroundGrid(p, width, height) {
    p.push();
    p.stroke(0, 220, 180, 15);
    p.strokeWeight(1);

    for (let x = 40; x < width - 40; x += 40) {
        p.line(x, 0, x, height);
    }
    for (let y = 0; y < height; y += 40) {
        p.line(0, y, width, y);
    }
    p.pop();
}

// Top-layer post-processing: scanlines and moving refresh bar
function applyCrtPostProcess(p, width, height) {
    p.push();

    // Draw static scanlines and vignette
    if (crtTexture) {
        p.image(crtTexture, 0, 0);
    }

    // Animated refresh roll-bar scrolling downwards
    let rollY = (p.frameCount * 2) % (height + 60) - 30;
    p.noStroke();
    p.fill(0, 255, 200, 14);
    p.rect(0, rollY, width, 20);

    p.pop();
}