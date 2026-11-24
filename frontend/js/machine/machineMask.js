const machineMaskSketch = (p) => {
    const containerId = "side-canvas-container";

    let faceMesh = [];
    let hud;
    let machineSpeech;

    let speaking = false;
    let mouthLevel = 0;

    const ROWS = 28;
    const COLS = 34;


    const FACE_W = 145;
    const FACE_H = 205;
    const FACE_D = 75;
    const FACE_SCALE = 0.80;

    let machineSupport = 74;
    let machineRefute = 26;

    p.setup = function setup() {
        p.createCanvas(640, 480, p.WEBGL);

        p.ortho(
            -p.width / 2,
            p.width / 2,
            -p.height / 2,
            p.height / 2,
            -1000,
            1000
        );

        p.pixelDensity(1);
        p.frameRate(30);

        hud = p.createGraphics(640, 480);
        hud.pixelDensity(1);

        createFace();

        machineSpeech = new MachineSpeech((isSpeaking) => {
            speaking = isSpeaking;
        });
    };

// --------------------------------------
// FACE GEOMETRY
// --------------------------------------

    function createFace() {
        faceMesh = [];

        for (let i = 0; i <= ROWS; i++) {
            let row = [];

            let v = p.map(i, 0, ROWS, -1, 1);

            for (let j = 0; j <= COLS; j++) {
                let u = p.map(j, 0, COLS, -1, 1);

                // Human face outline
                let outline = p.sqrt(p.max(0, 1 - v * v));

                // Narrower jaw
                let jaw = p.map(
                    p.constrain(v, 0, 1),
                    0, 1,
                    1, 0.72
                );

                let x = u * FACE_W * outline * jaw;
                let y = v * FACE_H;

                // Base curved surface
                let z = FACE_D * p.sqrt(p.max(0, 1 - u * u));

                // Forehead
                z += 12 * gauss(v, -0.65, 0.3);

                // Nose bridge and tip
                z += 30 *
                    gauss(u, 0, 0.14) *
                    gauss(v, 0.02, 0.34);

                z += 25 *
                    gauss(u, 0, 0.18) *
                    gauss(v, 0.25, 0.13);

                // Eye sockets
                z -= 22 *
                    gauss(p.abs(u), 0.43, 0.18) *
                    gauss(v, -0.25, 0.11);

                // Brow ridges
                z += 9 *
                    gauss(p.abs(u), 0.43, 0.23) *
                    gauss(v, -0.39, 0.07);

                // Cheekbones
                z += 13 *
                    gauss(p.abs(u), 0.52, 0.22) *
                    gauss(v, 0.08, 0.20);

                // Mouth indentation
                z -= 13 *
                    gauss(u, 0, 0.37) *
                    gauss(v, 0.52, 0.09);

                // Lips
                z += 7 *
                    gauss(u, 0, 0.30) *
                    gauss(v, 0.46, 0.04);

                z += 8 *
                    gauss(u, 0, 0.30) *
                    gauss(v, 0.59, 0.05);

                // Chin
                z += 18 *
                    gauss(u, 0, 0.38) *
                    gauss(v, 0.79, 0.15);

                row.push({
                    x, y, z, u, v,
                    phase: p.random(p.TWO_PI)
                });
            }

            faceMesh.push(row);
        }
    }

    function gauss(x, center, spread) {
        let d = (x - center) / spread;
        return p.exp(-d * d);
    }

// --------------------------------------
// MAIN DRAW
// --------------------------------------

    p.draw = function draw() {
        try {
            p.background(3, 7, 14);

            const time = p.frameCount * 0.2;

            const target = speaking
                ? 7 + 9 * p.abs(p.sin(time * 3))
                : 0;

            mouthLevel = p.lerp(mouthLevel, target, 0.4);

            p.push();

            try {
                const layout = calculateFaceLayout();

                p.translate(layout.x, layout.y, 0);

                p.rotateY(p.sin(p.frameCount * 0.005) * 0.20);
                p.rotateX(-0.04);

                p.scale(layout.scale);

                drawFace(time);

            } finally {
                p.pop();
            }

            drawHUD();

        } catch (error) {
            console.error("MACHINE MASK ERROR:", error);
            p.noLoop();
        }
    };

    function drawFace(time) {
        let points = [];

        // Calculate animated geometry once
        for (let i = 0; i <= ROWS; i++) {
            let row = [];

            for (let j = 0; j <= COLS; j++) {
                let vertex = faceMesh[i][j];

                let x = vertex.x;
                let y = vertex.y;
                let z = vertex.z;

                z += 0.7 * p.sin(time + vertex.phase);

                // Mouth deformation
                let mouth =
                    gauss(vertex.u, 0, 0.25) *
                    gauss(vertex.v, 0.53, 0.13);

                let lipDirection = vertex.v > 0.53 ? 1 : -1;

                y += mouthLevel * mouth * lipDirection;

                z -= mouthLevel * mouth * 0.25;

                row.push({
                    x, y, z,
                    phase: vertex.phase
                });
            }

            points.push(row);
        }

        // Mesh lines
        p.noFill();
        p.stroke(70, 150, 190, 95);
        p.strokeWeight(0.6);

        p.beginShape(p.LINES);

        for (let i = 0; i <= ROWS; i++) {
            for (let j = 0; j <= COLS; j++) {
                let vertex = points[i][j];

                if (j < COLS) {
                    let r = points[i][j + 1];

                    p.vertex(vertex.x, vertex.y, vertex.z);
                    p.vertex(r.x, r.y, r.z);
                }

                if (i < ROWS) {
                    let below = points[i + 1][j];

                    p.vertex(vertex.x, vertex.y, vertex.z);
                    p.vertex(below.x, below.y, below.z);
                }
            }
        }

        p.endShape();

        // Points
        p.stroke(175, 225, 245, 190);
        p.strokeWeight(2);

        p.beginShape(p.POINTS);

        for (let i = 0; i <= ROWS; i++) {
            for (let j = 0; j <= COLS; j++) {
                let vertex = points[i][j];

                p.vertex(vertex.x, vertex.y, vertex.z);
            }
        }

        p.endShape();

        drawEyes(time);
    }

    function drawHUD() {
        hud.clear();

        hud.textAlign(p.CENTER);
        hud.textFont("monospace");
        hud.noStroke();

        hud.fill(170, 205, 225);
        hud.textSize(18);
        hud.text("MACHINE MIRROR", 320, 34);

        hud.fill(100, 155, 180);
        hud.textSize(13);
        hud.text("EVIDENCE ANALYSIS", 320, 420);

        hud.fill(195, 230, 245);
        hud.textSize(15);

        hud.text(
            `SUPPORTS ${machineSupport}%  |  REFUTES ${machineRefute}%`,
            320,
            450
        );

        p.push();

        p.resetMatrix();
        p.noStroke();

        // Render HUD as a texture on a WEBGL plane
        p.texture(hud);
        p.translate(0, 0, 250);
        p.plane(p.width, p.height);

        p.pop();
    }

    function drawEyes(time) {
        const eyeY = -15;
        const eyeZ = 80;
        const eyeSpacing = 60;

        for (const side of [-1, 1]) {
            p.push();

            p.translate(side * eyeSpacing, eyeY, eyeZ);

            // Subtle eye movement
            const lookX = p.sin(time * 0.7) * 2;
            const lookY = p.cos(time * 0.5) * 1.5;

            // Outer eye shape
            p.noFill();
            p.stroke(80, 175, 215, 170);
            p.strokeWeight(1.3);

            p.beginShape();
            for (let a = 0; a <= p.TWO_PI + 0.01; a += 0.12) {
                const x = 40 * p.cos(a);
                const y = 28 * p.sin(a);
                p.vertex(x, y, 0);
            }
            p.endShape(p.CLOSE);

            // Iris rings
            p.push();
            p.translate(lookX, lookY, 3);

            p.noFill();

            for (let radius of [18, 10]) {
                p.stroke(110, 215, 245, 180);
                p.strokeWeight(1);

                p.beginShape();
                for (let a = 0; a <= p.TWO_PI + 0.01; a += 0.2) {
                    p.vertex(
                        radius * p.cos(a),
                        radius * p.sin(a),
                        0
                    );
                }
                p.endShape(p.CLOSE);
            }

            // Dark pupil
            p.noStroke();
            p.fill(5, 15, 25);
            p.ellipse(0, 0, 7, 7);

            // Central glowing p.point
            p.stroke(220, 245, 255);
            p.strokeWeight(3);
            p.point(0, 0, 5);

            p.pop();
            p.pop();
        }
    }

    function calculateFaceLayout() {
        const availableWidth = p.width * 0.75;
        const availableHeight = p.height * 0.75;

        const faceWidth = FACE_W * 2;
        const faceHeight = FACE_H * 2;

        const scaleX = availableWidth / faceWidth;
        const scaleY = availableHeight / faceHeight;

        return {
            scale: Math.min(scaleX, scaleY),
            x: 0,
            y: 0
        };
    }

    function speakMachine() {
        machineSpeech.speak(machineSupport);
    }

    // Integration hook: call machineMask.setAnalysis(analysis) after ANALYZE.
    p.setAnalysis = (analysis) => {
        if (!analysis) return;
        const support = Number(analysis.pSupports);
        const refute = Number(analysis.pRefutes);
        if (!Number.isFinite(support) || !Number.isFinite(refute)) return;
        machineSupport = Math.round(p.constrain(support, 0, 1) * 100);
        machineRefute = Math.round(p.constrain(refute, 0, 1) * 100);
    };

    p.speakPrediction = (human, machine, verifiedLabel) => {

        return machineSpeech.speakDialogue(
            human,
            machine,
            verifiedLabel
        );
    };
    p.stopSpeech = () => machineSpeech?.stop();
};

window.machineMask = new p5(
    machineMaskSketch,
    "side-canvas-container"
);
