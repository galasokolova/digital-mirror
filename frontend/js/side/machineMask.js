const machineMaskSketch = (p) => {
    const containerId = "side-canvas-container";
    let stars = [];
    let cosmicParticles = [];
    let cosmicGlow;

    const STAR_COUNT = 250;
    const PARTICLE_COUNT = 90;

    let faceMesh = [];
    let hud;
    let speakButton;

    let speaking = false;
    let mouthLevel = 0;

    const ROWS = 36;
    const COLS = 44;

    const FACE_W = 145;
    const FACE_H = 205;
    const FACE_D = 75;

    let machineSupport = 74;
    let machineRefute = 26;

    p.setup = function setup() {
        p.createCanvas(640, 480, p.WEBGL);

        createCosmos();

        cosmicGlow = p.createGraphics(p.width, p.height);
        cosmicGlow.pixelDensity(1);

        p.pixelDensity(1);
        p.frameRate(30);


        hud = p.createGraphics(640, 480);
        hud.pixelDensity(1);

        createFace();

        speakButton = p.createButton("SPEAK");
        speakButton.parent(containerId);
        speakButton.addClass("machine-speak-button");
        speakButton.mousePressed(speakMachine);
    }

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

            console.log("1. Background OK");

            drawCosmos();
            console.log("2. Cosmos OK");

            let time = p.frameCount * 0.2;

            let target = speaking
                ? 7 + 9 * p.abs(p.sin(time * 3))
                : 0;

            mouthLevel = p.lerp(mouthLevel, target, 0.4);

            p.push();

            try {
                p.translate(0, 8, 0);
                p.rotateY(p.sin(p.frameCount * 0.005) * 0.20);
                p.rotateX(-0.04);

                drawFace(time);
                console.log("3. Face OK");
            } finally {
                p.pop();
            }

            drawHUD();
            console.log("4. HUD OK");

        } catch (error) {
            console.error("MACHINE MASK ERROR:", error);
            p.noLoop();
        }
    };


// --------------------------------------
// FACE RENDERING
// --------------------------------------

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

// --------------------------------------
// HUD
// --------------------------------------


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
                const x = 35 * p.cos(a);
                const y = 21 * p.sin(a);
                p.vertex(x, y, 0);
            }
            p.endShape(p.CLOSE);

            // Iris rings
            p.push();
            p.translate(lookX, lookY, 3);

            p.noFill();

            for (let radius of [9, 6]) {
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


// --------------------------------------
// SPEECH
// --------------------------------------

    function speakMachine() {
        if (!("speechSynthesis" in window)) {
            console.log("Speech not supported");
            return;
        }

        window.speechSynthesis.cancel();

        const message =
            `My analysis indicates ${machineSupport} percent ` +
            `support and ${machineRefute} percent contradiction.`;

        const utterance = new SpeechSynthesisUtterance(message);

        utterance.lang = "en-US";
        utterance.rate = 0.82;
        utterance.pitch = 0.65;

        utterance.onstart = () => {
            speaking = true;
        };

        utterance.onend = () => {
            speaking = false;
        };

        utterance.onerror = () => {
            speaking = false;
        };

        window.speechSynthesis.speak(utterance);
    }


    function createCosmos() {
        stars = [];
        cosmicParticles = [];

        // Distant stars
        for (let i = 0; i < STAR_COUNT; i++) {
            stars.push({
                x: p.random(-p.width / 2, p.width / 2),
                y: p.random(-p.height / 2, p.height / 2),
                z: p.random(-500, -250),
                size: p.random(0.6, 2.2),
                phase: p.random(p.TWO_PI),
                speed: p.random(0.4, 1.5)
            });
        }

        // Orbiting cosmic particles
        for (let i = 0; i < PARTICLE_COUNT; i++) {
            cosmicParticles.push({
                angle: p.random(p.TWO_PI),
                radius: p.random(225, 340),
                depth: p.random(-170, -80),
                speed: p.random(0.0004, 0.002),
                size: p.random(1, 2.8),
                phase: p.random(p.TWO_PI)
            });
        }
    }

// --------------------------------------
// COSMIC BACKGROUND
// --------------------------------------

    function drawCosmos() {
        const t = p.frameCount * 0.015;

        p.push();

        // Soft nebula glow behind face
        //drawNebula();

        // Stars
        p.noFill();

        for (const s of stars) {
            const flicker =
                90 + 130 * p.abs(p.sin(t * s.speed + s.phase));

            p.stroke(170, 205, 255, flicker);
            p.strokeWeight(s.size);

            p.point(s.x, s.y, s.z);
        }

        // Orbiting particles

        for (const particle of cosmicParticles) {
            const a = particle.angle + p.frameCount * particle.speed;

            const x = p.cos(a) * particle.radius;
            const y = p.sin(a) * particle.radius * 0.85;

            const alpha =
                80 + 110 * p.abs(p.sin(t + particle.phase));

            p.stroke(125, 170, 255, alpha);
            p.strokeWeight(particle.size);

            p.point(x, y, particle.depth);
        }


        p.pop();
    }

// --------------------------------------
// NEBULA GLOW
// --------------------------------------


    function drawNebula() {
        const ctx = cosmicGlow.drawingContext;

        cosmicGlow.clear();

        const cx = p.width / 2;
        const cy = p.height / 2;

        const pulse = 1 + 0.05 * p.sin(p.frameCount * 0.01);

        const gradient = ctx.createRadialGradient(
            cx, cy, 20,
            cx, cy, 300 * pulse
        );

        gradient.addColorStop(
            0,
            "rgba(45, 85, 160, 0.23)"
        );

        gradient.addColorStop(
            0.45,
            "rgba(65, 35, 135, 0.15)"
        );

        gradient.addColorStop(
            1,
            "rgba(0, 0, 0, 0)"
        );

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, p.width, p.height);

        p.push();

        // Draw the glow as a texture on a WEBGL plane
        p.noStroke();
        p.texture(cosmicGlow);
        p.translate(0, 0, -400);
        p.plane(p.width, p.height);

        p.pop();
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

    p.speakPrediction = () => speakMachine();
};

const machineMask = new p5(machineMaskSketch, "side-canvas-container");
