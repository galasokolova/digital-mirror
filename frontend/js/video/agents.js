function spawnEyeSpores(faces, distortionLevel, agentsList, hasPixels) {

    if (faces.length === 0 || distortionLevel <= 0.01 || !hasPixels) {
        return;
    }

    const keypoints = faces[0].keypoints;

    // Central points of eyes in FaceMesh (468, 473) or eyelids (159, 386)
    const leftEye = keypoints[468] || keypoints[159];
    const rightEye = keypoints[473] || keypoints[386];

    // Possible spawn with population limit
    if (random() < distortionLevel && agentsList.length < 30) {
        if (leftEye) {
            agentsList.push(new NeedleAgent(leftEye.x, leftEye.y));
        }
        if (rightEye) {
            agentsList.push(new NeedleAgent(rightEye.x, rightEye.y));
        }
    }
}

function spawnBorderSpores(agentsList, maxBorderAgents = 15) {
    const maxCap = (typeof CONFIG !== 'undefined' && CONFIG.spores?.maxAgents) ? CONFIG.spores.maxAgents : 30;

    if (agentsList.length >= maxCap + maxBorderAgents) return;

    if (random() < 0.2) {
        let edge = floor(random(4));
        let spawnX, spawnY;

        if (edge === 0) {
            spawnX = random(width);
            spawnY = random(0, 10);
        } else if (edge === 1) {
            spawnX = random(width - 10, width);
            spawnY = random(height);
        } else if (edge === 2) {
            spawnX = random(width);
            spawnY = random(height - 10, height);
        } else {
            spawnX = random(0, 10);
            spawnY = random(height);
        }

        let borderAgent = new NeedleAgent(spawnX, spawnY);
        borderAgent.life = 250; // Give them more time to grow

        // Force neon phosphor color:
        borderAgent.r = 0;
        borderAgent.g = 255;
        borderAgent.b = 200;

        // Increase base needle span:
        borderAgent.diameter = 8;

        agentsList.push(borderAgent);

    }
}


class NeedleAgent {
    constructor(startX, startY) {
        this.x = startX;
        this.y = startY;
        this.stepSize = 4;
        this.diameter = 5;

        // Initial color — black
        this.r = 0;
        this.g = 0;
        this.b = 0;

        this.life = 200;
        this.directions = ["NW", "N", "NE", "W", "E", "SW", "S", "SE"];
    }

    update(intensity, pixels) {
        // Step count scales with lie/distortion intensity
        let steps = int(map(intensity, 0, 1, 3, 20));

        for (let i = 0; i < steps; i++) {
            let choice = random(this.directions);

            if (choice === "NW")      { this.x -= this.stepSize; this.y -= this.stepSize; }
            else if (choice === "N")  { this.y -= this.stepSize; }
            else if (choice === "NE") { this.x += this.stepSize; this.y -= this.stepSize; }
            else if (choice === "W")  { this.x -= this.stepSize; }
            else if (choice === "E")  { this.x += this.stepSize; }
            else if (choice === "SW") { this.x -= this.stepSize; this.y += this.stepSize; }
            else if (choice === "S")  { this.y += this.stepSize; }
            else if (choice === "SE") { this.x += this.stepSize; this.y += this.stepSize; }

            // Screen boundary constraint
            this.x = constrain(this.x, 0, width - 1);
            this.y = constrain(this.y, 0, height - 1);

            // Read pixel color underneath the agent
            let pxIndex = (int(this.x) + int(this.y) * width) * 4;
            let pr = pixels[pxIndex];
            let pg = pixels[pxIndex + 1];
            let pb = pixels[pxIndex + 2];

            // Mutate palette on color similarity
            if (dist(pr, pg, pb, this.r, this.g, this.b) < 30) {
                this.r = (this.r + 0.2) % 256;
                this.g = (this.g + 0.5) % 256;
                this.b = (this.b + 1.2) % 256;
            }
        }

        // Aging rate depends on intensity
        this.life -= map(intensity, 0, 1, 3, 0.8);
    }

    render(pg, intensity) {
        let check_x = this.x;
        let check_y = this.y;

        let lenSmall = (this.diameter + 5) * map(intensity, 0, 1, 0.8, 2.0);
        let lenBig = (this.diameter + 15) * map(intensity, 0, 1, 0.8, 2.5);

        pg.strokeWeight(1);

        // Secondary emerald bursts
        if (int(check_x) % 3 === 0 || int(check_y) % 3 === 0) {
            pg.stroke(0, 220, 80, this.life * 0.7);
            pg.line(
                check_x,
                check_y,
                check_x + random(-lenBig, lenBig),
                check_y + random(-lenBig, lenBig)
            );
        }

        // Rare long blue spikes
        if (int(check_x) % 20 === 0 || int(check_y) % 20 === 0) {
            pg.stroke(20, 80, 255, this.life * 0.5);
            pg.line(
                check_x,
                check_y,
                check_x + random(-lenBig * 1.5, lenBig * 1.5),
                check_y + random(-lenBig * 1.5, lenBig * 1.5)
            );
        }

        // Primary needle with current mutated color
        pg.stroke(this.r, this.g, this.b, this.life * 0.9);
        pg.line(
            check_x,
            check_y,
            check_x + random(-lenSmall, lenSmall),
            check_y + random(-lenSmall, lenSmall)
        );
    }

    isDead() {
        return this.life <= 0;
    }
}