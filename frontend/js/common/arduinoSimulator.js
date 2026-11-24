class ArduinoSimulator {

    constructor(flow) {
        this.flow = flow;
        this.busy = false;

        this.createPanel();
        this.updateUI();
    }

    createPanel() {
        const oldPanel = document.getElementById("arduino-simulator");
        if (oldPanel) oldPanel.remove();

        this.panel = document.createElement("section");
        this.panel.id = "arduino-simulator";

        this.panel.innerHTML = `

            <div id="statement-container">

                <div id="statement-label">
                    STATEMENT
                </div>

                <div id="spoken-statement">
                    Press START to begin.
                </div>

            </div>

            <h3>DIGITAL MIRROR CONTROLLER</h3>
            <p>Arduino simulation mode</p>

            <div id="sim-phase">PHASE: IDLE</div>

            <label for="sim-pot">
                HOW MUCH DO YOU BELIEVE THIS CLAIM?
            </label>

            <div class="trust-control">
                <input
                    id="sim-pot"
                    type="range"
                    min="0"
                    max="1023"
                    step="1"
                    value="512"
                >
                <output id="sim-score">0.50</output>
            </div>

            <div class="trust-labels">
                <span>REFUTES</span>
                <span>UNSURE</span>
                <span>SUPPORTS</span>
            </div>

            <div class="navigation-buttons">
                <button id="sim-next" type="button">
                    START/NEXT
                </button>

                <button id="sim-confirm" type="button">
                    CONFIRM
                </button>
            </div>

            <div id="sim-status" aria-live="polite">
                Ready
            </div>
        `;

        document.body.appendChild(this.panel);
        this.informationVideo = new InformationVideo();

        this.pot = this.panel.querySelector("#sim-pot");
        this.nextButton = this.panel.querySelector("#sim-next");
        this.confirmButton = this.panel.querySelector("#sim-confirm");
        this.status = this.panel.querySelector("#sim-status");

        this.pot.addEventListener("input", () => {
            const score = Number(this.pot.value) / 1023;

            this.flow.setHumanScore(score);
            // window.setHumanPreviewIntensity?.(score);
            this.updateUI();
        });


        this.nextButton.addEventListener("click", async () => {
            await this.run(async () => {

                // 1. Load and analyze the next claim
                window.machineMask?.stopSpeech?.();
                await this.flow.next();

                // 2. Reset previous visual distortions
                window.resetHumanMirror?.();
                window.machineFrame?.reset();
                window.machineInterference?.setDistortion(0);

                // 3. Update machine prediction
                window.machineMask?.setAnalysis?.(this.flow.analysis);

                // 4. Reset human evaluation to the middle position
                this.pot.value = 512;
                this.flow.setHumanScore(512 / 1023);
                //window.setHumanPreviewIntensity?.(512 / 1023);

                // 5. Display the new claim
                await this.informationVideo.loadClaim(
                    this.flow.claim
                );

                this.status.textContent = "Evaluate the statement.";
            });
        });



        this.confirmButton.addEventListener("click", async () => {
            await this.run(async () => {

                // 1. Send human evaluation to Spring Boot.
                // Backend compares human and machine predictions
                // against verified truth.
                const result = await this.flow.confirm();

                const humanDistortion = Number(result.humanDistortion);
                const machineDistortion = Number(result.machineDistortion);

                if (
                    !Number.isFinite(humanDistortion) ||
                    !Number.isFinite(machineDistortion)
                ) {
                    throw new Error("Invalid distortion values from backend");
                }

                console.log("CONFIRM RESULT:", result);

                // 2. Human Mirror
                window.setHumanDistortion?.(humanDistortion);

                // 3. Machine Mirror — border needles
                window.machineFrame?.setDistortion(machineDistortion);

                // 4. Machine Mirror — CRT interference
                window.machineInterference?.setDistortion(machineDistortion);

                // 5. Display results
                this.status.textContent =
                    `Human distortion: ${humanDistortion.toFixed(2)}
             | Machine distortion: ${machineDistortion.toFixed(2)}`;

                const reveal = await this.flow.reveal();

                console.log("REVEAL:", reveal);

                // Machine emotional dialogue
                await window.machineMask?.speakPrediction?.(
                    result.humanScore,
                    result.machineScore,
                    reveal.verifiedLabel
                );
            });
        });

    }

    async run(action) {
        if (this.busy) return;

        this.busy = true;
        this.updateUI();

        try {
            await action();
        } catch (error) {
            console.error(error);
            this.status.textContent = error.message;
        } finally {
            this.busy = false;
            this.updateUI();
        }
    }

    updateUI() {
        const phase = this.flow.phase;

        this.panel.querySelector("#sim-phase").textContent =
            `PHASE: ${phase}`;

        this.panel.querySelector("#sim-score").textContent =
            this.flow.humanScore.toFixed(2);

        this.pot.disabled =
            this.busy || phase !== "CHOOSING";

        this.nextButton.disabled =
            this.busy ||
            phase === "ANALYZING" ||
            phase === "CONFIRMING" ||
            phase === "ANIMATING";

        this.confirmButton.disabled =
            this.busy || phase !== "CHOOSING";
    }
}

window.addEventListener("DOMContentLoaded", () => {
    window.arduinoSimulator =
        new ArduinoSimulator(interactionFlow);
});
