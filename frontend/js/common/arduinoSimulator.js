class ArduinoSimulator {

    constructor(flow) {
        this.flow = flow;
        this.busy = false;

        this.createPanel();
        this.updateUI();
    }

    createPanel() {
        this.panel = document.createElement("div");
        this.panel.id = "arduino-simulator";

        Object.assign(this.panel.style, {
            position: "fixed",
            bottom: "20px",
            left: "20px",
            zIndex: "9999",
            background: "rgba(15, 20, 35, 0.95)",
            color: "white",
            padding: "16px",
            borderRadius: "12px",
            fontFamily: "Arial, sans-serif",
            width: "270px",
            boxShadow: "0 4px 20px rgba(0,0,0,0.4)"
        });

        this.panel.innerHTML = `
      <div style="font-weight:bold;margin-bottom:12px">
        ARDUINO SIMULATOR
      </div>

      <div id="sim-phase" style="margin-bottom:10px">
        IDLE
      </div>

      <div style="display:flex;justify-content:space-between">
        <span>REFUTES</span>
        <span id="sim-score">0.50</span>
        <span>SUPPORTS</span>
      </div>

      <input
        id="sim-pot"
        type="range"
        min="0"
        max="1023"
        value="512"
        style="width:100%;margin:12px 0"
      >

      <div style="display:flex;gap:10px">
        <button id="sim-next">NEXT</button>
        <button id="sim-confirm">CONFIRM</button>
      </div>

      <div id="sim-status"
           style="margin-top:12px;font-size:12px">
        Ready
      </div>
    `;

        document.body.appendChild(this.panel);

        this.pot = this.panel.querySelector("#sim-pot");
        this.nextButton = this.panel.querySelector("#sim-next");
        this.confirmButton = this.panel.querySelector("#sim-confirm");
        this.status = this.panel.querySelector("#sim-status");

        this.pot.addEventListener("input", () => {
            const score = Number(this.pot.value) / 1023;

            this.flow.setHumanScore(score);
            this.updateUI();
        });

        this.nextButton.addEventListener("click", async () => {
            await this.run(async () => {
                await this.flow.next();
                this.pot.value = 512;
                this.flow.setHumanScore(512 / 1023);
                this.status.textContent = this.flow.claim.text;
            });
        });

        this.confirmButton.addEventListener("click", async () => {
            await this.run(async () => {
                const result = await this.flow.confirm();

                this.status.textContent =
                    `Human distortion: ${result.humanDistortion.toFixed(2)}
           | Machine distortion: ${result.machineDistortion.toFixed(2)}`;

                // Temporary: reveal immediately after confirmation.
                // Later the animation will run between CONFIRM and REVEAL.
                await this.flow.reveal();

                console.log("REVEAL:", this.flow.revealData);
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

        this.pot.disabled = this.busy || phase !== "CHOOSING";

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
