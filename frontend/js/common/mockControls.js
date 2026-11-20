
function initMockControls(actions = {}) {
    const oldPanel = document.getElementById("mock-controls");
    if (oldPanel) oldPanel.remove();

    const panel = document.createElement("section");
    panel.id = "mock-controls";

    panel.innerHTML = `
        <h3>DIGITAL MIRROR CONTROLLER</h3>
        <p>Hardware simulation mode</p>

        <label for="trust-slider">
            HOW MUCH DO YOU TRUST THIS ARTICLE?
        </label>

        <div class="trust-control">
            <input
                id="trust-slider"
                type="range"
                min="0"
                max="1"
                step="0.01"
                value="0.5"
            >
            <output id="trust-value">0.50</output>
        </div>

        <div class="trust-labels">
            <span>DON'T BELIEVE</span>
            <span>UNSURE</span>
            <span>BELIEVE</span>
        </div>

        <div class="navigation-buttons">
            <button id="article-back" type="button">BACK</button>
            <button id="article-select" type="button">SELECT</button>
            <button id="article-next" type="button">NEXT</button>
        </div>
    `;

    document.body.appendChild(panel);

    const slider = document.getElementById("trust-slider");
    const output = document.getElementById("trust-value");

    slider.addEventListener("input", () => {
        const trust = Number(slider.value);
        output.textContent = trust.toFixed(2);
        actions.onTrustChange?.(trust);
    });

    document.getElementById("article-back")
        .addEventListener("click", () => actions.onBack?.());

    document.getElementById("article-select")
        .addEventListener("click", () => {
            actions.onSelect?.(Number(slider.value));
        });

    document.getElementById("article-next")
        .addEventListener("click", () => actions.onNext?.());

    actions.onTrustChange?.(Number(slider.value));
}
