/**
 * DIGITAL MIRROR — MACHINE INTERFERENCE
 * CRT-style horizontal bands made of tiny needles.
 *
 * Distortion range: 0..1
 * 0 = no interference
 * 1 = strong interference
 */

class MachineInterference {

    constructor(container, options = {}) {
        if (!container) {
            throw new Error("MachineInterference requires a container");
        }

        this.container = container;
        this.distortion = 0;
        this.bands = [];
        this.frame = 0;
        this.maxBands = options.maxBands ?? 7;
        this.running = true;

        this.canvas = document.createElement("canvas");
        this.canvas.className = "machine-interference-overlay";
        this.canvas.setAttribute("aria-hidden", "true");

        Object.assign(this.canvas.style, {
            position: "absolute",
            inset: "0",
            width: "100%",
            height: "100%",
            pointerEvents: "none",
            zIndex: "5"
        });

        if (getComputedStyle(container).position === "static") {
            container.style.position = "relative";
        }

        container.appendChild(this.canvas);
        this.ctx = this.canvas.getContext("2d");

        this.resize = this.resize.bind(this);
        this.animate = this.animate.bind(this);

        this.resize();

        this.observer = typeof ResizeObserver !== "undefined"
            ? new ResizeObserver(this.resize)
            : null;

        if (this.observer) {
            this.observer.observe(this.container);
        } else {
            window.addEventListener("resize", this.resize);
        }

        this.rafId = requestAnimationFrame(this.animate);
    }

    setDistortion(value) {
        const numeric = Number(value);

        this.distortion = Number.isFinite(numeric)
            ? Math.max(0, Math.min(1, numeric))
            : 0;

        if (this.distortion === 0) {
            this.reset();
        }
    }

    reset() {
        this.bands.length = 0;
        this.ctx.clearRect(0, 0, this.width, this.height);
    }

    resize() {
        const rect = this.container.getBoundingClientRect();

        this.width = Math.max(1, Math.round(rect.width));
        this.height = Math.max(1, Math.round(rect.height));

        const dpr = Math.min(window.devicePixelRatio || 1, 2);

        this.canvas.width = Math.round(this.width * dpr);
        this.canvas.height = Math.round(this.height * dpr);

        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this.bands.length = 0;
    }

    spawnBand() {
        const d = this.distortion;

        if (d <= 0 || this.bands.length >= this.maxBands) {
            return;
        }

        // Short bursts rather than continuous full-screen noise.
        if (Math.random() > 0.012 + d * 0.12) {
            return;
        }

        this.bands.push({
            y: -8 - Math.random() * this.height * 0.18,
            height: 4 + d * 24 + Math.random() * 12,
            speed: 0.8 + d * 3.4 + Math.random() * 1.5,
            drift: (Math.random() - 0.5) * (4 + 14 * d),
            phase: Math.random() * Math.PI * 2,
            opacity: 0.35 + Math.random() * 0.5
        });
    }

    drawBand(band) {
        const ctx = this.ctx;
        const d = this.distortion;
        const y = band.y;
        const bandHeight = band.height;

        ctx.save();

        ctx.beginPath();
        ctx.rect(0, y, this.width, bandHeight);
        ctx.clip();

        // Soft phosphor glow.
        const glow = ctx.createLinearGradient(
            0, y, 0, y + bandHeight
        );

        glow.addColorStop(0, "rgba(0,225,195,0)");
        glow.addColorStop(
            0.5,
            `rgba(0,225,195,${0.035 + d * 0.10})`
        );
        glow.addColorStop(1, "rgba(0,225,195,0)");

        ctx.fillStyle = glow;
        ctx.fillRect(0, y, this.width, bandHeight);

        // Tiny horizontal and slightly slanted needles.
        const count = Math.round(
            (16 + d * 150) * (this.width / 640)
        );

        const needleLength = 1.5 + d * 5;

        for (let i = 0; i < count; i++) {
            const x = Math.random() * this.width;
            const py = y + Math.random() * bandHeight;

            // Broken signal gaps.
            if (
                Math.sin(
                    x * 0.035 +
                    band.phase +
                    this.frame * 0.09
                ) > 0.78
            ) {
                continue;
            }

            const len = needleLength *
                (0.4 + Math.random() * 1.5);

            const slant = (Math.random() - 0.5) *
                (1 + d * 3);

            const alpha = band.opacity *
                (0.3 + 0.7 * Math.random());

            const color = Math.random();

            if (color < 0.08) {
                ctx.strokeStyle =
                    `rgba(95,130,255,${alpha})`;
            } else if (color < 0.22) {
                ctx.strokeStyle =
                    `rgba(0,220,105,${alpha})`;
            } else {
                ctx.strokeStyle =
                    `rgba(55,245,215,${alpha})`;
            }

            ctx.lineWidth =
                Math.random() < 0.12 ? 1.3 : 0.7;

            ctx.beginPath();
            ctx.moveTo(x + band.drift, py);
            ctx.lineTo(
                x + band.drift + len,
                py + slant
            );
            ctx.stroke();
        }

        // Occasional signal tears at high distortion.
        if (d > 0.65 && Math.random() < d * 0.22) {
            const tearY =
                y + Math.random() * bandHeight;

            ctx.fillStyle =
                `rgba(130,235,245,${d * 0.22})`;

            ctx.fillRect(
                Math.random() * this.width * 0.3,
                tearY,
                this.width *
                (0.2 + Math.random() * 0.5),
                1
            );
        }

        ctx.restore();
    }

    animate() {
        if (!this.running) return;

        this.frame++;

        this.ctx.clearRect(
            0, 0, this.width, this.height
        );

        if (this.distortion > 0) {
            this.spawnBand();

            for (let i = this.bands.length - 1; i >= 0; i--) {
                const band = this.bands[i];

                band.y += band.speed;
                band.phase += 0.025;

                if (band.y > this.height + band.height) {
                    this.bands.splice(i, 1);
                } else {
                    this.drawBand(band);
                }
            }
        }

        this.rafId = requestAnimationFrame(this.animate);
    }

    destroy() {
        this.running = false;

        cancelAnimationFrame(this.rafId);

        this.observer?.disconnect();

        if (!this.observer) {
            window.removeEventListener(
                "resize",
                this.resize
            );
        }

        this.canvas.remove();
    }
}
