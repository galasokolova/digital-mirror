class MachineFrame {
    constructor(container) {
        this.container = container;
        this.needles = new NeedleSystem({
            maxAgents: 6,
            spawnRate: 0.03,
            agentOptions: {life: 100, r: 0, g: 255, b: 200, diameter: 8, stepSize: 2}
        });
        this.frame = 0;

        this.canvas = document.createElement("canvas");
        this.canvas.className = "machine-frame-overlay";

        Object.assign(this.container.style, {
            position: "relative",
            overflow: "hidden"
        });

        Object.assign(this.canvas.style, {
            position: "absolute",
            inset: "0",
            width: "100%",
            height: "100%",
            pointerEvents: "none",
            zIndex: "10"
        });

        this.container.appendChild(this.canvas);
        this.ctx = this.canvas.getContext("2d");

        this.resize();
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);

        this.observer = new ResizeObserver(() => this.resize());
        this.observer.observe(this.container);
    }

    resize() {
        const bounds = this.container.getBoundingClientRect();

        this.width = Math.max(1, Math.round(bounds.width));
        this.height = Math.max(1, Math.round(bounds.height));

        const dpr = Math.min(window.devicePixelRatio || 1, 2);

        this.canvas.width = Math.round(this.width * dpr);
        this.canvas.height = Math.round(this.height * dpr);

        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    setDistortion(level) {
        this.needles.setDistortion(level);
    }

    reset() {
        this.needles.reset();
    }

    drawTechFrame() {
        const ctx = this.ctx;
        const lineCount = 22;
        const spacing = 3.5;

        ctx.lineWidth = 1;

        for (let i = 0; i < lineCount; i++) {
            const offset = i * spacing;
            const alpha = (180 - i * (170 / (lineCount - 1))) / 255;

            const w = this.width - offset * 2;
            const h = this.height - offset * 2;

            if (w <= 0 || h <= 0) break;

            ctx.strokeStyle = `rgba(40,220,160,${alpha})`;
            ctx.strokeRect(offset, offset, w, h);
        }
    }

    drawCorners() {
        const ctx = this.ctx;
        const size = 14;
        const pulse = 0.7 + 0.3 * Math.sin(this.frame * 0.045);

        ctx.fillStyle = `rgba(40,220,160,${pulse})`;

        ctx.fillRect(2, 2, size, size);
        ctx.fillRect(this.width - size - 2, 2, size, size);
        ctx.fillRect(2, this.height - size - 2, size, size);
        ctx.fillRect(
            this.width - size - 2,
            this.height - size - 2,
            size,
            size
        );
    }

    animate() {
        this.frame++;

        const ctx = this.ctx;

        ctx.clearRect(0, 0, this.width, this.height);

        this.needles.spawnBorder(this.width, this.height);
        this.needles.update(this.width, this.height);
        this.needles.renderCanvas(ctx, this.width, this.height);
        this.drawTechFrame();
        this.drawCorners();

        requestAnimationFrame(this.animate);
    }
}
