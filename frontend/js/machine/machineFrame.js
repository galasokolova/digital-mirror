class MachineFrame {
    constructor(container) {
        this.container = container;
        this.agents = [];
        this.maxAgents = 35;
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

    spawnAgent() {
        if (this.agents.length >= this.maxAgents) return;
        if (Math.random() > 0.25) return;

        const edge = Math.floor(Math.random() * 4);
        let x, y;

        switch (edge) {
            case 0:
                x = Math.random() * this.width;
                y = 0;
                break;
            case 1:
                x = this.width;
                y = Math.random() * this.height;
                break;
            case 2:
                x = Math.random() * this.width;
                y = this.height;
                break;
            default:
                x = 0;
                y = Math.random() * this.height;
        }

        this.agents.push({
            x,
            y,
            life: 200,
            size: 5,
            r: 0,
            g: 255,
            b: 200
        });
    }

    drawAgents() {
        const ctx = this.ctx;

        for (let i = this.agents.length - 1; i >= 0; i--) {
            const a = this.agents[i];

            const directions = [
                [-1, -1], [0, -1], [1, -1],
                [-1,  0],          [1,  0],
                [-1,  1], [0,  1], [1,  1]
            ];

            const direction =
                directions[Math.floor(Math.random() * directions.length)];

            a.x = Math.max(0, Math.min(this.width, a.x + direction[0] * 4));
            a.y = Math.max(0, Math.min(this.height, a.y + direction[1] * 4));

            const edgeDistance = Math.min(
                a.x,
                a.y,
                this.width - a.x,
                this.height - a.y
            );

            const fade = Math.pow(
                Math.max(0, 1 - edgeDistance / 150),
                2
            );

            if (fade > 0.01) {
                const alpha = Math.max(0, a.life / 200) * fade;
                const length = 5 + Math.random() * 15;

                ctx.lineWidth = 1;

                // Main turquoise needle
                ctx.strokeStyle = `rgba(${a.r},${a.g},${a.b},${alpha})`;
                ctx.beginPath();
                ctx.moveTo(a.x, a.y);
                ctx.lineTo(
                    a.x + (Math.random() - 0.5) * length * 2,
                    a.y + (Math.random() - 0.5) * length * 2
                );
                ctx.stroke();

                // Occasional emerald spark
                if (Math.random() < 0.25) {
                    ctx.strokeStyle = `rgba(0,220,80,${alpha * 0.7})`;
                    ctx.beginPath();
                    ctx.moveTo(a.x, a.y);
                    ctx.lineTo(
                        a.x + (Math.random() - 0.5) * 30,
                        a.y + (Math.random() - 0.5) * 30
                    );
                    ctx.stroke();
                }

                // Rare blue spike
                if (Math.random() < 0.06) {
                    ctx.strokeStyle = `rgba(20,80,255,${alpha * 0.7})`;
                    ctx.beginPath();
                    ctx.moveTo(a.x, a.y);
                    ctx.lineTo(
                        a.x + (Math.random() - 0.5) * 45,
                        a.y + (Math.random() - 0.5) * 45
                    );
                    ctx.stroke();
                }
            }

            a.life -= 0.7;

            if (a.life <= 0) {
                this.agents.splice(i, 1);
            }
        }
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

        this.spawnAgent();
        this.drawAgents();
        this.drawTechFrame();
        this.drawCorners();

        requestAnimationFrame(this.animate);
    }
}
