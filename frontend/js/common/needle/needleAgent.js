// Shared particle behavior; rendering is delegated to the owning canvas.
class NeedleAgent {
  constructor(x, y, options = {}) {
    this.x = x; this.y = y;
    this.stepSize = options.stepSize ?? 4;
    this.diameter = options.diameter ?? 5;
    this.initialLife = options.life ?? 200;
    this.life = this.initialLife;
    this.r = options.r ?? 0;
    this.g = options.g ?? 0;
    this.b = options.b ?? 0;
  }
  update(intensity, width, height, pixels = null) {
    const level = Math.max(0, Math.min(1, intensity));
    const directions = [[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]];
    const steps = Math.floor(3 + 17 * level);
    for (let i = 0; i < steps; i++) {
      const [dx, dy] = directions[Math.floor(Math.random() * directions.length)];
      this.x = Math.max(0, Math.min(width - 1, this.x + dx * this.stepSize));
      this.y = Math.max(0, Math.min(height - 1, this.y + dy * this.stepSize));
      if (pixels && pixels.length >= width * height * 4) {
        const offset = (Math.floor(this.x) + Math.floor(this.y) * width) * 4;
        const distance = Math.hypot(pixels[offset]-this.r, pixels[offset+1]-this.g, pixels[offset+2]-this.b);
        if (distance < 30) {
          this.r = (this.r + 0.2) % 256;
          this.g = (this.g + 0.5) % 256;
          this.b = (this.b + 1.2) % 256;
        }
      }
    }
    this.life -= 3 - 2.2 * level;
  }
  renderP5(pg, intensity) {
    const level = Math.max(0, Math.min(1, intensity));
    const small = (this.diameter + 5) * (0.8 + 1.2 * level);
    const big = (this.diameter + 15) * (0.8 + 1.7 * level);
    const rand = (size) => (Math.random() * 2 - 1) * size;
    pg.strokeWeight(1);
    if (Math.floor(this.x) % 3 === 0 || Math.floor(this.y) % 3 === 0) {
      pg.stroke(0, 220, 80, this.life * 0.7);
      pg.line(this.x, this.y, this.x + rand(big), this.y + rand(big));
    }
    if (Math.floor(this.x) % 20 === 0 || Math.floor(this.y) % 20 === 0) {
      pg.stroke(20, 80, 255, this.life * 0.5);
      pg.line(this.x, this.y, this.x + rand(big * 1.5), this.y + rand(big * 1.5));
    }
    pg.stroke(this.r, this.g, this.b, this.life * 0.9);
    pg.line(this.x, this.y, this.x + rand(small), this.y + rand(small));
  }
  renderCanvas(ctx, intensity, width, height) {
    const edge = Math.min(this.x, this.y, width - this.x, height - this.y);
    const fade = Math.pow(Math.max(0, 1 - edge / 100), 2);
    const alpha = Math.max(0, this.life / this.initialLife) * fade;
    if (alpha <= 0.01) return;
    const len = 4 + 12 * intensity;
    const line = (r,g,b,a,scale=1) => {
      ctx.strokeStyle = `rgba(${r},${g},${b},${Math.max(0, Math.min(1,a))})`;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(this.x,this.y);
      ctx.lineTo(this.x + (Math.random()-.5)*len*2*scale, this.y + (Math.random()-.5)*len*2*scale);
      ctx.stroke();
    };
    line(this.r,this.g,this.b,alpha);
    if (Math.random()<0.25) line(0,220,80,alpha*.7,1.5);
    if (Math.random()<0.06) line(20,80,255,alpha*.7,2);
  }
  isDead() { return this.life <= 0; }
}
