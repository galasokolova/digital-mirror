// Each mirror owns its own NeedleSystem instance and intensity.
class NeedleSystem {
  constructor({maxAgents = 30, spawnRate = 0.3, agentOptions = {}} = {}) {
    this.agents = [];
    this.maxAgents = maxAgents;
    this.spawnRate = spawnRate;
    this.agentOptions = agentOptions;
    this.intensity = 0;
  }
  setDistortion(value) {
    const n = Number(value);
    if (Number.isFinite(n)) this.intensity = Math.max(0, Math.min(1, n));
  }
  reset() { this.agents.length = 0; this.intensity = 0; }
  spawnAt(x,y) {
    if (this.agents.length >= this.maxAgents) return;
    this.agents.push(new NeedleAgent(x,y,this.agentOptions));
  }
  spawnEyes(faces) {
    if (!faces?.length || this.intensity <= .01 || Math.random() >= this.spawnRate * this.intensity) return;
    const k = faces[0]?.keypoints;
    if (!k) return;
    const left = k[468] || k[159], right = k[473] || k[386];
    if (left) this.spawnAt(left.x,left.y);
    if (right) this.spawnAt(right.x,right.y);
  }
  spawnBorder(width,height) {
    if (this.intensity <= .01 || Math.random() >= this.spawnRate * this.intensity) return;
    const side = Math.floor(Math.random()*4);
    const x = side===1 ? width-2 : side===3 ? 2 : Math.random()*width;
    const y = side===0 ? 2 : side===2 ? height-2 : Math.random()*height;
    this.spawnAt(x,y);
  }
  update(width,height,pixels=null) {
    for (let i=this.agents.length-1;i>=0;i--) {
      const a=this.agents[i]; a.update(this.intensity,width,height,pixels);
      if(a.isDead()) this.agents.splice(i,1);
    }
  }
  renderP5(layer) { for (const a of this.agents) a.renderP5(layer,this.intensity); }
  renderCanvas(ctx,width,height) { for (const a of this.agents) a.renderCanvas(ctx,this.intensity,width,height); }
}
