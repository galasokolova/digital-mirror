class InteractionFlow {

    constructor() {
        this.phase = "IDLE";

        this.claim = null;
        this.analysis = null;
        this.confirmation = null;
        this.revealData = null;

        this.humanScore = 0.5;
    }

    async next() {
        if (this.phase === "ANALYZING" ||
            this.phase === "ANIMATING") {
            throw new Error("Cannot start next claim now");
        }

        this.phase = "ANALYZING";

        try {
            this.claim = await ClaimAPI.next();
            this.analysis = await ClaimAPI.analyze(this.claim.id);

            this.confirmation = null;
            this.revealData = null;
            this.humanScore = 0.5;

            this.phase = "CHOOSING";

            return {
                claim: this.claim,
                analysis: this.analysis
            };
        } catch (error) {
            this.phase = "IDLE";
            throw error;
        }
    }

    setHumanScore(value) {
        if (this.phase !== "CHOOSING") {
            return;
        }

        if (!Number.isFinite(value)) {
            throw new Error("Invalid human score");
        }

        this.humanScore = Math.max(0, Math.min(1, value));
    }

    async confirm() {
        if (this.phase !== "CHOOSING") {
            throw new Error("Cannot confirm in phase " + this.phase);
        }

        this.phase = "CONFIRMING";

        try {
            this.confirmation = await ClaimAPI.confirm(
                this.claim.id,
                this.humanScore
            );

            this.phase = "ANIMATING";

            return this.confirmation;
        } catch (error) {
            this.phase = "CHOOSING";
            throw error;
        }
    }

    async reveal() {
        if (this.phase !== "ANIMATING") {
            throw new Error("Cannot reveal in phase " + this.phase);
        }

        this.revealData = await ClaimAPI.reveal(this.claim.id);
        this.phase = "REVEALED";

        return this.revealData;
    }
}

const interactionFlow = new InteractionFlow();
