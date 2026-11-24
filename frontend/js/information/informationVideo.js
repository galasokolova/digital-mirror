class InformationVideo {

    constructor() {
        this.video = document.getElementById(
            "information-video"
        );

        this.statement = document.getElementById(
            "spoken-statement"
        );

        this.words = [];
        this.lastWordIndex = -1;

        this.animate = this.animate.bind(this);

        requestAnimationFrame(this.animate);
    }

    async loadClaim(claim) {

        this.video.pause();

        this.words = [];
        this.lastWordIndex = -1;
        this.statement.textContent = "";

        const data = VIDEO_REGISTRY[claim.id];

        // Fallback for claims without video
        if (!data) {
            this.video.removeAttribute("src");
            this.video.load();

            this.statement.textContent = claim.text;
            return;
        }

        // Check that the transcript matches the claim
        const transcript = data.words
            .map(word => word.text)
            .join(" ");

        if (transcript !== claim.text.trim()) {
            this.video.removeAttribute("src");
            this.video.load();

            this.statement.textContent = claim.text;

            console.warn(
                "Video transcript does not match claim:",
                claim.id
            );

            return;
        }

        this.words = data.words;

        this.video.src = data.src;
        this.video.load();

        try {
            await this.video.play();
        } catch (error) {
            console.warn(
                "Press PLAY to start the video.",
                error
            );
        }
    }

    animate() {

        this.updateStatement();

        requestAnimationFrame(this.animate);
    }

    updateStatement() {

        if (this.words.length === 0) return;

        const currentTime = this.video.currentTime;

        let currentIndex = -1;

        for (let i = 0; i < this.words.length; i++) {

            if (currentTime >= this.words[i].time) {
                currentIndex = i;
            } else {
                break;
            }
        }

        if (currentIndex === this.lastWordIndex) {
            return;
        }

        this.lastWordIndex = currentIndex;

        this.statement.replaceChildren();

        for (let i = 0; i <= currentIndex; i++) {

            const span = document.createElement("span");

            span.textContent = this.words[i].text + " ";

            if (i === currentIndex) {
                span.classList.add("current-word");
            }

            this.statement.appendChild(span);
        }
    }
}