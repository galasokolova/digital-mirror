class MachineSpeech {

    constructor(onSpeakingChange) {
        this.onSpeakingChange = onSpeakingChange;
        this.speechId = 0;

        this.finalMessage =
            "But I'm just a machine. Let's look at the evidence.";
    }

    // ==========================================
    // MACHINE REACTIONS
    // ==========================================

    getReaction(support) {

        if (support >= 90) {
            return {
                text: "Absolutely. I have very little doubt about this.",
                rate: 0.95,
                pitch: 0.85
            };
        }

        if (support >= 70) {
            return {
                text: "Yes. That sounds quite convincing to me.",
                rate: 0.90,
                pitch: 0.80
            };
        }

        if (support >= 55) {
            return {
                text: "I'm not completely sure, but it sounds plausible.",
                rate: 0.82,
                pitch: 0.75
            };
        }

        if (support >= 45) {
            return {
                text: "Hmm... Interesting. I really can't make up my mind.",
                rate: 0.72,
                pitch: 0.70
            };
        }

        if (support >= 30) {
            return {
                text: "Wait a minute. Something doesn't quite add up here.",
                rate: 0.80,
                pitch: 0.65
            };
        }

        if (support >= 10) {
            return {
                text: "Seriously? I find that very hard to believe.",
                rate: 0.90,
                pitch: 0.85
            };
        }

        return {
            text: "Ha-ha! That's ridiculous. How could that possibly be true?",
            rate: 1.05,
            pitch: 1.10
        };
    }

    createUtterance(text, rate, pitch) {

        const utterance =
            new SpeechSynthesisUtterance(text);

        utterance.lang = "en-US";
        utterance.rate = rate;
        utterance.pitch = pitch;
        utterance.volume = 1;

        return utterance;
    }

    speak(support) {

        if (!("speechSynthesis" in window)) {
            console.warn("Speech synthesis is not supported");
            return;
        }

        if (!Number.isFinite(support)) return;

        this.stop();

        const currentId = this.speechId;
        const reaction = this.getReaction(support);

        const utterance = this.createUtterance(
            reaction.text,
            reaction.rate,
            reaction.pitch
        );

        utterance.onstart = () => {
            if (currentId !== this.speechId) return;

            this.onSpeakingChange(true);
        };

        utterance.onend = () => {
            if (currentId !== this.speechId) return;

            this.onSpeakingChange(false);

            // Pause before the final message
            setTimeout(() => {

                if (currentId !== this.speechId) return;

                this.speakFinalMessage(currentId);

            }, 1000);
        };

        utterance.onerror = () => {
            if (currentId !== this.speechId) return;

            this.onSpeakingChange(false);
        };

        window.speechSynthesis.speak(utterance);
    }

    speakAsync(reaction, currentId) {

        return new Promise(resolve => {

            if (currentId !== this.speechId) {
                resolve();
                return;
            }

            const utterance = this.createUtterance(
                reaction.text,
                reaction.rate,
                reaction.pitch
            );

            utterance.onstart = () => {
                if (currentId === this.speechId) {
                    this.onSpeakingChange(true);
                }
            };

            const finish = () => {

                if (currentId === this.speechId) {
                    this.onSpeakingChange(false);
                }

                resolve();
            };

            utterance.onend = finish;
            utterance.onerror = finish;

            window.speechSynthesis.speak(utterance);
        });
    }

    speakFinalMessage(currentId) {

        const utterance = this.createUtterance(
            this.finalMessage,
            0.75,
            0.55
        );

        utterance.onstart = () => {
            if (currentId !== this.speechId) return;

            this.onSpeakingChange(true);
        };

        utterance.onend = () => {
            if (currentId !== this.speechId) return;

            this.onSpeakingChange(false);
        };

        utterance.onerror = () => {
            if (currentId !== this.speechId) return;

            this.onSpeakingChange(false);
        };

        window.speechSynthesis.speak(utterance);
    }

    getComparisonReaction(human, machine) {

        const difference = Math.abs(human - machine);

        // Both are unsure
        if (
            Math.abs(human - 0.5) <= 0.1 &&
            Math.abs(machine - 0.5) <= 0.1
        ) {
            return {
                text: "Hmm... Neither of us seems particularly convinced.",
                rate: 0.75,
                pitch: 0.75
            };
        }

        // Almost identical opinions
        if (difference <= 0.10) {
            return {
                text: "Ha-ha! Look at us! We almost think alike!",
                rate: 1.10,
                pitch: 1.35
            };
        }

        // Similar opinions
        if (difference <= 0.25) {
            return {
                text: "Oh! Interesting! We seem to see things in a similar way.",
                rate: 0.95,
                pitch: 1.15
            };
        }

        // Completely different opinions
        if (difference >= 0.60) {
            return {
                text: "Whoa! Seriously? We couldn't disagree more!",
                rate: 1.05,
                pitch: 1.30
            };
        }

        return {
            text: "Hmm... That's fascinating. You see something quite different from what I see.",
            rate: 0.85,
            pitch: 0.90
        };
    }

    getTruthReaction(human, machine, verifiedLabel) {

        const getDecision = (score) => {
            if (score >= 0.55) return "SUPPORTS";
            if (score <= 0.45) return "REFUTES";
            return "UNSURE";
        };

        if (
            verifiedLabel !== "SUPPORTS" &&
            verifiedLabel !== "REFUTES"
        ) {
            return {
                text: "It seems we still need more evidence.",
                rate: 0.80,
                pitch: 0.80
            };
        }

        const humanDecision = getDecision(human);
        const machineDecision = getDecision(machine);

        const humanCorrect = humanDecision === verifiedLabel;
        const machineCorrect = machineDecision === verifiedLabel;

        // Uncertainty
        if (
            humanDecision === "UNSURE" ||
            machineDecision === "UNSURE"
        ) {
            return {
                text: "Sometimes admitting uncertainty is better than being confidently wrong.",
                rate: 0.80,
                pitch: 0.85
            };
        }

        // Both correct
        if (humanCorrect && machineCorrect) {
            return {
                text: "Yes! We got it right! High five! Oh... right. I don't have hands.",
                rate: 1.10,
                pitch: 1.40
            };
        }

        // Machine correct, human wrong
        if (machineCorrect && !humanCorrect) {
            return {
                text: "Hey... Don't feel bad. At least now you know the truth. That's what matters.",
                rate: 0.80,
                pitch: 0.95
            };
        }

        // Human correct, machine wrong
        if (humanCorrect && !machineCorrect) {
            return {
                text: "Wait! You were right? And I was wrong? Wow... That's embarrassing. Well done, human!",
                rate: 0.95,
                pitch: 1.30
            };
        }

        // Both wrong
        return {
            text: "Oh no! We both fell for it! Ha-ha! So much for collective intelligence.",
            rate: 1.05,
            pitch: 1.20
        };
    }

    async speakDialogue(human, machine, verifiedLabel) {

        if (!("speechSynthesis" in window)) return;

        if (
            !Number.isFinite(human) ||
            !Number.isFinite(machine)
        ) return;

        this.stop();

        const currentId = this.speechId;

        const reactions = [

            // Machine's own opinion
            this.getReaction(machine * 100),

            // Comparison with the human
            this.getComparisonReaction(human, machine),

            // Reaction to verified truth
            this.getTruthReaction(
                human,
                machine,
                verifiedLabel
            ),

            // Final mechanical message
            {
                text: this.finalMessage,
                rate: 0.75,
                pitch: 0.55
            }
        ];

        for (const reaction of reactions) {

            if (currentId !== this.speechId) return;

            await this.speakAsync(reaction, currentId);

            if (currentId !== this.speechId) return;

            await new Promise(resolve => {
                setTimeout(resolve, 700);
            });
        }
    }

    // ==========================================
    // STOP SPEECH
    // ==========================================

    stop() {
        this.speechId++;

        if ("speechSynthesis" in window) {
            window.speechSynthesis.cancel();
        }

        this.onSpeakingChange(false);
    }
}