
const mirrorState = {
    articleIndex: 0,
    trust: 0.5,
    manipulationScore: null,
    distortionIndex: 0,
    phase: "IDLE"
};

function updateTrust(value) {
    mirrorState.trust = Math.max(0, Math.min(1, value));

    if (mirrorState.manipulationScore !== null) {
        mirrorState.distortionIndex = Math.abs(
            mirrorState.trust -
            (1 - mirrorState.manipulationScore)
        );
    }
}

function resetMirrorEvaluation() {
    mirrorState.trust = 0.5;
    mirrorState.manipulationScore = null;
    mirrorState.distortionIndex = 0;
    mirrorState.phase = "IDLE";
}


function getCurrentArticle() {
    return articles[mirrorState.articleIndex];
}

function changeArticle(direction) {
    const nextIndex = mirrorState.articleIndex + direction;

    if (nextIndex < 0 || nextIndex >= articles.length) {
        return;
    }

    mirrorState.articleIndex = nextIndex;
    resetMirrorEvaluation();
    mirrorState.phase = "READING";
}

function confirmTrust() {
    mirrorState.phase = "RESULT";

    console.log("Article:", getCurrentArticle().id);
    console.log("Confirmed trust:", mirrorState.trust);
}
