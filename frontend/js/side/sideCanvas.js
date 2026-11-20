
const sideCanvasSketch = (p) => {

    const W = 640;
    const H = 480;

    p.setup = () => {
        const canvas = p.createCanvas(W, H);
        canvas.parent("side-canvas-container");

        p.textFont("Courier New");
    };

    p.draw = () => {
        const article = getCurrentArticle();

        p.background(14, 18, 24);

        drawHeader();
        drawArticle(article);
        drawFooter();
    };

    function drawHeader() {
        p.push();

        p.fill(0, 220, 180);
        p.noStroke();
        p.textSize(13);
        p.textStyle(p.BOLD);
        p.text("DIGITAL MIRROR / ARTICLE READER", 30, 35);

        p.stroke(0, 220, 180, 100);
        p.line(30, 50, W - 30, 50);

        p.pop();
    }

    function drawArticle(article) {
        if (!article) return;

        p.push();

        p.noStroke();
        p.textAlign(p.LEFT, p.TOP);

        p.fill(120, 160, 155);
        p.textSize(11);
        p.textStyle(p.NORMAL);

        p.text(
            `ARTICLE ${mirrorState.articleIndex + 1} / ${articles.length}`,
            30,
            75
        );

        p.fill(255);
        p.textSize(22);
        p.textStyle(p.BOLD);
        p.textLeading(28);

        p.text(article.title, 30, 110, W - 60, 100);

        p.fill(0, 220, 180);
        p.textSize(11);
        p.textStyle(p.NORMAL);
        p.text(article.source, 30, 205);

        p.fill(205, 215, 215);
        p.textSize(15);
        p.textLeading(23);

        p.text(article.text, 30, 245, W - 60, 145);

        p.pop();
    }

    function drawFooter() {
        p.push();

        p.stroke(0, 220, 180, 70);
        p.line(30, 405, W - 30, 405);

        p.noStroke();
        p.fill(0, 220, 180);
        p.textSize(12);

        if (mirrorState.phase === "RESULT") {
            p.text(
                `YOUR TRUST: ${mirrorState.trust.toFixed(2)}`,
                30,
                435
            );

            p.fill(150);
            p.textSize(10);
            p.text("ML ANALYSIS: NOT CONNECTED", 30, 455);

        } else {
            p.text(
                "READ THE ARTICLE AND SELECT YOUR TRUST LEVEL",
                30,
                440
            );
        }

        p.pop();
    }
};

new p5(sideCanvasSketch);
