// Instance mode sketch for the newspaper / media archive panel
const sideCanvasSketch = (p) => {
    const PANEL_WIDTH = 640;
    const PANEL_HEIGHT = 480;

    p.setup = () => {
        let canvas = p.createCanvas(PANEL_WIDTH, PANEL_HEIGHT);
        canvas.parent('side-canvas-container');

        // Initialize modular background effects
        initPaperTexture(p, PANEL_WIDTH, PANEL_HEIGHT);
        initCrtTexture(p, PANEL_WIDTH, PANEL_HEIGHT);
    };

    p.draw = () => {
        let rawTime = (typeof sliderTime !== 'undefined' && sliderTime) ? sliderTime.value() : 0;
        renderNewsPanel(rawTime);
    };

    function renderNewsPanel(rawTime) {
        p.push();

        let year = Math.round(p.map(rawTime, 0, 1023, 1960, 2026));

        // Base background colors and background-level effects
        if (year < 1995) {
            p.background(242, 236, 218); // Craft paper
            applyPaperGrain(p);
        } else if (year < 2022) {
            p.background(230, 232, 235); // Web 2.0 gray
        } else {
            p.background(14, 18, 24);    // Cyber dark terminal
            drawCyberBackgroundGrid(p, PANEL_WIDTH, PANEL_HEIGHT);
        }

        // UI Content
        drawNewspaperTimeline(year, rawTime);
        drawArticleSection(year);
        drawAiAnalysisSection(year);

        // Foreground CRT screen artifacts
        if (year >= 2022) {
            applyCrtPostProcess(p, PANEL_WIDTH, PANEL_HEIGHT);
        }

        p.pop();
    }

    function drawNewspaperTimeline(year, rawTime) {
        p.push();
        p.stroke(year > 2021 ? 70 : 180);
        p.strokeWeight(1);
        p.line(40, 60, PANEL_WIDTH - 40, 60);

        let markerX = p.map(rawTime, 0, 1023, 40, PANEL_WIDTH - 40);
        p.fill(year > 2021 ? p.color(0, 220, 180) : p.color(180, 40, 30));
        p.noStroke();
        p.circle(markerX, 60, 10);

        p.textSize(10);
        p.textAlign(p.CENTER, p.TOP);
        p.fill(year > 2021 ? 160 : 100);
        p.text("1960\n(PRINT)", 50, 70);
        p.text("1995\n(WEB)", PANEL_WIDTH / 2, 70);
        p.text("2026\n(SYNTHETIC)", PANEL_WIDTH - 60, 70);

        p.textSize(24);
        p.textStyle(p.BOLD);
        p.textAlign(p.RIGHT, p.TOP);
        p.fill(year > 2021 ? p.color(0, 220, 180) : p.color(30));
        p.text(year, PANEL_WIDTH - 40, 20);
        p.pop();
    }

    function drawArticleSection(year) {
        p.push();
        let isDark = year > 2021;

        p.textAlign(p.LEFT, p.TOP);
        p.textFont(isDark ? "Courier New" : "Georgia");
        p.textSize(12);
        p.textStyle(isDark ? p.BOLD : p.ITALIC);
        p.fill(isDark ? p.color(0, 220, 180, 200) : 100);
        p.text(isDark ? "/// SYNTHETIC FEED ARCHIVE [DEEP_STREAM]" : "THE CHRONICLE DISPATCH — HISTORICAL ARCHIVE", 40, 115);

        p.stroke(isDark ? p.color(0, 220, 180, 60) : 180);
        p.line(40, 135, PANEL_WIDTH - 40, 135);

        p.noStroke();
        p.textSize(20);
        p.textStyle(p.BOLD);
        p.fill(isDark ? 240 : 20);
        p.textLeading(26);
        let headlineText = (typeof currentHeadline !== 'undefined' && currentHeadline)
            ? currentHeadline
            : "Awaiting archival dispatch...";
        p.text(headlineText, 40, 150, PANEL_WIDTH - 80, 160);
        p.pop();
    }

    function drawAiAnalysisSection(year) {
        if (typeof aiAnalysisResult === 'undefined' || !aiAnalysisResult) return;

        p.push();
        let isDark = year > 2021;
        let boxY = 320;
        let boxH = 140;

        p.fill(isDark ? p.color(20, 26, 36) : p.color(255, 255, 255, 180));
        p.stroke(isDark ? p.color(0, 220, 180, 100) : p.color(200));
        p.rect(40, boxY, PANEL_WIDTH - 80, boxH, 4);

        p.noStroke();
        p.fill(isDark ? p.color(0, 220, 180) : p.color(180, 50, 40));
        p.textSize(11);
        p.textStyle(p.BOLD);
        p.text("CRITICAL DECONSTRUCTION (LLM VERDICT):", 55, boxY + 15);

        p.fill(isDark ? 220 : 40);
        p.textFont(isDark ? "Courier New" : "Georgia");
        p.textSize(12);
        p.textLeading(18);
        p.textStyle(p.NORMAL);
        p.text(aiAnalysisResult, 55, boxY + 35, PANEL_WIDTH - 110, boxH - 45);
        p.pop();
    }
};

new p5(sideCanvasSketch);