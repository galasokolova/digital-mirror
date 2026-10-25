// Calculate actual fake strength (fact-checking reduces misinformation)
function calculateEffectiveFake(valFake, valFact) {
    return max(0, valFake - valFact);
}

// Calculate blur radius and opacity
function calculateBlurSettings(valLogic, valFact) {
    let blurAmount = 20 - map(valLogic, 0, 1023, 0, 8) - map(valFact, 0, 1023, 0, 12);
    blurAmount = max(0, blurAmount);

    let alphaVal = map(blurAmount, 0, 20, 0, 230);
    return { blurAmount, alphaVal };
}

// Eye center coordinates based on FaceMesh keypoints
function getEyeCenters(keypoints) {
    let leftX = (keypoints[159].x + keypoints[145].x) / 2;
    let leftY = (keypoints[159].y + keypoints[145].y) / 2;

    let rightX = (keypoints[386].x + keypoints[374].x) / 2;
    let rightY = (keypoints[386].y + keypoints[374].y) / 2;

    return { leftX, leftY, rightX, rightY };
}

// Eye area dimensions based on pupil distance
function getSpotDimensions(leftX, leftY, rightX, rightY) {
    let eyeDist = dist(leftX, leftY, rightX, rightY);
    return {
        spotWidth: eyeDist * 0.7,
        spotHeight: eyeDist * 0.5
    };
}