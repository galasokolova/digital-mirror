const API_BASE_URL = "http://localhost:8080/api";

let currentHeadline = "Waiting for headline...";
let aiAnalysisResult = "";
let isAiLoading = false;

// Receive a random headline from Spring Boot
async function fetchRandomHeadline() {
  try {
    let res = await fetch(`${API_BASE_URL}/headline/random`);
    if (!res.ok) throw new Error(`HTTP error! Status: ${res.status}`);

    let data = await res.json();
    currentHeadline = data.text;
    console.log("Headline downloaded:", currentHeadline);
    return data;
  } catch (err) {
    console.error("Error attempting to get a headline:", err);
    currentHeadline = "Offline: Media stream unavailable.";
    return null;
  }
}

// Sending current actuators' values for analysis
async function sendAnalysisRequest(tYear, logicVal, factVal, fakeVal) {
  if (isAiLoading) return; // "429 Too Many Requests" protection
  isAiLoading = true;

  let payload = {
    headline: currentHeadline,
    timelineYear: tYear,
    logicValue: logicVal,
    factCheckValue: factVal,
    fakeGeneratorValue: fakeVal
  };

  try {
    let res = await fetch(`${API_BASE_URL}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error(`HTTP error! Status: ${res.status}`);

    let result = await res.json();
    aiAnalysisResult = result.analysis;
    console.log("AI analysis received:", aiAnalysisResult);
    return result;
  } catch (err) {
    console.error("Error sending data to Spring Boot:", err);
    aiAnalysisResult = "Error connecting to AI analysis service.";
    return null;
  } finally {
    isAiLoading = false;
  }
}