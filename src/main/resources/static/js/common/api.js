const API_BASE_URL = "http://localhost:8080/api";

let currentHeadline = "Waiting for headline...";
let aiAnalysisResult = "";
let isAiLoading = false;

// Global data of Arduino
let hardwareState = {
  timeline: 512,
  logic: 512,
  factCheck: 512,
  fakeGenerator: 0
};

// request Spring Boot
let isPolling = false;

async function pollHardwareState() {
  // Если предыдущий запрос еще выполняется — не создаем очередь, пропускаем тик
  if (isPolling) return;
  isPolling = true;

  try {
    let res = await fetch(`${API_BASE_URL}/hardware/state`);
    if (res.ok) {
      let data = await res.json();
      hardwareState = data;

      // Обновляем бегунки в UI
      if (typeof sliderTime !== 'undefined' && sliderTime) sliderTime.value(data.timeline);
      if (typeof sliderLogic !== 'undefined' && sliderLogic) sliderLogic.value(data.logic);
      if (typeof sliderFact !== 'undefined' && sliderFact) sliderFact.value(data.factCheck);
      if (typeof sliderFake !== 'undefined' && sliderFake) sliderFake.value(data.fakeGenerator);
    }
  } catch (err) {
  } finally {
    isPolling = false;
  }
}

// Receive a random headline from Spring Boot
async function fetchRandomHeadline() {
  try {
    let res = await fetch(`${API_BASE_URL}/headline/random`);
    if (!res.ok) throw new Error(`HTTP error! Status: ${res.status}`);

    let data = await res.json();
    currentHeadline = data.text;
    return data;
  } catch (err) {
    currentHeadline = "Offline: Media stream unavailable.";
    return null;
  }
}

// Sending current actuators' values for analysis
async function sendAnalysisRequest(tYear, logicVal, factVal, fakeVal) {
  if (isAiLoading) return;
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
    return result;
  } catch (err) {
    aiAnalysisResult = "Error connecting to AI analysis service.";
    return null;
  } finally {
    isAiLoading = false;
  }
}