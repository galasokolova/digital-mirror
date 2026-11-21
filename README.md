# The Anatomy of Distortion: A Digital Mirror

**An interactive installation exploring human trust, artificial intelligence, and the perception of information.**

## About the Project

*The Anatomy of Distortion: A Digital Mirror* is an interactive art installation that explores the relationship between human judgment, machine interpretation, and the credibility of digital information.

We encounter an enormous amount of information every day. News headlines, social media posts, and AI-generated content influence how we understand the world. Yet people do not evaluate information purely through facts. Our beliefs, experiences, emotions, and expectations shape what we choose to trust.

Artificial intelligence approaches information differently. A machine learning model identifies patterns learned from data and produces numerical predictions.

**What happens when a person's belief conflicts with a machine's assessment?**

Digital Mirror transforms this difference into a visual experience.

Instead of displaying a simple verdict about whether an article is true or false, the installation invites visitors to reflect on their own confidence in information and observe how their judgment interacts with an algorithmic prediction.

The project does not assume that artificial intelligence knows the truth. It explores the uncertainty and limitations of both human and machine interpretation.

## The Concept

The installation consists of two interconnected visual spaces.

**The Information Canvas**

The visitor reads an article selected from a small, curated dataset. Using a physical potentiometer, they express how much they trust the information on a continuous scale from 0 to 1.

Three buttons control the interaction:

- **BACK** — return to the previous article.
- **SELECT** — confirm the visitor's trust evaluation.
- **NEXT** — move to the next article.

**The Digital Mirror**

A camera captures the visitor's face, while a face-tracking system identifies facial landmarks.

Generative visual agents produce organic patterns, particles, and distortions around the face. The mirror becomes a visual representation of the relationship between personal belief and algorithmic assessment.

The intention is not to illustrate emotions directly, but to make the abstract difference between human and machine judgments visible.

## How It Works

The interaction is designed around five stages:

1. **Read** — The visitor encounters an article.
2. **Evaluate** — The visitor chooses a trust value between 0 and 1.
3. **Predict** — A locally trained machine learning model estimates a manipulation score for the article.
4. **Compare** — The system calculates the difference between the visitor's trust and the model's assessment.
5. **Visualize** — Generative agents transform this difference into a visual distortion on the visitor's digital reflection.

### Distortion Index

The installation uses the following formula:

**D = |T − (1 − M)|**

Where:

- **T** — Human Trust, from 0 to 1.
- **M** — Model-estimated Manipulation Score, from 0 to 1.
- **D** — Distortion Index, from 0 to 1.

A lower value indicates closer agreement between the visitor's trust and the model's estimated credibility. A higher value indicates greater disagreement.

The Distortion Index controls the intensity of the generative visual effect.

Importantly, this is a measure of disagreement between two assessments, **not a scientifically established measure of truth or misinformation**.

## Why This Project Matters

Digital Mirror addresses several questions:

- Why do people trust some information and reject other information?
- How does artificial intelligence evaluate patterns associated with potentially manipulative content?
- What happens when human intuition and algorithmic predictions disagree?
- Can generative art help people reflect on their own information judgments?

The installation is intended to encourage critical thinking rather than provide definitive answers.

Its central idea is that technology should not simply replace human judgment. It can also create opportunities to question, compare, and better understand that judgment.

## Academic Context

This project is being developed as an interdisciplinary academic project at **IADE**, during the **2026/2027 academic year, first semester**.

It connects four curricular units:

| Curricular Unit | Contribution to the Project |
|---|---|
| **Aprendizagem Automática Aplicada** (Applied Machine Learning) | Dataset preparation, model training, evaluation, and prediction |
| **Computação Física e Internet das Coisas** (Physical Computing and Internet of Things) | Arduino hardware, potentiometer input, physical buttons, and sensor communication |
| **Introdução à Inteligência Artificial** (Introduction to Artificial Intelligence) | AI concepts, algorithmic decision-making, interpretation, and limitations |
| **Programação Criativa** (Creative Programming) | p5.js, generative agents, interactive visualization, and FaceMesh integration |

The project brings together technical implementation, artistic experimentation, and critical reflection on artificial intelligence.

## Technology Stack

| Component | Technologies |
|---|---|
| Backend | Java 21, Spring Boot, Maven |
| Frontend | JavaScript, HTML, CSS |
| Creative Coding | p5.js |
| Face Tracking | ml5.js FaceMesh |
| Machine Learning | Local model — under development |
| Physical Interaction | Arduino Uno, potentiometer, three buttons (planned configuration) |
| Communication | REST API, USB serial |

## System Architecture

The system combines three main components.

**Frontend**

Displays the article reader and the digital mirror. It manages the interaction state, camera input, face tracking, and generative visual effects.

**Backend**

A Spring Boot application serves the frontend and provides an integration layer for hardware communication and future model predictions.

**Physical Controller**

An Arduino-based interface is planned to provide one analog trust input and three navigation buttons.

During development, the physical controls are simulated using HTML elements.

The machine learning component will operate locally, without requiring paid external AI API requests.

## Project Structure

```text
digital-mirror/
├── orchestrator/     # Java Spring Boot backend
├── frontend/         # p5.js, FaceMesh, article reader
├── arduino/          # Firmware and hardware documentation
├── .gitignore
└── README.md
```

## Current Development Status

### Implemented

- Spring Boot application and frontend serving
- Two interconnected p5.js canvases
- Camera capture and FaceMesh tracking
- Generative NeedleAgent visual effects
- Organic particles and accumulated drawing layers
- Local demonstration articles
- Trust slider ranging from 0 to 1
- BACK, SELECT, and NEXT interface controls
- Arduino serial communication prototype
- Mock mode without OpenAI API requests

### In Development

- Curated and labeled article dataset
- Local machine learning model
- Model evaluation and validation
- Integration of ML predictions into the application
- Distortion Index-driven visualization
- Updated Arduino hardware with one potentiometer and three buttons
- Final installation and exhibition experience

The current visual prototype uses Trust as a temporary effect-control value. It does not yet use predictions from a trained manipulation-detection model.

## Project Documentation

| Area | Documentation | Description |
|---|---|---|
| Project | [Main README](README.md) | Project concept, architecture, and development status |
| Machine Learning | [ML README](ml/README.md) | ML pipeline, datasets, scripts, and model training |
| ML Research | [Initial Research Report](ml/docs/ML_RESEARCH_REPORT.md) | Dataset exploration, LIAR experiments, and initial limitations |
| FEVER Research | [FEVER Experiments Report](ml/docs/FEVER_EXPERIMENTS_REPORT.md) | FEVER preparation, experiments E01–E06d, difficulties, and findings |
| Arduino | [Arduino README](arduino/README.md) | Hardware, controls, firmware, and serial communication |

## Running the Prototype

### Requirements

- Java 21
- A modern browser with camera access
- Internet access for the currently CDN-hosted frontend libraries
- Arduino hardware is not required when using simulated controls

### Start the Application

From the repository root:

```powershell
cd orchestrator
.\mvnw.cmd spring-boot:run
```

Open:

`http://localhost:8080/`

Allow browser access to the camera.

The frontend is served by Spring Boot, so a separate frontend development server is not required.

For the existing Arduino prototype, see [Arduino Hardware Documentation](arduino/README.md).

## Future Development

The next development stage focuses on building a small, reproducible machine learning pipeline using labeled news articles.

The model will estimate a manipulation-related score, subject to the limitations of its dataset and evaluation.

Its predictions will be compared with visitor trust evaluations to calculate the Distortion Index and drive the generative artwork.

The longer-term goal is to create a complete physical installation in which interaction, machine learning, and visual expression form a single experience.

---

*Digital Mirror does not tell visitors what to believe. It invites them to examine how belief is formed — and what happens when a machine sees information differently.*