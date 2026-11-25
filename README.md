# The Anatomy of Distortion: A Digital Mirror

**An interactive art installation about disinformation surrounding Russia’s war against Ukraine.**

Digital Mirror invites visitors to evaluate claims from video clips and then examine documented evidence. A visitor and a machine each score the same claim. Their images distort according to how far their individual scores are from the verified reference. The installation asks how confidently we judge what we hear and what happens when we encounter evidence that challenges that judgment.

## Installation Layout

| Position | Panel | What the visitor sees |
|---|---|---|
| Left | Human Mirror | A live camera image of the visitor, with face tracking and visual distortion after confirmation. |
| Centre | Information Canvas | A video clip containing the claim. After confirmation, this panel shows the documented evidence, an explanation of what happened, and links to sources. |
| Right | Machine Mirror | An animated mask that represents the machine’s assessment and reacts through distortion and speech. |

A framed area beneath the three panels displays the text of the statement and asks, **“How credible is this statement? Do you agree with it?”** The agreement control runs from 0 to 1. The prototype uses an on-screen slider. The physical installation is designed for an Arduino potentiometer.

## How a Round Works

1. Press **START/NEXT** to start the installation or load the next claim. The central panel plays the clip, and the statement text appears in the frame below the panels.
2. Set an agreement score between 0 and 1. A score of 0 means complete disagreement, 0.5 means uncertainty, and 1 means complete agreement.
3. The local evidence classifier evaluates the claim against a curated evidence passage. Its `p_supports` output is the machine’s agreement score.
4. Press **CONFIRM** to submit the score. Each mirror distorts according to its own assessment. At the same stage, the central panel reveals the evidence and source links so the visitor can see what the documentation establishes.
5. The machine comments on how closely the two scores align and ends with **“But I’m just a machine. Let’s look at the evidence.”** Press **START/NEXT** to begin another round.

### Distortion Index

For each participant, **D = |S − V|**, where **S** is their own agreement score and **V** is the verified reference value. `REFUTES` maps to 0 and `SUPPORTS` maps to 1.

| Reference value | Agreement score | Distortion |
|---|---:|---:|
| 0 | 0.85 | 0.85 |
| 1 | 0.85 | 0.15 |

The human and machine scores enter this calculation independently. The comparison between their scores is used to choose the machine’s spoken reaction.

## Academic Context

The project is being developed at **IADE** during the first semester of the **2026/2027 academic year**.

| Curricular unit | Contribution |
|---|---|
| Aprendizagem Automática Aplicada | Dataset preparation, model training, and evaluation |
| Computação Física e Internet das Coisas | Arduino, potentiometer, buttons, and USB serial communication |
| Introdução à Inteligência Artificial | Interpretation of model outputs and their limitations |
| Programação Criativa | p5.js, FaceMesh, animated masks, and interactive visual effects |

## Technology and Project Structure

| Directory | Role |
|---|---|
| [`frontend/`](frontend/) | JavaScript, HTML, CSS, p5.js, ml5.js FaceMesh, video playback, and speech synthesis |
| [`orchestrator/`](orchestrator/) | Java 21 and Spring Boot. Serves the frontend, handles claim sessions, calls the ML API, and calculates distortion. |
| [`ml/`](ml/README.md) | Python, scikit-learn, and FastAPI for local evidence classification, plus the small-talk training plan |
| [`arduino/`](arduino/README.md) | Arduino Uno firmware and hardware documentation |

## Prototype Status

The three-panel layout, video and statement display, simulated controls, independent distortion calculations, and machine speech after confirmation are implemented. Spring Boot returns the evidence and source link after confirmation. Rendering that response in the central panel is part of the planned installation flow.

The physical controls and the small-talk classifier are in development. The planned small-talk feature lets the machine respond briefly to greetings, thanks, and expressions of surprise during the installation.

## Running the Prototype

1. Follow the [evidence classifier setup](ml/evidence_classifier/README.md) to restore or train E06b and start the local FastAPI service on port 8000.
2. With Java 21 installed, start Spring Boot from the repository root using PowerShell.

   ```powershell
   cd orchestrator
   .\mvnw.cmd spring-boot:run
   ```

3. Open [http://localhost:8080/](http://localhost:8080/) and allow camera access. Use the on-screen slider and buttons to run a round.

## Research

- [Initial ML research](ml/evidence_classifier/docs/ML_RESEARCH_REPORT.md)
- [FEVER experiments](ml/evidence_classifier/docs/FEVER_EXPERIMENTS_REPORT.md)
- [Small-talk classifier plan](ml/intent_classifier/README.md)
