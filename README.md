# Digital Mirror

**Digital Mirror** is an interactive installation prototype exploring how digital information, artificial intelligence, and physical human interaction can influence a visual experience.

The project combines a Java Spring Boot backend, a browser-based visual interface, and Arduino hardware controls.

## Technology Stack

- **Backend:** Java, Spring Boot, Maven
- **Frontend:** JavaScript, p5.js, ml5.js, HTML, CSS
- **Hardware:** Arduino Uno, four potentiometers
- **Communication:** REST API and USB serial

## Project Structure

```text
digital-mirror/
├── orchestrator/    # Spring Boot application and REST API
├── frontend/        # Browser interface and visual effects
├── arduino/         # Arduino firmware and hardware documentation
├── .gitignore
└── README.md
```

## Architecture

The Spring Boot orchestrator serves the frontend, exposes REST endpoints, and communicates with the Arduino.

The frontend provides the visual interface, camera processing, and interactive effects.

The Arduino reads four analog controls and returns their values to the orchestrator using a request-response serial protocol.

## Current Development Status

**Implemented or integrated:**
- Spring Boot application
- Browser-based visual interface
- Camera and FaceMesh integration
- Arduino serial communication
- Four physical analog controls
- Separate frontend, backend, and hardware directories

**Under development:**
- AI-powered news analysis and verification
- Further interactive visualization features
- Final installation design and user experience

Some AI-related responses currently use placeholder data.

## Running the Project

### Requirements

- Java and Maven
- A modern browser with camera access
- Arduino Uno with the provided firmware
- USB connection to the Arduino

### Start the Orchestrator

From the repository root:

```powershell
cd orchestrator
.\mvnw.cmd spring-boot:run
```

Open:

http://localhost:8080/

The frontend is served by Spring Boot; a separate frontend development server is not required.

## Hardware

See [Arduino Hardware Documentation](arduino/README.md) for the firmware, communication protocol, and prototype photographs.

## Development

Digital Mirror is an evolving prototype. Its architecture, interaction design, and AI components are being developed incrementally.