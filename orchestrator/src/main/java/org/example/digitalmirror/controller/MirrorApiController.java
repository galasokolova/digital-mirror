package org.example.digitalmirror.controller;

import org.example.digitalmirror.dto.AnalysisRequestDto;
import org.example.digitalmirror.dto.AnalysisResponseDto;
import org.example.digitalmirror.dto.HardwareStateDto;
import org.example.digitalmirror.dto.HeadlineDto;
import org.example.digitalmirror.service.AiAnalysisService;
import org.example.digitalmirror.service.ArduinoSerialService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@CrossOrigin(origins = {"https://preview.p5js.org", "https://editor.p5js.org", "http://localhost:3000", "*"})
@RestController
@RequestMapping("/api")
public class MirrorApiController {

    private final AiAnalysisService aiService;
    private final ArduinoSerialService hardwareService;

    public MirrorApiController(AiAnalysisService aiService, ArduinoSerialService hardwareService) {
        this.aiService = aiService;
        this.hardwareService = hardwareService;
    }

    @GetMapping("/headline/random")
    public ResponseEntity<HeadlineDto> getRandomHeadline() {
        return ResponseEntity.ok(aiService.getRandomHeadline());
    }

    @PostMapping("/analyze")
    public ResponseEntity<AnalysisResponseDto> analyzeClaim(
            @RequestBody AnalysisRequestDto request
    ) {
        return ResponseEntity.ok(aiService.analyzeClaim(request));
    }

    @GetMapping("/hardware/state")
    public ResponseEntity<HardwareStateDto> getHardwareState() {
        return ResponseEntity.ok(hardwareService.getLatestHardwareState());
    }
}