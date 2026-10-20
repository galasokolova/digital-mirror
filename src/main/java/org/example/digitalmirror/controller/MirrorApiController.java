package org.example.digitalmirror.controller;

import org.example.digitalmirror.dto.AnalysisRequestDto;
import org.example.digitalmirror.dto.AnalysisResponseDto;
import org.example.digitalmirror.dto.HeadlineDto;
import org.example.digitalmirror.service.AiAnalysisService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@CrossOrigin(origins = {"https://preview.p5js.org", "https://editor.p5js.org", "http://localhost:3000", "*"})
@RestController
@RequestMapping("/api")
public class MirrorApiController {

    private final AiAnalysisService aiService;

    public MirrorApiController(AiAnalysisService aiService) {
        this.aiService = aiService;
    }

    // GET http://localhost:8080/api/headline/random
    @GetMapping("/headline/random")
    public ResponseEntity<HeadlineDto> getRandomHeadline() {
        return ResponseEntity.ok(aiService.getRandomHeadline());
    }

    // POST http://localhost:8080/api/analyze (get data from p5.js)
    @PostMapping("/analyze")
    public ResponseEntity<AnalysisResponseDto> analyzeClientState(@RequestBody AnalysisRequestDto request) {
        AnalysisResponseDto response = aiService.analyzeDistortion(request);
        return ResponseEntity.ok(response);
    }
}
