package org.example.digitalmirror.controller;

import org.example.digitalmirror.dto.MlPredictionRequestDto;
import org.example.digitalmirror.dto.MlPredictionResponseDto;
import org.example.digitalmirror.service.MlInferenceService;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/ml")
public class MlController {

    private final MlInferenceService mlService;

    public MlController(MlInferenceService mlService) {
        this.mlService = mlService;
    }

    @PostMapping("/predict")
    public ResponseEntity<MlPredictionResponseDto> predict(
            @RequestBody MlPredictionRequestDto request
    ) {
        return ResponseEntity.ok(
                mlService.predict(request)
        );
    }
}
