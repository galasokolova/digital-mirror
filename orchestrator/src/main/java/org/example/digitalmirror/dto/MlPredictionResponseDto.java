package org.example.digitalmirror.dto;

public record MlPredictionResponseDto(
        String claim,
        String evidence,
        String wiki_title,
        String prediction,
        double p_supports,
        double p_refutes,
        double confidence,
        String model_version
) {}
