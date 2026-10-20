package org.example.digitalmirror.dto;

public record AnalysisResponseDto(
        String analysis,
        double biasScore,
        String visualTone
) {}
