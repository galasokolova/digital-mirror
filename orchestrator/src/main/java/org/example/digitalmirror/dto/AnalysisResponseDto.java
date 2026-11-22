package org.example.digitalmirror.dto;

public record AnalysisResponseDto(
        String claimId,
        String prediction,
        double supportProbability,
        double refuteProbability
) {}