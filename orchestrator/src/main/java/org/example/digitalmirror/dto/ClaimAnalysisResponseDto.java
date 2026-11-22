package org.example.digitalmirror.dto;

public record ClaimAnalysisResponseDto(
        String claimId,
        String claim,
        String prediction,
        double pSupports,
        double pRefutes,
        String modelVersion
) {}
