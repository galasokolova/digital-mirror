package org.example.digitalmirror.dto;

public record ConfirmResponseDto(
        String claimId,
        double humanScore,
        double machineScore,
        double humanDistortion,
        double machineDistortion
) {}