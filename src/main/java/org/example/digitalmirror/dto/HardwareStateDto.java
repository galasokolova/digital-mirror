package org.example.digitalmirror.dto;

public record HardwareStateDto(
        int timeline,
        int logic,
        int factCheck,
        int fakeGenerator
) {}
