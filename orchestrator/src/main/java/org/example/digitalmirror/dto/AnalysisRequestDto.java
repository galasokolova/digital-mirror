package org.example.digitalmirror.dto;

public record AnalysisRequestDto(
        String headline,
        int logicValue,        // 0..1023
        int factCheckValue,    // 0..1023
        int fakeGeneratorValue,// 0..1023
        int timelineYear       // 1960..2026
) {}
