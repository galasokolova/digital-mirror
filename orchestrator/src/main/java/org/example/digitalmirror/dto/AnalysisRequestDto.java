package org.example.digitalmirror.dto;

public record AnalysisRequestDto(
        String claimId,
        String claim,
        String evidence,
        String wikiTitles
) {}