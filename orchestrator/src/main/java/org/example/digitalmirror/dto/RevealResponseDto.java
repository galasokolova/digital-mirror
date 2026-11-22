
package org.example.digitalmirror.dto;

public record RevealResponseDto(
        String claimId,
        String claim,
        String verifiedLabel,
        String evidence,
        String explanation,
        String sourceTitle,
        String sourceUrl
) {}
