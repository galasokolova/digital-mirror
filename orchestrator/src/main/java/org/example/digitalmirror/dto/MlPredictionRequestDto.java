
package org.example.digitalmirror.dto;

public record MlPredictionRequestDto(
        String claim,
        String evidence,
        String wiki_title
) {}
