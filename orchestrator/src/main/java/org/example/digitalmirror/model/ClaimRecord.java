package org.example.digitalmirror.model;

public record ClaimRecord(
        String id,
        String claim,
        String evidence,
        String wikiTitle,
        String verifiedLabel,
        String explanation,
        String sourceTitle,
        String sourceUrl
) {}
