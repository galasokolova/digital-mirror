package org.example.digitalmirror.dto;

public record HeadlineDto(
        String id,
        String text,
        int year,
        int baseDistortion,
        String type,  // "REAL" or "FAKE"
        String era    // "PRE_INTERNET", "INTERNET_ERA", "AI_ERA"
) {
    public HeadlineDto(String id, String text, int year, int baseDistortion) {
        this(id, text, year, baseDistortion, "REAL", "INTERNET_ERA");
    }
}