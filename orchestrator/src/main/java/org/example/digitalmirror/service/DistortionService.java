package org.example.digitalmirror.service;

import org.springframework.stereotype.Service;

@Service
public class DistortionService {

    public double calculate(double score, String verifiedLabel) {

        if (!Double.isFinite(score) || score < 0 || score > 1) {
            throw new IllegalArgumentException(
                    "Score must be between 0 and 1"
            );
        }

        double verifiedValue = switch (verifiedLabel) {
            case "SUPPORTS" -> 1.0;
            case "REFUTES" -> 0.0;
            default -> throw new IllegalArgumentException(
                    "Unknown verified label: " + verifiedLabel
            );
        };

        return Math.abs(score - verifiedValue);
    }
}