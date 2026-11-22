
package org.example.digitalmirror.service;

import org.example.digitalmirror.dto.ClaimDto;
import org.example.digitalmirror.model.ClaimRecord;

import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class ClaimService {

    private final List<ClaimRecord> claims = List.of(
            new ClaimRecord(
                    "demo-001",
                    "The committee approved the proposal.",
                    "The committee rejected the proposal.",
                    "Committee",
                    "REFUTES",
                    "The evidence states that the committee rejected the proposal.",
                    "Demonstration example — not a verified source",
                    ""
            )
    );

    public ClaimDto getNextClaim() {
        ClaimRecord record = claims.getFirst();

        return new ClaimDto(
                record.id(),
                record.claim()
        );
    }

    public Optional<ClaimRecord> findById(String id) {
        return claims.stream()
                .filter(claim -> claim.id().equals(id))
                .findFirst();
    }
}
