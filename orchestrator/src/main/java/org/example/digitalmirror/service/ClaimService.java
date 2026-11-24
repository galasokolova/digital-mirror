package org.example.digitalmirror.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.example.digitalmirror.dto.ClaimDto;
import org.example.digitalmirror.model.ClaimRecord;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

@Service
public class ClaimService {
    private final List<ClaimRecord> claims;
    private final AtomicInteger currentIndex = new AtomicInteger(0);

    public ClaimService(ObjectMapper objectMapper,
                        @Value("classpath:data/claims.json") Resource claimsResource) {
        try (var input = claimsResource.getInputStream()) {
            this.claims = List.copyOf(objectMapper.readValue(
                    input, new TypeReference<List<ClaimRecord>>() {}));
        } catch (IOException e) {
            throw new IllegalStateException("Cannot load data/claims.json", e);
        }
        if (claims.isEmpty()) {
            throw new IllegalStateException("data/claims.json must contain at least one claim");
        }
        Set<String> ids = new HashSet<>();
        for (ClaimRecord claim : claims) {
            if (claim.id() == null || claim.id().isBlank() || !ids.add(claim.id())) {
                throw new IllegalStateException("Missing or duplicate claim ID: " + claim.id());
            }
            if (claim.claim() == null || claim.claim().isBlank()
                    || claim.evidence() == null || claim.evidence().isBlank()
                    || !("SUPPORTS".equals(claim.verifiedLabel())
                    || "REFUTES".equals(claim.verifiedLabel()))) {
                throw new IllegalStateException("Invalid claim data: " + claim.id());
            }
        }
    }

    public ClaimDto getNextClaim() {
        int index = currentIndex.getAndUpdate(i -> i == claims.size() - 1 ? 0 : i + 1);
        ClaimRecord record = claims.get(index);
        return new ClaimDto(record.id(), record.claim());
    }

    public Optional<ClaimRecord> findById(String id) {
        return claims.stream().filter(claim -> claim.id().equals(id)).findFirst();
    }
}
