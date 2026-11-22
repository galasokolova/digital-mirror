package org.example.digitalmirror.service;

import org.example.digitalmirror.dto.*;
import org.example.digitalmirror.model.ClaimRecord;
import org.springframework.stereotype.Service;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class ClaimSessionService {

    private final ClaimService claimService;
    private final AiAnalysisService aiAnalysisService;
    private final DistortionService distortionService;

    private final Map<String, ClaimAnalysisResponseDto> predictions =
            new ConcurrentHashMap<>();

    private final Map<String, ConfirmResponseDto> confirmations =
            new ConcurrentHashMap<>();

    public ClaimSessionService(
            ClaimService claimService,
            AiAnalysisService aiAnalysisService,
            DistortionService distortionService
    ) {
        this.claimService = claimService;
        this.aiAnalysisService = aiAnalysisService;
        this.distortionService = distortionService;
    }

    public ClaimAnalysisResponseDto analyze(String claimId) {

        ClaimAnalysisResponseDto prediction =
                aiAnalysisService.analyzeByClaimId(claimId);

        predictions.put(claimId, prediction);
        confirmations.remove(claimId);

        return prediction;
    }

    public ConfirmResponseDto confirm(
            String claimId,
            double humanScore
    ) {

        if (!Double.isFinite(humanScore)
                || humanScore < 0.0
                || humanScore > 1.0) {
            throw new IllegalArgumentException(
                    "Human score must be between 0 and 1"
            );
        }

        // Prevent changing an already confirmed answer.
        if (confirmations.containsKey(claimId)) {
            throw new IllegalStateException(
                    "Claim already confirmed: " + claimId
            );
        }

        ClaimAnalysisResponseDto prediction =
                predictions.get(claimId);

        if (prediction == null) {
            throw new IllegalStateException(
                    "Analyze the claim before confirming"
            );
        }

        ClaimRecord claim = claimService.findById(claimId)
                .orElseThrow(() ->
                        new IllegalArgumentException(
                                "Unknown claim: " + claimId
                        )
                );

        double machineScore = prediction.pSupports();

        double humanDistortion =
                distortionService.calculate(
                        humanScore,
                        claim.verifiedLabel()
                );

        double machineDistortion =
                distortionService.calculate(
                        machineScore,
                        claim.verifiedLabel()
                );

        ConfirmResponseDto result = new ConfirmResponseDto(
                claimId,
                humanScore,
                machineScore,
                humanDistortion,
                machineDistortion
        );

        confirmations.put(claimId, result);

        return result;
    }

    public RevealResponseDto reveal(String claimId) {

        if (!confirmations.containsKey(claimId)) {
            throw new IllegalStateException(
                    "Confirm the claim before revealing"
            );
        }

        ClaimRecord claim = claimService.findById(claimId)
                .orElseThrow(() ->
                        new IllegalArgumentException(
                                "Unknown claim: " + claimId
                        )
                );

        return new RevealResponseDto(
                claim.id(),
                claim.claim(),
                claim.verifiedLabel(),
                claim.evidence(),
                claim.explanation(),
                claim.sourceTitle(),
                claim.sourceUrl()
        );
    }
}
