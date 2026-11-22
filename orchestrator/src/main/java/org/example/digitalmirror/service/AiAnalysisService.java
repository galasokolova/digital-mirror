package org.example.digitalmirror.service;

import org.example.digitalmirror.dto.AnalysisRequestDto;
import org.example.digitalmirror.dto.AnalysisResponseDto;
import org.example.digitalmirror.dto.HeadlineDto;
import org.example.digitalmirror.dto.MlPredictionRequestDto;
import org.example.digitalmirror.dto.MlPredictionResponseDto;
import org.example.digitalmirror.dto.ClaimAnalysisResponseDto;
import org.example.digitalmirror.model.ClaimRecord;

import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Random;
import java.util.UUID;

@Service
public class AiAnalysisService {

    private final MlInferenceService mlInferenceService;
    private final Random random = new Random();
    private final ClaimService claimService;

    // Temporary claims for integration testing
    private final List<HeadlineDto> headlineBank = List.of(
            new HeadlineDto(
                    UUID.randomUUID().toString(),
                    "The committee approved the proposal."
            ),
            new HeadlineDto(
                    UUID.randomUUID().toString(),
                    "The European Union imposed sanctions on Russia."
            ),
            new HeadlineDto(
                    UUID.randomUUID().toString(),
                    "The first human landed on the Moon in 1969."
            )
    );

    public AiAnalysisService(MlInferenceService mlInferenceService,
                             ClaimService claimService) {
        this.mlInferenceService = mlInferenceService;
        this.claimService = claimService;
    }

    public HeadlineDto getRandomHeadline() {
        return headlineBank.get(
                random.nextInt(headlineBank.size())
        );
    }


    public ClaimAnalysisResponseDto analyzeByClaimId(String claimId) {

        ClaimRecord record = claimService.findById(claimId)
                .orElseThrow(() ->
                        new IllegalArgumentException(
                                "Unknown claim ID: " + claimId
                        )
                );

        MlPredictionRequestDto request =
                new MlPredictionRequestDto(
                        record.claim(),
                        record.evidence(),
                        record.wikiTitle()
                );

        MlPredictionResponseDto prediction =
                mlInferenceService.predict(request);

        return new ClaimAnalysisResponseDto(
                record.id(),
                record.claim(),
                prediction.prediction(),
                prediction.p_supports(),
                prediction.p_refutes(),
                prediction.model_version()
        );
    }


    public AnalysisResponseDto analyzeClaim(AnalysisRequestDto request) {

        MlPredictionRequestDto mlRequest =
                new MlPredictionRequestDto(
                        request.claim(),
                        request.evidence(),
                        request.wikiTitles()
                );

        MlPredictionResponseDto prediction =
                mlInferenceService.predict(mlRequest);

        return new AnalysisResponseDto(
                request.claimId(),
                prediction.prediction(),
                prediction.p_supports(),
                prediction.p_refutes()
        );
    }
}
