package org.example.digitalmirror.service;

import org.example.digitalmirror.dto.MlPredictionRequestDto;
import org.example.digitalmirror.dto.MlPredictionResponseDto;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

@Service
public class MlInferenceService {

    private final RestClient restClient;

    public MlInferenceService(
            RestClient.Builder builder,
            @Value("${ml.api.url:http://127.0.0.1:8000}")
            String mlApiUrl
    ) {
        this.restClient = builder
                .baseUrl(mlApiUrl)
                .build();
    }

    public MlPredictionResponseDto predict(
            MlPredictionRequestDto request
    ) {
        if (request == null
                || request.claim() == null
                || request.claim().isBlank()
                || request.evidence() == null
                || request.evidence().isBlank()) {
            throw new IllegalArgumentException(
                    "Claim and evidence are required."
            );
        }

        MlPredictionResponseDto response = restClient.post()
                .uri("/predict")
                .body(request)
                .retrieve()
                .body(MlPredictionResponseDto.class);

        if (response == null) {
            throw new IllegalStateException(
                    "ML API returned an empty response."
            );
        }

        return response;
    }
}
