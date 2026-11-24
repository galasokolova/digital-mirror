
package org.example.digitalmirror.service;

import org.example.digitalmirror.dto.*;
import org.example.digitalmirror.model.ClaimRecord;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

class ClaimSessionServiceTest {

    private ClaimService claimService;
    private AiAnalysisService aiAnalysisService;
    private ClaimSessionService sessionService;

    @BeforeEach
    void setUp() {
        claimService = mock(ClaimService.class);
        aiAnalysisService = mock(AiAnalysisService.class);

        DistortionService distortionService =
                new DistortionService();

        sessionService = new ClaimSessionService(
                claimService,
                aiAnalysisService,
                distortionService
        );

        ClaimRecord claim = new ClaimRecord(
                "test-001",
                "The committee approved the proposal.",
                "The committee rejected the proposal.",
                "Committee",
                "REFUTES",
                "The evidence contradicts the claim.",
                "Demonstration",
                ""
        );

        when(claimService.findById("test-001"))
                .thenReturn(Optional.of(claim));

        ClaimAnalysisResponseDto prediction =
                new ClaimAnalysisResponseDto(
                        "test-001",
                        "The committee approved the proposal.",
                        "REFUTES",
                        0.30,
                        0.70,
                        "test-model"
                );

        when(aiAnalysisService.analyzeByClaimId("test-001"))
                .thenReturn(prediction);
    }

    @Test
    void shouldCalculateBothDistortions() {
        sessionService.analyze("test-001");

        ConfirmResponseDto result =
                sessionService.confirm("test-001", 0.90);

        assertEquals(0.90, result.humanDistortion(), 1e-9);
        assertEquals(0.30, result.machineDistortion(), 1e-9);

        assertEquals(0.90, result.humanScore(), 1e-9);
        assertEquals(0.30, result.machineScore(), 1e-9);
    }

    @Test
    void shouldRejectConfirmationBeforeAnalysis() {
        assertThrows(
                IllegalStateException.class,
                () -> sessionService.confirm("test-001", 0.90)
        );
    }

    @Test
    void shouldRejectDuplicateConfirmation() {
        sessionService.analyze("test-001");
        sessionService.confirm("test-001", 0.90);

        assertThrows(
                IllegalStateException.class,
                () -> sessionService.confirm("test-001", 0.20)
        );
    }

    @Test
    void shouldRejectInvalidHumanScore() {
        sessionService.analyze("test-001");

        assertThrows(
                IllegalArgumentException.class,
                () -> sessionService.confirm("test-001", 1.5)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> sessionService.confirm("test-001", -0.2)
        );
    }

    @Test
    void shouldAllowRevealOnlyAfterConfirm() {
        sessionService.analyze("test-001");

        assertThrows(
                IllegalStateException.class,
                () -> sessionService.reveal("test-001")
        );

        sessionService.confirm("test-001", 0.90);

        RevealResponseDto reveal =
                sessionService.reveal("test-001");

        assertEquals("REFUTES", reveal.verifiedLabel());
    }
}
