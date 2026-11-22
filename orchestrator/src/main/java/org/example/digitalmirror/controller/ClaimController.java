package org.example.digitalmirror.controller;

import org.example.digitalmirror.dto.*;
import org.example.digitalmirror.service.ClaimService;
import org.example.digitalmirror.service.ClaimSessionService;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/claims")
public class ClaimController {

    private final ClaimService claimService;
    private final ClaimSessionService sessionService;

    public ClaimController(
            ClaimService claimService,
            ClaimSessionService sessionService
    ) {
        this.claimService = claimService;
        this.sessionService = sessionService;
    }

    @GetMapping("/next")
    public ResponseEntity<ClaimDto> nextClaim() {
        return ResponseEntity.ok(
                claimService.getNextClaim()
        );
    }

    @PostMapping("/{id}/analyze")
    public ResponseEntity<ClaimAnalysisResponseDto> analyze(
            @PathVariable String id
    ) {
        return ResponseEntity.ok(
                sessionService.analyze(id)
        );
    }

    @PostMapping("/{id}/confirm")
    public ResponseEntity<ConfirmResponseDto> confirm(
            @PathVariable String id,
            @RequestBody ConfirmRequestDto request
    ) {
        return ResponseEntity.ok(
                sessionService.confirm(
                        id,
                        request.humanScore()
                )
        );
    }

    @GetMapping("/{id}/reveal")
    public ResponseEntity<RevealResponseDto> reveal(
            @PathVariable String id
    ) {
        return ResponseEntity.ok(
                sessionService.reveal(id)
        );
    }

    @ExceptionHandler(IllegalStateException.class)
    public ResponseEntity<String> handleInvalidState(
            IllegalStateException ex
    ) {
        return ResponseEntity.status(HttpStatus.CONFLICT)
                .body(ex.getMessage());
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<String> handleInvalidArgument(
            IllegalArgumentException ex
    ) {
        return ResponseEntity.badRequest()
                .body(ex.getMessage());
    }
}
