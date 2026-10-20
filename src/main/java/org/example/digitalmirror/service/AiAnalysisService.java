package org.example.digitalmirror.service;

import dev.langchain4j.model.openai.OpenAiChatModel;
import org.example.digitalmirror.dto.AnalysisRequestDto;
import org.example.digitalmirror.dto.AnalysisResponseDto;
import org.example.digitalmirror.dto.HeadlineDto;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Random;
import java.util.UUID;

@Service
public class AiAnalysisService {
    private final OpenAiChatModel chatModel;
    private final Random random = new Random();

    // Headliners Mock for testing
    private final List<HeadlineDto> headlineBank = List.of(
            new HeadlineDto(UUID.randomUUID().toString(), "Sputnik 1 Launched: Space Race Begins in Cold War Orbit", 1957, 30),
            new HeadlineDto(UUID.randomUUID().toString(), "Moon Landing Broadcast Live Across Global Satellite Network", 1969, 15),
            new HeadlineDto(UUID.randomUUID().toString(), "Deepfake Video Sparks International Verification Crisis", 2024, 85),
            new HeadlineDto(UUID.randomUUID().toString(), "Autonomous Media Synthesizers Curate Epistemic Realities", 2026, 95)
    );

    public AiAnalysisService(OpenAiChatModel chatModel) {
        this.chatModel = chatModel;
    }

    public String testConnection() {
        return chatModel.generate("Hi! Reply with one single word if connected.");
    }

    public HeadlineDto getRandomHeadline() {
        return headlineBank.get(random.nextInt(headlineBank.size()));
    }

    public AnalysisResponseDto analyzeDistortion(AnalysisRequestDto req) {
        // Estimate rate of distortion
        int netFake = Math.max(0, req.fakeGeneratorValue() - req.factCheckValue());

        String prompt = String.format("""
            You are the critical core of an interactive art installation called 'The Anatomy of Distortion: A Digital Mirror'.
            Deconstruct this media headline: '%s'
            Context Year: %d.
            Parameters (0-1023): Logic=%d, FactCheck=%d, FakeGen=%d.
            
            Provide a short 1-2 sentence epistemic critique exposing ideological bias, spin, or synthetic manipulation.
            Keep it poignant and artistic.
            """,
                req.headline(),
                req.timelineYear(),
                req.logicValue(),
                req.factCheckValue(),
                req.fakeGeneratorValue()
        );

        String aiReply = chatModel.generate(prompt);

        // estimate tone for p5.js
        String tone = netFake > 400 ? "GLITCH_RED" : (req.logicValue() > 600 ? "CLEAR" : "BLURRED");
        double biasScore = netFake / 1023.0;

        return new AnalysisResponseDto(aiReply.trim(), biasScore, tone);
    }
}