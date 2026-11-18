
package org.example.digitalmirror.service;

import dev.langchain4j.model.openai.OpenAiChatModel;
import org.example.digitalmirror.dto.AnalysisRequestDto;
import org.example.digitalmirror.dto.AnalysisResponseDto;
import org.example.digitalmirror.dto.HeadlineDto;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Random;
import java.util.UUID;

@Service
public class AiAnalysisService {

    private final ObjectProvider<OpenAiChatModel> chatModelProvider;
    private final boolean liveMode;
    private final Random random = new Random();

    // Headlines mock for testing
    private final List<HeadlineDto> headlineBank = List.of(
            new HeadlineDto(UUID.randomUUID().toString(),
                    "Sputnik 1 Launched: Space Race Begins in Cold War Orbit", 1957, 30),
            new HeadlineDto(UUID.randomUUID().toString(),
                    "Moon Landing Broadcast Live Across Global Satellite Network", 1969, 15),
            new HeadlineDto(UUID.randomUUID().toString(),
                    "Deepfake Video Sparks International Verification Crisis", 2024, 85),
            new HeadlineDto(UUID.randomUUID().toString(),
                    "Autonomous Media Synthesizers Curate Epistemic Realities", 2026, 95)
    );

    public AiAnalysisService(
            ObjectProvider<OpenAiChatModel> chatModelProvider,
            @Value("${ai.mode:mock}") String aiMode
    ) {
        this.chatModelProvider = chatModelProvider;
        this.liveMode = "live".equalsIgnoreCase(aiMode);
    }

    private OpenAiChatModel requireLiveModel() {
        OpenAiChatModel model = chatModelProvider.getIfAvailable();

        if (model == null) {
            throw new IllegalStateException(
                    "OpenAI model is not configured for live mode."
            );
        }

        return model;
    }

    public String testConnection() {
        if (!liveMode) {
            return "[MOCK] OpenAI connection disabled.";
        }

        return requireLiveModel().generate(
                "Hi! Reply with one single word if connected."
        );
    }

    public HeadlineDto getRandomHeadline() {
        return headlineBank.get(
                random.nextInt(headlineBank.size())
        );
    }

    public AnalysisResponseDto analyzeDistortion(AnalysisRequestDto req) {

        int netFake = Math.max(
                0,
                req.fakeGeneratorValue() - req.factCheckValue()
        );

        String tone = netFake > 400
                ? "GLITCH_RED"
                : (req.logicValue() > 600 ? "CLEAR" : "BLURRED");

        double biasScore = netFake / 1023.0;

        String aiReply;

        if (!liveMode) {
            aiReply = """
                    [MOCK ANALYSIS]
                    This is a simulated response for the Digital Mirror prototype.
                    No factual verification or AI analysis has been performed.
                    """;
        } else {
            String prompt = String.format("""
                    You are the critical core of an interactive art installation
                    called 'The Anatomy of Distortion: A Digital Mirror'.

                    Deconstruct this media headline: '%s'
                    Context Year: %d.
                    Parameters (0-1023): Logic=%d, FactCheck=%d, FakeGen=%d.

                    Provide a short 1-2 sentence critique exposing ideological
                    bias, spin, or synthetic manipulation.
                    Keep it artistic.
                    """,
                    req.headline(),
                    req.timelineYear(),
                    req.logicValue(),
                    req.factCheckValue(),
                    req.fakeGeneratorValue()
            );

            aiReply = requireLiveModel().generate(prompt);
        }

        return new AnalysisResponseDto(
                aiReply.trim(),
                biasScore,
                tone
        );
    }
}
