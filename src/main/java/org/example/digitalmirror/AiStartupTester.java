package org.example.digitalmirror;

import dev.langchain4j.model.openai.OpenAiChatModel;
import org.example.digitalmirror.service.AiAnalysisService;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

@Component
public class AiStartupTester implements CommandLineRunner {
    private final AiAnalysisService aiService;
    private final OpenAiChatModel chatModel;

    public AiStartupTester(AiAnalysisService aiService, OpenAiChatModel chatModel) {
        this.aiService = aiService;
        this.chatModel = chatModel;
    }

    @Override
    public void run(String... args) {
        System.out.println("🔄 Checking OpenAI connection...");
        try {
            String reply = aiService.testConnection(chatModel);
            System.out.println("\n✅ Success! AI replies:");
            System.out.println("----------------------------------------");
            System.out.println(reply);
            System.out.println("----------------------------------------");
        } catch (Exception e) {
            System.err.println("❌ OpenAI API connection failed: " + e.getMessage());
        }
    }
}
