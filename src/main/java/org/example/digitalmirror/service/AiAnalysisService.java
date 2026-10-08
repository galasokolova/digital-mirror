package org.example.digitalmirror.service;

import dev.langchain4j.model.openai.OpenAiChatModel;
import org.springframework.stereotype.Service;

@Service
public class AiAnalysisService {
    private final OpenAiChatModel chatModel;


    public AiAnalysisService(OpenAiChatModel chatModel) {
        this.chatModel = chatModel;
    }

    public String testConnection(OpenAiChatModel chatModel){
        return chatModel.generate("""
                            Hi! If you reply this prompt,
                            it means my Java + IntelliJ Idea + LangChain4j configured correctly.
                            Reply with one word!
                            """);
    }

    public  String analyzeArticle(String text, int logicLevel, int factLevel){
        // prompt for AI

        return chatModel.generate("Analyze this text: \n" + text);
    }
}
