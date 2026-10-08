package org.example.digitalmirror.config;

import dev.langchain4j.model.openai.OpenAiChatModel;
import io.github.cdimascio.dotenv.Dotenv;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class AiConfig {

    @Bean
    public OpenAiChatModel openAiChatModel() {
        Dotenv dotenv = Dotenv.configure().ignoreIfMissing().load();
        String apiKey = dotenv.get("OPENAI_API_KEY", System.getenv("OPENAI_API_KEY"));

        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("Environment variable OPENAI_API_KEY is missing!");
        }

        return OpenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gpt-4o-mini")
                .temperature(0.7)
                .build();
    }
}
