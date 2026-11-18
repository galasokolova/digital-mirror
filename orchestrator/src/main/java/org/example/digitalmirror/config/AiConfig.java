
package org.example.digitalmirror.config;

import dev.langchain4j.model.openai.OpenAiChatModel;
import io.github.cdimascio.dotenv.Dotenv;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class AiConfig {

    @Bean
    @ConditionalOnProperty(
            name = "ai.mode",
            havingValue = "live"  // OpenAiChatModel is created only if ai.mode=live
    )
    public OpenAiChatModel openAiChatModel() {
        Dotenv dotenv = Dotenv.configure()
                .ignoreIfMissing()
                .load();

        String apiKey = dotenv.get(
                "OPENAI_API_KEY",
                System.getenv("OPENAI_API_KEY")
        );

        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException(
                    "OPENAI_API_KEY is required in live mode."
            );
        }

        return OpenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gpt-4o-mini")
                .temperature(0.7)
                .build();
    }
}
