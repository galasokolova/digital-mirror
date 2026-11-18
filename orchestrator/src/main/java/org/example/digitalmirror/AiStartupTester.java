
package org.example.digitalmirror;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

@Component
public class AiStartupTester implements CommandLineRunner {

    private final String aiMode;

    public AiStartupTester(
            @Value("${ai.mode:mock}") String aiMode
    ) {
        this.aiMode = aiMode;
    }

    @Override
    public void run(String... args) {
        System.out.println("----------------------------------------\uD83E\uDD16");
        System.out.println("Digital Mirror AI mode: " + aiMode);

        if ("live".equalsIgnoreCase(aiMode)) {
            System.out.println(
                    "OpenAI live mode enabled. API requests may incur costs."
            );
        } else {
            System.out.println(
                    "MOCK MODE: OpenAI requests are disabled."
            );
        }

        System.out.println("----------------------------------------\uD83E\uDD16");
    }
}
