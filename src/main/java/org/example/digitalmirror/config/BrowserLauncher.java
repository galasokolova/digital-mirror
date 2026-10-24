package org.example.digitalmirror.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

import java.io.IOException;

@Component
public class BrowserLauncher {

    @Value("${server.port:8080}")
    private String serverPort;

    @EventListener(ApplicationReadyEvent.class)
    public void launchBrowser() {
        String url = "http://localhost:" + serverPort + "/";
        String os = System.getProperty("os.name").toLowerCase();

        try {
            if (os.contains("win")) {
                // Windows (cmd /c start)
                new ProcessBuilder("cmd", "/c", "start", url).start();
            } else if (os.contains("mac")) {
                // macOS
                new ProcessBuilder("open", url).start();
            } else if (os.contains("nix") || os.contains("nux")) {
                // Linux (Ubuntu, Debian и др.)
                new ProcessBuilder("xdg-open", url).start();
            }
            System.out.println("🌐 Browser opened at: " + url);
        } catch (IOException e) {
            System.err.println("⚠️ Could not open browser automatically: " + e.getMessage());
        }
    }
}