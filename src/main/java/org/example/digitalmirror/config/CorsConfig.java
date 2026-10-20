package org.example.digitalmirror.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;
import org.springframework.web.filter.CorsFilter;

@Configuration
public class CorsConfig {

    @Bean
    public CorsFilter corsFilter() {
        CorsConfiguration config = new CorsConfiguration();
        config.setAllowCredentials(false);
        config.addAllowedOriginPattern("*"); // Allows https://preview.p5js.org, and localhost
        config.addAllowedHeader("*");
        config.addAllowedMethod("*");

        // Chromium Private Network Access support (requests from public web to localhost)
        config.addExposedHeader("Access-Control-Allow-Private-Network");

        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", config);
        return new CorsFilter(source);
    }
}