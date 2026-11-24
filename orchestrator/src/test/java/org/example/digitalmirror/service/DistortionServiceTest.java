package org.example.digitalmirror.service;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.*;

class DistortionServiceTest {

    private DistortionService service;

    @BeforeEach
    void setUp() {
        service = new DistortionService();
    }

    @ParameterizedTest
    @CsvSource({
            "1.0, SUPPORTS, 0.0",
            "0.0, SUPPORTS, 1.0",
            "0.0, REFUTES, 0.0",
            "1.0, REFUTES, 1.0",
            "0.5, REFUTES, 0.5",
            "0.5, SUPPORTS, 0.5",
            "0.2, SUPPORTS, 0.8",
            "0.8, REFUTES, 0.8"
    })
    void shouldCalculateDistortion(
            double score,
            String verifiedLabel,
            double expected
    ) {
        double actual = service.calculate(
                score,
                verifiedLabel
        );

        assertEquals(expected, actual, 1e-9);
    }

    @ParameterizedTest
    @CsvSource({
            "-0.1",
            "1.1"
    })
    void shouldRejectScoresOutsideRange(double score) {
        assertThrows(
                IllegalArgumentException.class,
                () -> service.calculate(score, "SUPPORTS")
        );
    }

    @Test
    void shouldRejectNaN() {
        assertThrows(
                IllegalArgumentException.class,
                () -> service.calculate(
                        Double.NaN,
                        "SUPPORTS"
                )
        );
    }

    @Test
    void shouldRejectInfinity() {
        assertThrows(
                IllegalArgumentException.class,
                () -> service.calculate(
                        Double.POSITIVE_INFINITY,
                        "REFUTES"
                )
        );
    }

    @Test
    void shouldRejectUnknownVerifiedLabel() {
        assertThrows(
                IllegalArgumentException.class,
                () -> service.calculate(
                        0.5,
                        "UNKNOWN"
                )
        );
    }
}
