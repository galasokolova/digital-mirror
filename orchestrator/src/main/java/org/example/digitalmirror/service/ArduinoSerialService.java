package org.example.digitalmirror.service;

import com.fazecast.jSerialComm.SerialPort;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.example.digitalmirror.dto.HardwareStateDto;
import org.springframework.stereotype.Service;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.atomic.AtomicReference;

@Service
public class ArduinoSerialService {

    private SerialPort activePort;
    private Thread listenerThread;
    private volatile boolean isRunning = false;

    private final AtomicReference<HardwareStateDto> latestState =
            new AtomicReference<>(new HardwareStateDto(512, 512, 512, 0));

    @PostConstruct
    public void startSerialListener() {
        SerialPort[] ports = SerialPort.getCommPorts();
        if (ports.length == 0) {
            System.err.println("❌ [Hardware] No COM ports found!");
            return;
        }

        for (SerialPort port : ports) {
            String name = port.getSystemPortName().toUpperCase();
            String desc = port.getDescriptivePortName().toLowerCase();
            if (name.equals("COM3") || desc.contains("arduino") || desc.contains("ch340") || desc.contains("usb-serial")) {
                activePort = port;
                break;
            }
        }

        if (activePort == null) {
            activePort = ports[0];
        }

        activePort.setBaudRate(9600);
        // Таймаут чтения 150 мс
        activePort.setComPortTimeouts(SerialPort.TIMEOUT_READ_BLOCKING, 150, 0);

        if (!activePort.openPort()) {
            System.err.println("❌ [Hardware] Port open FAILED!");
            return;
        }

        activePort.setDTR();
        activePort.setRTS();

        System.out.println("✅ [Hardware] Connected on port: " + activePort.getSystemPortName());
        isRunning = true;

        listenerThread = new Thread(this::pollArduinoLoop, "Arduino-Polling-Thread");
        listenerThread.setDaemon(true);
        listenerThread.start();
    }

    private void pollArduinoLoop() {
        try {
            Thread.sleep(1500);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return;
        }

        System.out.println("🚀 [Hardware] Direct polling loop ACTIVE.");

        try (InputStream in = activePort.getInputStream();
             OutputStream out = activePort.getOutputStream()) {

            byte[] buffer = new byte[128];

            while (isRunning) {
                // Signal to the Arduino board
                out.write('?');
                out.flush();

                // Reading the reply
                StringBuilder sb = new StringBuilder();
                int b;
                long startTime = System.currentTimeMillis();

                while ((System.currentTimeMillis() - startTime) < 100) {
                    if (in.available() > 0) {
                        b = in.read();
                        if (b == -1 || b == '\n') break;
                        if (b != '\r') sb.append((char) b);
                    } else {
                        Thread.sleep(2);
                    }
                }

                String line = sb.toString().trim();
                if (!line.isEmpty()) {
                    String[] parts = line.split(",");
                    if (parts.length == 4) {
                        try {
                            int t = Integer.parseInt(parts[0].trim());
                            int l = Integer.parseInt(parts[1].trim());
                            int f = Integer.parseInt(parts[2].trim());
                            int fk = Integer.parseInt(parts[3].trim());

                            latestState.set(new HardwareStateDto(t, l, f, fk));
                        } catch (NumberFormatException ignored) {}
                    }
                }

                // Frequency of request
                Thread.sleep(40);
            }
        } catch (Exception e) {
            System.err.println("💥 [Hardware] Polling error: " + e.getMessage());
        }
    }

    public HardwareStateDto getLatestHardwareState() {
        return latestState.get();
    }

    @PreDestroy
    public void cleanup() {
        isRunning = false;
        if (activePort != null && activePort.isOpen()) {
            activePort.clearDTR();
            activePort.clearRTS();
            activePort.closePort();
            System.out.println("🔌 [Hardware] Port closed cleanly.");
        }
    }
}