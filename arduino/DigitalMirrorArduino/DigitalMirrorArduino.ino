void setup() {
    Serial.begin(9600);
}

void loop() {
    // Reply only when Java explicitly requests data
    if (Serial.available() > 0) {

        while (Serial.available() > 0) {
            Serial.read();
        }

        int t = analogRead(A0);
        int l = analogRead(A1);
        int f = analogRead(A2);
        int fk = analogRead(A3);

        Serial.print(t);
        Serial.print(",");
        Serial.print(l);
        Serial.print(",");
        Serial.print(f);
        Serial.print(",");
        Serial.println(fk);
    }
}