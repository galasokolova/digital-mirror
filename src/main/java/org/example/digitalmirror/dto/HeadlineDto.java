package org.example.digitalmirror.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class HeadlineDto {
    private String id;
    private String text;
    private int year;
    private int baseDistortion;
}
