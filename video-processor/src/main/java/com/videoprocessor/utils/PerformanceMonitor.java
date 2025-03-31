package com.videoprocessor.utils;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

public class PerformanceMonitor {
    private final ConcurrentHashMap<String, Long> startTimes;
    private final ConcurrentHashMap<String, Long> totalTimes;
    private final ConcurrentHashMap<String, Long> counts;

    public PerformanceMonitor() {
        this.startTimes = new ConcurrentHashMap<>();
        this.totalTimes = new ConcurrentHashMap<>();
        this.counts = new ConcurrentHashMap<>();
    }

    public void startTimer(String operation) {
        startTimes.put(operation, System.nanoTime());
    }

    public void stopTimer(String operation) {
        Long startTime = startTimes.get(operation);
        if (startTime != null) {
            long duration = System.nanoTime() - startTime;
            totalTimes.merge(operation, duration, Long::sum);
            counts.merge(operation, 1L, Long::sum);
        }
    }

    public String generateReport() {
        StringBuilder report = new StringBuilder("Performance Report:\n");
        totalTimes.forEach((operation, totalTime) -> {
            long count = counts.getOrDefault(operation, 1L);
            double avgMs = TimeUnit.NANOSECONDS.toMillis(totalTime) / (double) count;
            report.append(String.format(
                "%s: 总次数=%d, 平均耗时=%.2fms\n",
                operation, count, avgMs
            ));
        });
        return report.toString();
    }
}