package com.videoprocessor.benchmark;

import org.openjdk.jmh.annotations.*;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import com.videoprocessor.core.FrameProcessor;

import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(value = 1, warmups = 1)
@Warmup(iterations = 2)
@Measurement(iterations = 3)
public class ProcessingBenchmark {
    private FrameProcessor processor;
    private Mat testFrame;

    @Setup
    public void setup() {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        processor = new FrameProcessor();
        testFrame = Imgcodecs.imread("src/test/resources/test-frame.jpg");
    }

    @Benchmark
    public void benchmarkMD5Confusion() {
        processor.processMD5Confusion(testFrame);
    }

    @Benchmark
    public void benchmarkKeyFrameReplacement() {
        processor.replaceKeyFrame(testFrame, testFrame.clone());
    }

    @Benchmark
    public void benchmarkStickerGeneration() {
        processor.applyDynamicSticker(testFrame).join();
    }
}