package com.videoprocessor.core;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import static org.junit.jupiter.api.Assertions.*;

class FrameProcessorTest {
    private FrameProcessor processor;
    private Mat testFrame;

    @BeforeEach
    void setUp() {
        // 加载OpenCV库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        processor = new FrameProcessor();
        // 加载测试图片
        testFrame = Imgcodecs.imread("src/test/resources/test-frame.jpg");
    }

    @Test
    void testMD5Confusion() {
        Mat result = processor.processMD5Confusion(testFrame);
        assertNotNull(result);
        assertNotEquals(0, Core.countNonZero(result));
    }

    @Test
    void testKeyFrameReplacement() {
        Mat referenceFrame = testFrame.clone();
        Mat result = processor.replaceKeyFrame(testFrame, referenceFrame);
        assertNotNull(result);
        assertEquals(testFrame.size(), result.size());
    }

    @Test
    void testDynamicStickerGeneration() {
        Mat result = processor.applyDynamicSticker(testFrame)
            .join(); // 等待异步操作完成
        assertNotNull(result);
        assertTrue(result.rows() >= testFrame.rows());
        assertTrue(result.cols() >= testFrame.cols());
    }
}