package com.videoprocessor.core;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import com.videoprocessor.utils.PerformanceMonitor;
import java.security.MessageDigest;
import java.util.concurrent.CompletableFuture;

public class FrameProcessor {
    private final PerformanceMonitor monitor;

    public FrameProcessor() {
        this.monitor = new PerformanceMonitor();
    }

    /**
     * 对视频帧进行MD5混淆处理
     * @param frame 输入帧
     * @return 处理后的帧
     */
    public Mat processMD5Confusion(Mat frame) {
        monitor.startTimer("md5_confusion");
        try {
            // 将帧转换为字节数组
            MatOfByte mob = new MatOfByte();
            Imgcodecs.imencode(".jpg", frame, mob);
            byte[] imageBytes = mob.toArray();

            // 计算MD5
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] hash = md.digest(imageBytes);

            // 基于MD5值修改像素
            Mat result = frame.clone();
            for (int i = 0; i < hash.length; i++) {
                int row = (hash[i] & 0xFF) % result.rows();
                int col = (hash[(i + 1) % hash.length] & 0xFF) % result.cols();
                result.put(row, col, new byte[]{hash[i]});
            }

            return result;
        } catch (Exception e) {
            throw new RuntimeException("MD5混淆处理失败", e);
        } finally {
            monitor.stopTimer("md5_confusion");
        }
    }

    /**
     * 替换关键帧
     * @param frame 原始关键帧
     * @param referenceFrame 参考帧
     * @return 处理后的帧
     */
    public Mat replaceKeyFrame(Mat frame, Mat referenceFrame) {
        monitor.startTimer("key_frame_replacement");
        try {
            Mat result = new Mat();
            // 使用光流法进行帧间插值
            Mat flow = new Mat();
            Imgproc.calcOpticalFlowFarneback(
                frame, referenceFrame, flow, 0.5, 3, 15, 3, 5, 1.2, 0
            );

            // 基于光流进行帧重建
            Mat mask = new Mat();
            Core.normalize(flow, mask, 0, 255, Core.NORM_MINMAX);
            
            // 混合原始帧和参考帧
            Core.addWeighted(frame, 0.7, referenceFrame, 0.3, 0, result);
            
            return result;
        } finally {
            monitor.stopTimer("key_frame_replacement");
        }
    }

    /**
     * 异步生成并应用动态贴纸
     * @param frame 输入帧
     * @return 带有贴纸的帧
     */
    public CompletableFuture<Mat> applyDynamicSticker(Mat frame) {
        return CompletableFuture.supplyAsync(() -> {
            monitor.startTimer("sticker_generation");
            try {
                StableDiffusionClient sdClient = new StableDiffusionClient();
                Mat sticker = sdClient.generateSticker(frame);
                
                // 将贴纸叠加到原始帧上
                Mat result = frame.clone();
                Rect roi = new Rect(0, 0, sticker.cols(), sticker.rows());
                Mat destinationROI = result.submat(roi);
                Core.addWeighted(destinationROI, 0.7, sticker, 0.3, 0, destinationROI);
                
                return result;
            } finally {
                monitor.stopTimer("sticker_generation");
            }
        });
    }

    /**
     * 获取性能监控报告
     * @return 性能报告字符串
     */
    public String getPerformanceReport() {
        return monitor.generateReport();
    }
}