package com.segmentation.nima;

import android.graphics.Bitmap;

public interface QUSEventListener {
// Segmentation Methods
    void updateSegmentEvent(byte[] output_image, byte[] landmark_frame, Bitmap full_res, int fc);
    void updateEFOutputEvent(float EDVol, float ESVol, float EF, boolean biplane);
// QUS Runner Methods
    void updateResultEvent(float[] quality_results, float[] view_probabilities);

}
