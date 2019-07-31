package com.segmentation.nima;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Handler;
import android.os.Looper;
import android.os.Process;
import android.os.Trace;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

public class TensorFlowQUSRunnerSegment implements Classifier {

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

    /** A classifier specialized to label images using TensorFlow. */
        private static final String TAG = "nvw-segment";
        private static final boolean ATTEMPT_GPU = false;
        private static final int NUM_SEGNET_RUNNERS = 2;
        // Config values
        private int input_width, input_height, input_length;
        private float magnitude;
        private long[] input_dims, output_dims;
        private String[] input_names, output_names;
        // Pre-allocated buffers.
        private int[] input_values;
        private float[] curr_frame;
        // Multithreading
        private int frame_counter = 0;
        private Vector<SegNetRunner> SNRs;
        private int snr_toggle = 0;
        private boolean logStats = false;
        private QUSEventListener mQUSEventListener;
        private Vector<Interpreter> interpreters;
        // Hold a bunch of tflite parameters
        private final Interpreter.Options tfliteOptions = new Interpreter.Options();
        private TensorFlowQUSRunnerSegment() {
        }
        /**
         * Initializes a native TensorFlow session for classifying images.
         */
        public static Classifier create(
                QUSEventListener qusEventListener,
                AssetManager assetManager,
                String model_filename,
                float magnitude,
                String input_name,
                String segment_name,
                String landmark_name,
                long[] input_dims,
                long[] output_dims) {
            TensorFlowQUSRunnerSegment c = new TensorFlowQUSRunnerSegment();
            // Input variables
            c.magnitude = magnitude;
            c.input_dims = input_dims.clone();
            c.output_dims = output_dims.clone();
            // Inferred variables
            c.input_names = new String[]{input_name};
            c.output_names = new String[]{segment_name, landmark_name};
            int output_size = (int) c.factorial(output_dims);
            Log.i(TAG, " OUTPUT size = " + output_size);
            // Pre-allocated buffers
            c.input_length = (int) input_dims[0];
            c.input_width = (int) input_dims[1];
            c.input_height = (int) input_dims[2];
            c.curr_frame = new float[c.input_height * c.input_width];
            c.input_values = new int[c.input_height * c.input_width];
            c.interpreters = new Vector<Interpreter>(NUM_SEGNET_RUNNERS);
            c.SNRs = new Vector<SegNetRunner>(NUM_SEGNET_RUNNERS);
            // Need this for finishRecord callback
            c.mQUSEventListener = qusEventListener;
            // TODO: try this hsit out!
//            c.tfliteOptions.setNumThreads(3);
            // Attempt to use GPU if available
            if (!ATTEMPT_GPU) {
                Log.w(TAG,"Not attempting GPU in this build.");
            } else {
                Log.w(TAG,"Attempting to use GPU in this build");
            }
            // Check to see whether the app crashes if  you set this to true
            c.tfliteOptions.setUseNNAPI(false);
            // Load .tflite file and check its output sizes
            String model_filepath = model_filename+".tflite";
            Log.i(TAG, "Attempting to load the segmentation model: " + model_filepath);
            InputStream is = null;
            try {
                is = assetManager.open(model_filepath);
                Log.i(TAG,"File exists!");
            } catch (IOException ex) {
                Log.i(TAG,"File does NOT exist!");
            }
            Trace.beginSection("load");
            try {
                MappedByteBuffer model_mbb = null;
                model_mbb = c.loadModelFile(assetManager, model_filename);
                for(int i = 0; i < NUM_SEGNET_RUNNERS; i++) {
                    Interpreter interp = new Interpreter(model_mbb, c.tfliteOptions);
                    c.interpreters.add(interp);
                    c.SNRs.add(new SegNetRunner(interp, c.input_names, c.output_names, c.input_dims, c.output_dims, i, c.mQUSEventListener));
                }
                Log.i(TAG,NUM_SEGNET_RUNNERS+" segnet(s) loaded successfully!");
            } catch(IOException e){
                Log.e(TAG,"Failed to create MappedByteBuffer from model file: "+model_filename);
                e.printStackTrace();
            }
            Trace.endSection();
            // For Debugging input/output names
            Interpreter interp = c.interpreters.firstElement();
            int inp_idx = interp.getInputIndex(input_name);
            int seg_idx = interp.getOutputIndex(segment_name);
            int lnd_idx = interp.getOutputIndex(landmark_name);
            Log.i(TAG, " Input index: " + inp_idx);
            Log.i(TAG, " Outpt index: " + seg_idx);
            Log.i(TAG, " Outpt index: " + lnd_idx);
            int [] inp_shape = interp.getInputTensor(inp_idx).shape();
            int [] seg_shape = interp.getOutputTensor(seg_idx).shape();
            int [] lnd_shape = interp.getOutputTensor(lnd_idx).shape();
            Log.i(TAG,"\nInput shape:"); for (int i = 0; i < inp_shape.length; i++) Log.i(TAG, "dim "+i+" = "+inp_shape[i]);
            Log.i(TAG,"\nSegment shape:"); for (int i = 0; i < seg_shape.length; i++) Log.i(TAG, "dim "+i+" = "+seg_shape[i]);
            Log.i(TAG,"\nLandmark shape:"); for (int i = 0; i < lnd_shape.length; i++) Log.i(TAG, "dim "+i+" = "+lnd_shape[i]);
            return c;
        }
        /** Memory-map the model file in Assets. */
        private MappedByteBuffer loadModelFile(AssetManager am, String filename) throws IOException {
            AssetFileDescriptor fileDescriptor = am.openFd(filename+".tflite");
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
        @Override
        public void scoreImage(final Bitmap input_bitmap, final Bitmap full_res, int fc) {
            BubbleService.isProcessDone = false;
            //long _w = System.currentTimeMillis();
            // Get raw pointer to int ARGB8888 values
            input_bitmap.getPixels(input_values, 0, input_bitmap.getWidth(), 0, 0, input_bitmap.getWidth(), input_bitmap.getHeight());
            // Preprocess the image data from 0-255 int to normalized float based on the provided parameters.
            for (int i = 0; i < input_values.length; ++i) {
                // Only look at 1 of the RGB values to same some CPU
                curr_frame[i] = (input_values[i] & 0xff)/magnitude;
            }
            // Wait for current SNR to finish, then launch it again
            //Log.i(TAG,"Waiting for SNR1 to finish running...");
            SegNetRunner SNR = SNRs.get(snr_toggle);
            snr_toggle = (snr_toggle+1)%SNRs.size();
            SNR.wakeup(curr_frame, full_res, fc); //TODO: how does curr_frame (shared) not get overwritten??
            frame_counter++;
            //Log.i(TAG,"scoreimage took: "+(System.currentTimeMillis() - _w)+" ms");
        }
        @Override
        public void scoreImage(final Bitmap n1){ }
        @Override
        public void close() {
            // Send stop signal to all SNRs
                        for (SegNetRunner c :SNRs)
                c.stopThread();
            // Need to wait for all SNR's to be finished
            int num_stopped = 0;
            while(num_stopped != SNRs.size()) {
                num_stopped = 0;
                for (SegNetRunner c : SNRs)
                    num_stopped += (c.isExited())? 1 : 0;
            }
            // Close all interpreters
            for (Interpreter i : interpreters)
                i.close();
            input_values = null;
            input_dims = null;
            curr_frame = null;
        }
        @Override
        public void clearLastResult() {}
        @Override
        public void zeroLastResult() {}
        @Override
        public void setRequestedView(int n1) {}
        private static int factorial(long[] in) {
            long s = (in == null) ? 0 : 1;
            for (int i = 0; i < in.length; i++) {
                s *= in[i];
            }
            return (int) s;
        }
        private static class SegNetRunner extends BubbleService implements Runnable {
            private Interpreter interpreter;
            private String[] input_names, output_names;
            private long[] input_dims;
            private long[] output_dims;
            private float[] input_frame;
            //private float[][] sn_outputs = null;
            private byte[] segment_frame, landmark_frame;
            private int frame_counter;
            private int input_length, frame_size, output_size;
            private String id;
            private QUSEventListener QEL;
            private final Object tSync = new Object();
            private boolean isRunning, isExited;
            private boolean stopFlag, notifyFlag;
            private boolean logStats = false;
            private Thread t;
            private Bitmap full_res;
            private ByteBuffer input_bb, segment_bb, landmark_bb;


            private SegNetRunner(Interpreter _interpreter,
                                 String[] _input_names,
                                 String[] _output_names,
                                 long[] _input_dims,
                                 long[] _output_dims,
                                 int _id,
                                 QUSEventListener _qel) {
                interpreter = _interpreter;
                input_names = _input_names;
                output_names = _output_names;
                input_dims = _input_dims;
                output_dims = _output_dims;
                input_length = (int)input_dims[0];
                frame_size = (int)(input_dims[1]*input_dims[2]);
                output_size = factorial(output_dims);
                id = Integer.toString(_id);
                QEL = _qel;
                // TODO: do we even need these?
                input_frame = new float[frame_size*input_length];
                //sn_outputs = new float[2][output_size];
                //full_res = Bitmap.createBitmap(_frw,_frh,Bitmap.Config.ARGB_8888);
                segment_frame = new byte[output_size];
                landmark_frame = new byte[output_size];
                isRunning = false;
                isExited = false;
                stopFlag = false;
                notifyFlag = false;
                input_bb = ByteBuffer.allocateDirect(frame_size*input_length*4);
                input_bb.order(ByteOrder.LITTLE_ENDIAN);
                segment_bb = ByteBuffer.allocateDirect(output_size*4);
                segment_bb.order(ByteOrder.LITTLE_ENDIAN);
                landmark_bb = ByteBuffer.allocateDirect(output_size*4);
                landmark_bb.order(ByteOrder.LITTLE_ENDIAN);
                t = new Thread(this, "SN Runner");
                t.setPriority(10);
                t.start();
            }
            private void stopThread(){
                synchronized (tSync) {
                    stopFlag = true;
                    tSync.notify();
                }
            }
            private void wakeup(float[] _curr_frame, Bitmap _full_res, int fc) {
               synchronized (tSync) {
                    if (isRunning) {
                        Log.e(TAG, "SN" + id + " is still running!");
                        try {
                            while (isRunning) {
                                Log.i(TAG, "...");
                                tSync.wait(100);
                            }
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                    // Move these inside the lock so they dont get changed without consent
                    // TODO: I still dont get how this isnt getting overwritten...
                    //System.arraycopy(_curr_frame, 0, input_frame, 0, (int)(input_dims[0]*input_dims[1]));
                    //full_res = _full_res.copy(Bitmap.Config.ARGB_8888,false);
                    //TODO: deep-copy full-res bitmap here?
                    input_frame = _curr_frame;
                    full_res = _full_res;
                    // Copy frame_counter for synchronization purposes
                    frame_counter = fc;
                    notifyFlag = true;
                    tSync.notify();
                }
            }
            public boolean isRunning() {
                return isRunning;
            }
            public boolean isExited() {
                return isExited;
            }
            public void run() {
                while (true) {
                    //Process.setThreadPriority(-19);
                    // First, wait until you are woken up by super class
                    synchronized (tSync) {
                        if (notifyFlag)
                            Log.e(TAG, "Segnet" + id + " missed a notify(), better catch up!!");
                        try {
                            while (!notifyFlag && !stopFlag)
                                tSync.wait();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                            continue;
                        }
                        notifyFlag = false;
                        if (stopFlag) {
                            isExited = true;
                            return;
                        }
                    }
//                    Log.e(TAG, "priority "+Process.getThreadPriority(3));
                    isRunning = true;
                    long _w = System.currentTimeMillis();
                    Trace.beginSection("feed");
                    if(input_bb == null) return;
                    input_bb.rewind();
                    for(int i = 0; i < frame_size*input_length; i++) {
                        input_bb.putFloat(input_frame[i]);
                    }
                    Trace.endSection();
                    long feed = System.currentTimeMillis() - _w; _w = System.currentTimeMillis();
                    // Prepare in/outputs in desired format
                    Object[] input_obj = {input_bb};
                    Map<Integer, Object> output_map = new HashMap<>();
                    segment_bb.rewind();
                    landmark_bb.rewind();
                    output_map.put(0, segment_bb);
                    output_map.put(1, landmark_bb);
                    long prep = System.currentTimeMillis() - _w; _w = System.currentTimeMillis();
                    Trace.beginSection("run");
                    interpreter.runForMultipleInputsOutputs(input_obj, output_map);
                    Trace.endSection();
                    long run = System.currentTimeMillis() - _w; _w = System.currentTimeMillis();
                    //System.arraycopy(sn_outputs, 0, segment_frame, 0, output_size);
                    for (int i = 0; i < output_size; i++) {
                        segment_frame[i] = (segment_bb.getFloat(i*4) > BubbleService.THRESHOLD) ? (byte) 0x01 : (byte) 0x00;
                        landmark_frame[i] = (landmark_bb.getFloat(i*4) > BubbleService.THRESHOLD) ? (byte) 0x01 : (byte) 0x00;
                    }
                    long fetch = System.currentTimeMillis() - _w; _w = System.currentTimeMillis();
                    long total = feed+run+fetch;
                    Log.i(TAG,"TFLite took "+feed+" + "+prep+" + "+run+" + "+fetch+" = "+total+"ms");
                    isRunning = false;
                    //keepNLargestCCs(segment_frame,1);
                    //keepNLargestCCs(landmark_frame,2);
                    // Do something with the result that just got copied into segment_frame
//                    new Handler(Looper.getMainLooper()).post(new Runnable() {
//                        @Override
//                        public void run() {
                            // things to do on the main thread
                            QEL.updateSegmentEvent(segment_frame, landmark_frame, full_res, frame_counter);
                        }
//                      });
//
//                }
            }
            private void keepNLargestCCs(byte[] frame, int N) {
                ArrayList<ConnectedComponent> ccs = new ArrayList<>();
                int cc_count = 0;
                int w = (int)input_dims[1];
                int h = (int)input_dims[2];
                // Find all connected components in frame
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        // First pixel in the next connected component
                        if (frame[y * w + x] == 0x01) {
                            ConnectedComponent new_cc = new ConnectedComponent(w, h, frame, x, y, ++cc_count, (byte)0x01);
                            ccs.add(new_cc); // Add it to the list
                        }
                    }
                }
                if(N > ccs.size()){
                    Log.e(TAG,"Not enough CCs to keep!");
                    return;
                }
                // Sort the cc list
                Collections.sort(ccs, new Comparator<ConnectedComponent>() {
                    @Override
                    public int compare(ConnectedComponent o1, ConnectedComponent o2) {
                        return (o2.getCount() - o1.getCount());
                    }
                });
                // Select the largest N
                Vector valid_vals = new Vector<>();
                for (int n = 0; n < N; n++)
                    valid_vals.add(ccs.get(n).getValue());
                // Set valid CC's to 1
                for (int i = 0; i < frame.length; i++)
                    frame[i] = (valid_vals.contains((int)frame[i]))? (byte)0x01 : (byte)0x00;
        }
    }
}
