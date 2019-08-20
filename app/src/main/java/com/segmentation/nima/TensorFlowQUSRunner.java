package com.segmentation.nima;
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


import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
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
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

import static com.segmentation.nima.BubbleService.AP4_IDX;
import static com.segmentation.nima.BubbleService.PLAX_IDX;
import static com.segmentation.nima.BubbleService.PSAXM_IDX;
import static com.segmentation.nima.BubbleService.SUBC4_IDX;
import static com.segmentation.nima.BubbleService.UNINIT_IDX;

// Nima: Implements the interface, so methods should be defined here

/** A classifier specialized to label images using TensorFlow. */
public class TensorFlowQUSRunner implements Classifier {

    private static final String TAG = "nvw-quality";
    private static final boolean ATTEMPT_GPU = false;
    private static final int NUM_VIEWQUAL_RUNNERS = 3;

    // Config values
    private int input_width, input_height, cine_length;
    private int frame_counter;
    private long[] input_dims, output_dims;
    private String input_name;
    private String[] output_names;

    // scoreImage Vars
    private float[] curr_frame;

    // Pre-allocated buffers.
    private int[] input_values;

    // Multithreading
    //private float[] curr_buffer; // just a pointer between the two feature_buffers
    //private float[] feature_buffer1, feature_buffer2;
    //private boolean buffer_toggle;
    //private final Object buffer_lock = new Object();
    private Vector<ViewQualRunner> VQRs;
    private int vqr_toggle = 0;

    private QUSEventListener mQUSEventListener;

    // TFLite objects
    private Vector<Interpreter> interpreters;
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    private TensorFlowQUSRunner() {
    }

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param input_dims The input size. A square image of inputSize x inputSize is assumed.
     * @throws IOException
     */
    public static Classifier create(
            QUSEventListener qusEventListener,
            AssetManager assetManager,
            String model_filename,
            String input_name,
            String output_view_name,
            String output_quality_name,
            long[] input_dims,
            int num_view_classes,
            int num_qual_classes) {
        TensorFlowQUSRunner c = new TensorFlowQUSRunner();

        // Input variables
        c.input_name = input_name;
        c.input_dims = input_dims.clone();
        c.output_dims = new long[]{num_view_classes, num_qual_classes};

        // Default values
        //c.buffer_toggle = false;
        c.frame_counter = 0;                                                                        // Do I really need a frame counter?

        // Inferred variables
        c.output_names = new String[]{output_view_name, output_quality_name};

        // Pre-allocated buffers
        c.cine_length = (int) input_dims[0];
        c.input_width = (int) input_dims[1];
        c.input_height = (int) input_dims[2];
        c.curr_frame = new float[c.input_height * c.input_width];
        c.input_values = new int[c.input_height * c.input_width];

        // Runners
        c.VQRs = new Vector<ViewQualRunner>(NUM_VIEWQUAL_RUNNERS);
        c.interpreters = new Vector<Interpreter>(NUM_VIEWQUAL_RUNNERS);

        // Nima: Hahaha
        // TODO: try this hsit out!
       // c.tfliteOptions.setNumThreads(3);


        // Nima: Not going to attempt GPU
        // Attempt to use GPU if available
        if (!ATTEMPT_GPU) {
            Log.e(TAG, "Not attempting to include GPU support");
        }
        // Nima: What is this??
        // NNAPI not available
        c.tfliteOptions.setUseNNAPI(false);

        // Load .tflite file and check its output sizes
        String model_filepath = model_filename + ".tflite";
        Log.i(TAG, "Attempting to load the segmentation model: " + model_filepath);
        InputStream is = null;
        try {
            is = assetManager.open(model_filepath);
            Log.i(TAG, "File exists!");
        } catch (IOException ex) {
            Log.e(TAG, "File does NOT exist!");
            return null;
        }


        // Nima: Need to look into this as well
        // Need this for finishRecord callback
        c.mQUSEventListener = qusEventListener;

        Trace.beginSection("load");
        try {
            MappedByteBuffer model_mbb = null;
            model_mbb = c.loadModelFile(assetManager, model_filename);
            for(int i = 0; i < NUM_VIEWQUAL_RUNNERS; i++) {
                Interpreter interp = new Interpreter(model_mbb, c.tfliteOptions);
                c.interpreters.add(interp);
                c.VQRs.add(new ViewQualRunner(interp, c.input_name, c.output_names, c.input_dims, c.output_dims, i, c.mQUSEventListener));
            }
            Log.i(TAG, "Nets loaded successfully!");
        } catch (IOException e) {
            Log.e(TAG, "Failed to create MappedByteBuffer from model file: " + model_filename); // model name is fine
            e.printStackTrace();
        }
        Trace.endSection();

        // For debugging input/output names:
        Interpreter interp = c.interpreters.firstElement();
        int inp_idx = interp.getInputIndex(input_name);
        int view_idx = interp.getOutputIndex(output_view_name);
        int qual_idx = interp.getOutputIndex(output_quality_name);
        Log.i(TAG, " inp_idx index: " + inp_idx);
        Log.i(TAG, " view_idx index: " + view_idx);
        Log.i(TAG, " qual_idx index: " + qual_idx);

        int[] inp_shape = interp.getInputTensor(inp_idx).shape();
        int[] view_shape = interp.getOutputTensor(view_idx).shape();
        int[] qual_shape = interp.getOutputTensor(qual_idx).shape();
        Log.i(TAG, "\nInput shape:");
        for (int i = 0; i < inp_shape.length; i++) Log.i(TAG, "dim " + i + " = " + inp_shape[i]);
        Log.i(TAG, "\nSegment shape:");
        for (int i = 0; i < view_shape.length; i++) Log.i(TAG, "dim " + i + " = " + view_shape[i]);
        Log.i(TAG, "\nLandmark shape:");
        for (int i = 0; i < qual_shape.length; i++) Log.i(TAG, "dim " + i + " = " + qual_shape[i]);

        return c;
    }
    // Nima: Already have this
    /**
     * Memory-map the model file in Assets.
     */
    private MappedByteBuffer loadModelFile(AssetManager am, String filename) throws IOException {
        AssetFileDescriptor fileDescriptor = am.openFd(filename + ".tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @Override
    public void scoreImage(Bitmap input_bitmap, Bitmap full_res, int fc) {
    }

    // Nima: Where the magic happens
    @Override
    public void scoreImage(final Bitmap input_bitmap) {
        // Get raw pointer to int ARGB8888 values
        BubbleService.isProcessDone = false;
        int frame_offset = input_width * input_height * frame_counter;
        input_bitmap.getPixels(input_values, 0, input_bitmap.getWidth(), 0, 0, input_bitmap.getWidth(), input_bitmap.getHeight());

        // Preprocess the image data from 0-255 int to normalized float based on the provided parameters.
        float t;
        for (int i = 0; i < input_values.length; ++i) {
            // Only look at 1 of the RGB values to save some CPU
            curr_frame[i] = input_values[i] & 0xff;
        }

        // Wait for current CNNR to finish, then launch it again
        //Log.i(TAG,"Waiting for CNNR1 to finish running...");
        ViewQualRunner VQR = VQRs.get(vqr_toggle);
        vqr_toggle = (vqr_toggle + 1) % NUM_VIEWQUAL_RUNNERS;
        VQR.wakeup(curr_frame);

        // Nima: Could this be related to the main Activity?
        frame_counter++;
        if (frame_counter >= cine_length) {
            // reset frame_counter
            frame_counter = 0;
            /*synchronized (buffer_lock) {
                curr_buffer = (buffer_toggle) ? feature_buffer1 : feature_buffer2;
                buffer_toggle = !buffer_toggle;
            }*/
        }
        input_bitmap.recycle();
    }
    // Nima: Should be able to avoid this
    @Override
    public void setRequestedView(int clicked_view) {
        // This is basically also QUSRunner.Start()
        // i.e. it needs to get called once and only once each new view
//        for (ViewQualRunner c : VQRs)
//            c.setRequestedView(clicked_view);
    }

    @Override
    public void close() {
        // Notify all threads
        for (ViewQualRunner c : VQRs) {
            c.stopThread();
            c.getInterpreter().close();
        }

        // Need to wait for all CNNR's to be finished
        /*int num_stopped = 0;
        while(num_stopped != VQRs.length+1) {
            num_stopped = (RNNR1.isExited)? 1 : 0;
            for (CNNRunner c : CNNRs)
                num_stopped += (c.isExited())? 1 : 0;
        }*/

        curr_frame = null;
        input_values = null;
        input_dims = null;
        //feature_buffer1 = null;
        //feature_buffer2 = null;

    }
    public void tempClose() {
        // Notify all threads
        for (ViewQualRunner c : VQRs) {
            c.stopThread();
        }
    }

    @Override
    public void clearLastResult() {
        frame_counter = 0;
        for (ViewQualRunner c : VQRs)
            c.clearLastResult();
    }

    @Override
    public void zeroLastResult() {
        frame_counter = 0;
        for (ViewQualRunner c : VQRs)
            c.zeroLastResult();
    }

    private static int factorial(long[] in) {
        long s = (in == null) ? 0 : 1;
        for (int i = 0; i < in.length; i++) {
            s *= in[i];
        }
        return (int) s;
    }

    private static class ViewQualRunner implements Runnable {
        private Interpreter interpreter;
        private String input_name;
        private String[] output_names;
        private long[] input_dims;
        private long[] output_dims;
        private float[] input_frame;
        private float[] view_output, qual_output;
        private int requested_view;
        private ByteBuffer input_bb, view_bb, qual_bb;
        private int input_size;
        private String id;
        private final QUSEventListener QEL;

        private final Object tSync = new Object();
        private boolean isRunning, isExited, stopFlag, notifyFlag;
        private boolean logStats = false;
        private Thread t;
        private final Object score_mtx = new Object();

        private int[] runtimes;
        private int rt_counter = 0;

        private ViewQualRunner(Interpreter _interpreter,
                               String _input_name,
                               String[] _output_names,
                               long[] _input_dims,
                               long[] _output_dims,
                               int _id,
                               QUSEventListener _QEL) {
            interpreter = _interpreter;
            input_name = _input_name;
            output_names = _output_names;
            input_dims = _input_dims;
            output_dims = _output_dims;
            input_size = factorial(input_dims);
            id = Integer.toString(_id);
            QEL = _QEL;

            // TODO: do we even need these?
            input_frame = new float[input_size];
            view_output = new float[(int) output_dims[0]];
            qual_output = new float[(int) output_dims[1]];

            isRunning = false;
            isExited = false;
            stopFlag = false;
            notifyFlag = false;
            requested_view = UNINIT_IDX;

            runtimes = new int[100];

            input_bb = ByteBuffer.allocateDirect(input_size * 4);
            input_bb.order(ByteOrder.LITTLE_ENDIAN);

            view_bb = ByteBuffer.allocateDirect((int) output_dims[0] * 4);
            view_bb.order(ByteOrder.LITTLE_ENDIAN);

            qual_bb = ByteBuffer.allocateDirect((int) output_dims[1] * 4);
            qual_bb.order(ByteOrder.LITTLE_ENDIAN);
            t = new Thread(this, "CNN Runner");
            t.start();
        }

        private void wakeup(float[] _input_frame) {
            synchronized (tSync) {
                if (isRunning) {
                    Log.e(TAG, "VQR" + id + " is still running!");
                    try {
                        while (isRunning) {
                            Log.i(TAG, "...");
                            tSync.wait(5);
                        }
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }

                // Move these inside the lock so they dont get changed without consent
                input_frame = _input_frame;

                notifyFlag = true;
                tSync.notify();
            }
        }

        private void stopThread() {
            synchronized (tSync) {
                stopFlag = true;
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
                // First, wait until you are woken up by super class
                synchronized (tSync) {
                    if (notifyFlag)
                        Log.e(TAG, "CNN" + id + " missed a notify(), better catch up!!");
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

                isRunning = true;
                long _w = System.currentTimeMillis();

                // Construct the input byteBuffer
                if (input_bb == null) return;
                input_bb.rewind();
                for (int i = 0; i < input_size; i++)
                    input_bb.putFloat(input_frame[i]);
                long feed = System.currentTimeMillis() - _w;
                _w = System.currentTimeMillis();

                // Prepare in/outputs in desired format
                Object[] input_obj = {input_bb};
                Map<Integer, Object> output_map = new HashMap<>();
                view_bb.rewind();
                qual_bb.rewind();
                output_map.put(0, view_bb);
                output_map.put(1, qual_bb);
                long prep = System.currentTimeMillis() - _w;
                _w = System.currentTimeMillis();

                // Run it!
                Trace.beginSection("run");
                interpreter.runForMultipleInputsOutputs(input_obj, output_map);
                Trace.endSection();
                long run = System.currentTimeMillis() - _w;
                _w = System.currentTimeMillis();

                // Fetch output
                //System.arraycopy(sn_outputs, 0, segment_frame, 0, output_size);
                for (int i = 0; i < output_dims[0]; i++) view_output[i] = view_bb.getFloat(i * 4);
                for (int i = 0; i < output_dims[1]; i++) qual_output[i] = qual_bb.getFloat(i * 4);
                long fetch = System.currentTimeMillis() - _w;
                _w = System.currentTimeMillis();

                /*Log.i(TAG, "\tQuality mean = " + qual_output[0] * 100 + " +/- " + qual_output[1] * 100 + "%");
                Log.i(TAG, "\tView probs: \n\t" +
                        "AP2  : " + view_output[AP2_IDX] + "\n\t" +
                        "AP4  : " + view_output[AP4_IDX] + "\n\t" +
                        "PLX  : " + view_output[PLAX_IDX] + "\n\t" +
                        "SC4  : " + view_output[SUBC4_IDX] + "\n\t" +
                        "PSXAo: " + view_output[PSAXAo_IDX] + "\n\t" +
                        "PSXM : " + view_output[PSAXM_IDX] + "\n\t" +
                        "PSXPM: " + view_output[PSAXPM_IDX] + "\n\t" +
                        "PSXAp: " + view_output[PSAXAp_IDX]);*/

                //Log.i(TAG,"RNN took "+(System.currentTimeMillis()-_w));total
                // Report timing
                long total = feed + run + fetch;
                //Log.i(TAG, "TFLite took " + feed + " + " + prep + " + " + run + " + " + fetch + " = " + total + "ms");

                runtimes[rt_counter++] = (int)total;
                if(rt_counter >= 100){
                    rt_counter = 0;
                    float ave = 0;
                    for(int i = 0; i < 100; i++)
                        ave += runtimes[i];
                    Log.i(TAG,"Average runtime for VQR "+id+" = "+ave/100.0f);
                }

                // Set results for next callback
                synchronized (score_mtx) {
                    // TODO: do we need really need an extra save?
                    //System.arraycopy(qual_output, 0, lastQualResults, 0, num_qual_classes);
                    //System.arraycopy(view_output, 0, lastViewProbs, 0, num_view_classes);

                    // Create a updateResult callback
                    //long elapsed = System.currentTimeMillis() - watch;
                    //watch = System.currentTimeMillis();
                    QEL.updateResultEvent(view_output, qual_output);
                }
                isRunning = false;
            }
        }

        private synchronized void clearLastResult() {
            Log.i(TAG, "Clearing last result");
            synchronized (score_mtx) {
                Log.i(TAG, "This does happen");
                //TODO: fix this
                /*
                for(int i = 0; i < output_dims[0]; i++) view_output[i] = -1;
                for(int i = 0; i < output_dims[1]; i++) qual_output[i] = -1;
                QEL.updateResultEvent(requested_view, view_output, qual_output);
                */
            }
            Log.i(TAG, "last result cleared");
        }

        private synchronized void zeroLastResult() {
            synchronized (score_mtx) {
                // TODO: fix this
                /*
                lastResult = 0.0f; // Show what exactly...
                //lastView = num_view_classes; // Set to UNINIT
                QEL.updateResultEvent(requested_view, lastQualResults, lastViewProbs);
                //clearerHandler.postDelayed(resultClearer,150);
                */
            }
        }

        private void setRequestedView(int v) {
            switch (v) {
                case 0: requested_view = AP4_IDX; break;
                case 1: requested_view = PLAX_IDX; break;
                case 2: requested_view = PSAXM_IDX; break;
                case 3: requested_view = SUBC4_IDX; break;
            }
        }

        private Interpreter getInterpreter(){return interpreter;}
    }
}
/*
    private static class RNNRunner implements Runnable {
        // Input variables
        private TensorFlowInferenceInterface inferenceInterface;
        private String input_name;
        private String[] output_names;
        private long [] input_dims;
        private int num_qual_classes;
        private int num_view_classes;
        private QUSEventListener QEL;
        private char id;

        // Thread variables
        private int requested_view;
        private float[] lastQualResults, lastViewProbs;
        private float lastResult;
        private final Object score_mtx = new Object();
        private final Object tSync = new Object();
        private float [] input_features;
        public boolean isRunning, isExited, stopFlag, notifyFlag;
        private Thread t;
        private Handler clearerHandler = new Handler();
        private boolean logStats = false;

        private RNNRunner(  TensorFlowInferenceInterface _inferenceInterface,
                            String _input_name,
                            String[] _output_names,
                            long [] _input_dims,
                            int _num_qual_classes,
                            int _num_view_classes,
                            QUSEventListener _qusEventListener,
                            char _id) {
            inferenceInterface = _inferenceInterface;
            input_name = _input_name;
            output_names = _output_names;
            input_dims = _input_dims;
            num_qual_classes = _num_qual_classes;
            num_view_classes = _num_view_classes;
            QEL = _qusEventListener;
            id = _id;

            lastQualResults = new float[num_qual_classes];
            lastViewProbs = new float[num_view_classes]; // Set to UNINIT
            requested_view = num_view_classes; // Set to UNINIT
            clearLastResult();
            isRunning = false;
            isExited = false;
            stopFlag = false;
            notifyFlag = false;
            t = new Thread(this, "RNN Runner");
            t.start();
        }

        private void wakeup(float[] feature_buffer)
        {
            synchronized (tSync) {
                if (isRunning) {
                    Log.e(TAG, "RNN is STILL running!");
                    // Option 1: Wait for it to finish, and then immediately launch another
                    try {
                        while (isRunning) {
                            Log.e(TAG, "......");
                            tSync.wait(50); // waiting here should cause the calling CNNR to wait, causing the main thread to wait
                        }
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }

                this.input_features = feature_buffer;
                notifyFlag = true;
                tSync.notify();
            }
        }

        private void stopThread(){
            synchronized (tSync) {
                stopFlag = true;
                tSync.notify();
            }
        }

        public boolean isRunning(){return isRunning;}
        public boolean isExited() {
            return isExited;
        }

        public void run()
        {
            while(true) {
                // First, wait until you are woken up by super class
                synchronized (tSync) {
                    if (notifyFlag) Log.e(TAG, "RNN" + id + " missed a notify(), better catch up!!");
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

                // Let other threads see that the net is running
                isRunning = true;
                //next_saved_tensor.setRunning(true); // moving this to 20th CNN call

                // Notify has been called. 20 frames buffered. Run the inference call.
                //Log.v("nvw-runner","Running net, on timestamp "+next_saved_tensor.GetTimestamp());
                long _w = System.currentTimeMillis();

                //Trace.beginSection("feed");
                inferenceInterface.feed(input_name, input_features, input_dims[0], input_dims[1], input_dims[2], input_dims[3]);
                //Trace.endSection();
                //feed = System.currentTimeMillis() - _w; _w = System.currentTimeMillis();

                //Trace.beginSection("run");
                inferenceInterface.run(output_names, logStats);
                //Trace.endSection();
                //run = System.currentTimeMillis() - _w; _w = System.currentTimeMillis();

                // Copy the output Tensor back into the output array.
                float[] qual_outputs = new float[num_qual_classes];
                //Trace.beginSection("fetch");
                inferenceInterface.fetch(output_names[0], qual_outputs);
                //Trace.endSection();

                // Copy the output Tensor back into the output array.
                float[] view_outputs = new float[num_view_classes];
                //Trace.beginSection("fetch");
                inferenceInterface.fetch(output_names[1], view_outputs);
                //Trace.endSection();
                //fetch = System.currentTimeMillis() - _w; _w = System.currentTimeMillis();

                Log.i(TAG,"\tQuality mean = "+qual_outputs[0]*100+" +/- "+qual_outputs[1]*100+"%");
                Log.i(TAG, "\tView probs: \n\t" +
                        "AP2  : "+view_outputs[AP2_IDX]+"\n\t"+
                        "AP4  : "+view_outputs[AP4_IDX]+"\n\t"+
                        "PLX  : "+view_outputs[PLAX_IDX]+"\n\t"+
                        "SC4  : "+view_outputs[SUBC4_IDX]+"\n\t"+
                        "PSXAo: "+view_outputs[PSAXAo_IDX]+"\n\t"+
                        "PSXM : "+view_outputs[PSAXM_IDX]+"\n\t"+
                        "PSXPM: "+view_outputs[PSAXPM_IDX]+"\n\t"+
                        "PSXAp: "+view_outputs[PSAXAp_IDX]+"\n\t"+
                        "OTH  : "+view_outputs[OTHER_IDX]);*/

                /*float view_max = 0;
                int view_idx = -1;
                for(int i = 0; i < num_view_classes; i++){
                    if(view_outputs[i] > view_max){
                        view_max = view_outputs[i];
                        view_idx = i;
                    }
                }

                //Log.i(TAG,"RNN took "+(System.currentTimeMillis()-_w));

                // Set results for next callback
                synchronized(score_mtx) {
                    System.arraycopy(qual_outputs,0,lastQualResults,0,num_qual_classes);
                    System.arraycopy(view_outputs,0,lastViewProbs,0,num_view_classes);

                    // Create a updateResult callback
                    //long elapsed = System.currentTimeMillis() - watch;
                    //watch = System.currentTimeMillis();
                    QEL.updateResultEvent(requested_view, lastQualResults, lastViewProbs);
                }

                // Finally, set the runCompleted flag to let wakeup know
                isRunning = false;
                //next_saved_tensor.setRunning(true); This is done in SavedTensor.saveResult()
            }
        }

        private synchronized void clearLastResult() {
            Log.i(TAG, "Clearing last result");
            synchronized(score_mtx){
                Log.i(TAG, "This does happen");
                for(int i = 0; i < num_qual_classes; i++) lastQualResults[i] = 0; // Show N/A
                for(int i = 0; i < num_view_classes; i++) lastViewProbs[i] = 0;       // Set to UNINIT
                QEL.updateResultEvent(requested_view, lastQualResults, lastViewProbs);
                //clearerHandler.postDelayed(resultClearer,150);
            }
            Log.i(TAG, "last result cleared");
        }

        private synchronized void zeroLastResult() {
            synchronized(score_mtx){
                lastResult = 0.0f; // Show what exactly...
                //lastView = num_view_classes; // Set to UNINIT
                QEL.updateResultEvent(requested_view, lastQualResults, lastViewProbs);
                //clearerHandler.postDelayed(resultClearer,150);
            }
        }

        private void setRequestedView(int v){
            switch(v){
                case 0: requested_view = AP4_IDX; break;
                case 1: requested_view = PLAX_IDX; break;
                case 2: requested_view = PSAXM_IDX; break;
                case 3: requested_view = SUBC4_IDX; break;
            }
        }
    }
}
*/
