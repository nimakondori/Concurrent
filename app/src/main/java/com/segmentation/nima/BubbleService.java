package com.segmentation.nima;

import android.annotation.SuppressLint;
import android.app.Service;
import android.content.Intent;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.PixelFormat;
import android.graphics.Point;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Display;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import com.segmentation.nima.databinding.BottomsheetBinding;
import com.segmentation.nima.databinding.BubbleLayoutBinding;
import com.segmentation.nima.databinding.ClipLayoutBinding;
import com.segmentation.nima.databinding.LayoutBottomSheetBinding;
import com.segmentation.nima.databinding.ScreensheetBinding;
import com.segmentation.nima.databinding.SelectionBarBinding;
import com.segmentation.nima.databinding.TrashLayoutBinding;
import com.segmentation.nima.env.ImageUtils;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.nio.ByteBuffer;
import java.text.DecimalFormat;
import java.util.Vector;

import static com.segmentation.nima.MainActivity.sMediaProjection;
import static com.segmentation.nima.layoutBuilder.buildLayoutParamsForBottomSheet;
import static com.segmentation.nima.layoutBuilder.buildLayoutParamsForBubble;



public class BubbleService extends Service
        implements QUSEventListener {

    private static WindowManager mWindowManager;
    public  BubbleLayoutBinding mBubbleLayoutBinding;
    public  WindowManager.LayoutParams mBubbleLayoutParams;
    public SelectionBarBinding mSelectionBarBinding;
    public WindowManager.LayoutParams mSelectionLayoutParams;
    private TrashLayoutBinding mTrashLayoutBinding;
    private WindowManager.LayoutParams mTrashLayoutParams;
    private ClipLayoutBinding mClipLayoutBinding;
    private int[] closeRegion = null;//left, top, right, bottom
    private boolean isClipMode;
    private ImageReader imageReader;
    private VirtualDisplay virtualDisplay;
    private ScreensheetBinding mScreenSheetBinding;
    private WindowManager.LayoutParams mScreenSheetBindingParams;
    private BottomsheetBinding mBottomsheetBinding;
    private WindowManager.LayoutParams mBottomsheetlayoutParams;
    private LayoutBottomSheetBinding mLayoutBottomSheetBinding;
    private WindowManager.LayoutParams mLayoutBottomSheetParams;
    private Handler handler;
    private HandlerThread handlerThread;
    private Canvas canvas = new Canvas();
    private ResultsBinding display_results = new ResultsBinding();
    private Bitmap bitmapCut;
    private Bitmap bitmap;
    private Bitmap bit;
    private ImageView iV;
    private TextView tV, depth_textview;
    private BottomSheetHandler mBottomSheetHandler;
    public int[] ClipRegioBubble = new int[4];
    public static boolean displaySegment = false;
    private SeekBar mSeekBar;
    public static float depth = 10.0f;
    private String EF_string;

    //====================================================================================== QUS Segment  Variables ===================================================================================
    // ---------------------------------- Network Constants ----------------------------------------
    // Shared Net Constants
    public static final String[] VIEW_NAMES = {"AP2", "AP3", "AP4", "AP5", "PLAX",
            "RVIF", "SUBC4", "SUBC5", "SUBCIVC", "PSAXA",
            "PSAXM", "PSAXPM", "PSAXAPIX", "SUPRA", "OTHER",
            "UNINIT"}; // always add unint at the ned but don't include it in the length
    public static final int OTHER_IDX = 14;

    // Quality Net Constants

    private static final int CNN_INPUT_WIDTH = 128;
    private static final int CNN_INPUT_HEIGHT = 128;
    //    private static final int RNN_INPUT_FRAMES = 10;
    private static final int NETWORK_FRAMES = 1;
    private static final long[] CNN_INPUT_DIMS = {NETWORK_FRAMES, CNN_INPUT_WIDTH, CNN_INPUT_HEIGHT};
//    private static final long[] RNN_INPUT_DIMS = {RNN_INPUT_FRAMES, 14, 14, 34};

    // Segment Net Constants
    private static final String[] SEGNET_FILENAMES = {"SEG_AP4", "SEG_AP2"};
    private static final String SEGNET_INPUT_NAME = "input_1";
    private static final String SEGNET_OUTPUT_NAME = "LV_seg/Sigmoid";
    private static final String LANDMARK_OUTPUT_NAME = "LV_landmark/Sigmoid";
    private static final int SEGNET_INPUT_WIDTH = 128;
    private static final int SEGNET_INPUT_HEIGHT = 128;
    private static final float SEGNET_MAGNITUDE = 255;
    private static final int SEGNET_INPUT_FRAMES = 1;
    private static final long[] SEGNET_INPUT_DIMS = {SEGNET_INPUT_FRAMES, SEGNET_INPUT_WIDTH, SEGNET_INPUT_HEIGHT};
    private static final long[] SEGNET_OUTPUT_DIMS = {SEGNET_INPUT_WIDTH, SEGNET_INPUT_HEIGHT, 1};


    // -------------------------------------- App Constants ----------------------------------------
    // Shared Constants
    private final String TAG = "NKo-main";
    private static final String[] CINE_NAMES = {"AP4_mo2","AP2_mo2"};
    public static final int AP4_VIEW_IDX = 0;
    public static final int AP2_VIEW_IDX = 1;
    private static final int[] CIRCUIT = {AP4_VIEW_IDX, AP2_VIEW_IDX};

    // Quality Constants
    private static final int FILTER_LENGTH = 20;
    private static final float ALPHA = 0.1f;

    // Segment Constants
    // not final anymore
    public static int SEGMENT_RECORD_LENGTH = 60; // NOTE: THIS MUST BE <= # of .bmps
    // Max heart rate before aliasing = 90 bmp?
    // Min heart rate capturable = 45 bpm
    private static final int INITIAL_DEPTH_SETTING = 100; // [mm]
    public static final float THRESHOLD = 0.5f;
    private int frame_counter = 0;


    // -------------------------------------- App Variables ----------------------------------------
    // Shared Variables
    private int current_view = 0; // 0 -> AP4, 1 -> AP2

//        // Quality Variables
//        private Classifier QualityRunner;
//        private static float[] view_ema_arr = new float[NUM_VIEW_CLASSES];
//        private static float lastResult;
//        private static int lastReqView;
//        private static int lastPredView;

    // Segment Variables
    private Classifier SegmentRunner;
    private Classifier QUSRunner;
    private EFCalculator mEFCalculator;
    private Bitmap outputBitmap, resizedOutputBitmap;           // raw and resized output colormaps
    private int[] output_pixels;                                // pixel vector used to create output Bitmap
    private byte[] recorded_segment_data;                       // recorded 128x128 segmentations
    private byte[] recorded_landmark_data;                     // recorded 128x128 landmark
    private android.graphics.Matrix outputToResizeTransform;    // resizes the 128x128 output map to the preview dims
    private static boolean[] valid_segment_frames;
    public boolean EFCalculated;

    private QUSEventListener QEL;

// ======================================================================================= Timing Analysis Vars ==================================================================================

    private static double initialTime = 0;
    private static double finalTime = 0;
    public  static boolean stop = true;
    public  static boolean hasBeenRunning = false;
    public  static int count = 0;
    public  static boolean isProcessDone = true;

    // ======================================================================================= QUSRunner Variables ==================================================================================
    private static final String CNN_FILENAME = "allCNN_5m_lap";
    private static final String CNN_INPUT_NAME = "input";
    private static final String CNN_QUALITY_NAME = "pred2/Mean";
    private static final String CNN_VIEW_NAME = "pred3/concat";
    public static final int NUM_VIEW_CLASSES = 14;
    private static final int NUM_QUAL_CLASSES = 2;
    public static final int AP2_IDX = 0;
    public static final int AP3_IDX = 1;
    public static final int AP4_IDX = 2;
    public static final int PLAX_IDX = 4;
    public static final int SUBC4_IDX = 6;
    public static final int SUBIVC_IDX = 8;
    public static final int PSAXAo_IDX = 9;
    public static final int PSAXM_IDX = 10;
    public static final int PSAXPM_IDX = 11;
    public static final int PSAXAp_IDX = 12;
    //public static final int OTHER_IDX = 14;
    public static final int UNINIT_IDX = 14;
    private static float [] view_ema_arr = new float[NUM_VIEW_CLASSES];
    private static final float alpha = 0.1f;
    private final Object results_mtx = new Object();
    Vector res_mean_vec = new Vector();
    Vector res_std_vec = new Vector();
    private static float[] lastResults;
    //    private static float lastResults_qus = new float[2]
    private static int state;
    private static final int INPUT_WIDTH = 128;
    private static final int INPUT_HEIGHT = 128;
    private GoalProgressBar progressBar;
    public static float filt_mean = 0, filt_std = 0;

    public BubbleService() {
    }

    private static void onClick(View view) {
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
//        Log.d("kanna", "onStart");
//        if (sMediaProjection != null) {
//            Log.d("kanna", "mediaProjection alive");
//        }
//          Init segmentation buffers
        Bundle Extras = intent.getExtras();
        state = (int) Extras.get("State");

        if (state == 010)
            current_view = 0; //This loads AP4 model
        else if (state == 001)
            current_view = 1; //This should load the AP2 model
// Initiailize necessary stuff
            mBottomSheetHandler = new BottomSheetHandler(this);
            mSeekBar = new SeekBar(this);
            iV = new ImageView(this);
            tV = new TextView(this);
            depth_textview = new TextView(this);
            mBottomSheetHandler.settV1(tV);
            mBottomSheetHandler.setSeekBar(mSeekBar);
            display_results.setiV(iV);
            QEL = this;
            recorded_segment_data = new byte[SEGNET_INPUT_WIDTH * SEGNET_INPUT_HEIGHT * SEGMENT_RECORD_LENGTH];   // 128*128*100*4 bytes = 6.25 MB
            recorded_landmark_data = new byte[SEGNET_INPUT_WIDTH * SEGNET_INPUT_HEIGHT * SEGMENT_RECORD_LENGTH]; // 128*128*100*4 bytes = 6.25 MB
            valid_segment_frames = new boolean[SEGMENT_RECORD_LENGTH];
            mEFCalculator = new EFCalculator(this, SEGNET_INPUT_WIDTH, SEGNET_INPUT_HEIGHT, SEGMENT_RECORD_LENGTH);
            // Create output bitmaps
            outputBitmap = Bitmap.createBitmap(SEGNET_INPUT_WIDTH, SEGNET_INPUT_HEIGHT, Bitmap.Config.ARGB_8888);
            output_pixels = new int[SEGNET_INPUT_WIDTH * SEGNET_INPUT_HEIGHT];
            mEFCalculator.setDepth(depth);
            handlerThread = new HandlerThread("Inference");
            handlerThread.start();
            handler = new Handler(handlerThread.getLooper());
            initial();

// =======================================================================================================================================================================================
        return super.onStartCommand(intent, flags, startId);
    }

    private void initial() {
        LayoutInflater layoutInflater = LayoutInflater.from(this);
        mTrashLayoutBinding = TrashLayoutBinding.inflate(layoutInflater);
        if (mTrashLayoutParams == null) {
            mTrashLayoutParams = buildLayoutParamsForBubble(0, 200);
            mTrashLayoutParams.gravity = Gravity.BOTTOM | Gravity.CENTER_HORIZONTAL;
        }
        getWindowManager().addView(mTrashLayoutBinding.getRoot(), mTrashLayoutParams);
        mTrashLayoutBinding.getRoot().setVisibility(View.GONE);
        mBubbleLayoutBinding = BubbleLayoutBinding.inflate(layoutInflater);
        if (mBubbleLayoutParams == null) {
            mBubbleLayoutParams = buildLayoutParamsForBubble(60, 60);
            mBubbleLayoutBinding.bubble1.setBackground(getDrawable(R.drawable.video_camera));
        }
        getWindowManager().addView(mBubbleLayoutBinding.getRoot(),mBubbleLayoutParams);
        mBubbleLayoutBinding.setHandler(new BubbleHandler(this));
        // Inflate the screensheet here add view later
        mScreenSheetBinding = ScreensheetBinding.inflate(layoutInflater);
        mSelectionBarBinding = SelectionBarBinding.inflate(layoutInflater);
        mBottomsheetBinding = BottomsheetBinding.inflate(layoutInflater);
        if(mBottomsheetlayoutParams == null){
            mBottomsheetlayoutParams = buildLayoutParamsForBottomSheet(0,0);
        }
        getWindowManager().addView(mBottomsheetBinding.getRoot(), mBottomsheetlayoutParams);
        mBottomsheetBinding.setHandler(mBottomSheetHandler);
        mBottomsheetBinding.getRoot().setVisibility(View.GONE);
        mSeekBar = mBottomsheetBinding.seekBar;
        mSeekBar.setProgress(INITIAL_DEPTH_SETTING);
        addListenerOnDepthSeekBar();
        depth_textview = mBottomsheetBinding.tVDepth;
        tV = mBottomsheetBinding.tV;

        if(SegmentRunner == null)
        {
            //initialize segment runner here
            SegmentRunner =
                    TensorFlowQUSRunnerSegment.create(
                            QEL, //TODO: WILL THIS WORK???
                            getAssets(),
                            SEGNET_FILENAMES[current_view],
                            SEGNET_MAGNITUDE,
                            SEGNET_INPUT_NAME,
                            SEGNET_OUTPUT_NAME,
                            LANDMARK_OUTPUT_NAME,
                            SEGNET_INPUT_DIMS,
                            SEGNET_OUTPUT_DIMS);

            if(SegmentRunner == null) {
                Log.e("Nima", "Error has occurred while loading QUSRunner, exiting...");
                return;
            }
        }
        mLayoutBottomSheetBinding = LayoutBottomSheetBinding.inflate(layoutInflater);
        if (mLayoutBottomSheetParams == null) {
            mLayoutBottomSheetParams = buildLayoutParamsForBottomSheet(0, 0);
        }
        getWindowManager().addView(mLayoutBottomSheetBinding.getRoot(), mLayoutBottomSheetParams);
        mLayoutBottomSheetBinding.getRoot().setVisibility(View.GONE);
        mLayoutBottomSheetBinding.setHandler(new BottomSheetHandler(this));
        progressBar = mLayoutBottomSheetBinding.progressBar;
        if(QUSRunner == null) {
            QUSRunner =
                    TensorFlowQUSRunner.create(
                            this,
                            getAssets(),
                            CNN_FILENAME,
                            CNN_INPUT_NAME,
                            CNN_VIEW_NAME,
                            CNN_QUALITY_NAME,
                            CNN_INPUT_DIMS,
                            NUM_VIEW_CLASSES,
                            NUM_QUAL_CLASSES);
        }
        if(QUSRunner == null) {
            Log.e("Nima", "Error has occurred while loading QUSRunner, exiting...");
        }
        lastResults = new float [2];
        display_results.setBar(progressBar);
        if(mSelectionLayoutParams == null )
        {
            mSelectionLayoutParams = buildLayoutParamsForBubble(0,200);
        }
        getWindowManager().addView(mSelectionBarBinding.getRoot(), mSelectionLayoutParams);
        mSelectionBarBinding.getRoot().setVisibility(View.GONE);
        mSelectionBarBinding.qusBtn.setOnClickListener(v -> {
            stop = true;
            //Needs proper screen cleanup
            if(mSelectionBarBinding.getRoot().getVisibility() == View.VISIBLE)
                mLayoutBottomSheetBinding.getRoot().setVisibility(View.GONE);
            if(mLayoutBottomSheetBinding.getRoot().getVisibility() != View.VISIBLE)
                mLayoutBottomSheetBinding.getRoot().setVisibility(View.VISIBLE);
            if(mScreenSheetBinding.getRoot().getVisibility() != View.GONE)
                mScreenSheetBinding.getRoot().setVisibility(View.GONE);
            if(mBottomsheetBinding.getRoot().getVisibility() != View.GONE)
                mBottomsheetBinding.getRoot().setVisibility(View.GONE);
           // waiting for the threads to finish processing
            while(!isProcessDone)
            {}
            state = 100;
            stop = false;
            if(!hasBeenRunning)
                startClipMode();
        });
        mSelectionBarBinding.ap4Btn.setOnClickListener(v -> {
            stop = true;
            current_view = 0;
            SegmentRunner.close();
            SegmentRunner =
                    TensorFlowQUSRunnerSegment.create(
                            QEL, //TODO: WILL THIS WORK???
                            getAssets(),
                            SEGNET_FILENAMES[current_view],
                            SEGNET_MAGNITUDE,
                            SEGNET_INPUT_NAME,
                            SEGNET_OUTPUT_NAME,
                            LANDMARK_OUTPUT_NAME,
                            SEGNET_INPUT_DIMS,
                            SEGNET_OUTPUT_DIMS);
            //Needs proper screen cleanup
            if(mSelectionBarBinding.getRoot().getVisibility() == View.VISIBLE)
                mSelectionBarBinding.getRoot().setVisibility(View.GONE);
            if(mLayoutBottomSheetBinding.getRoot().getVisibility() == View.VISIBLE)
                mLayoutBottomSheetBinding.getRoot().setVisibility(View.GONE);
            if(mScreenSheetBinding.getRoot().getVisibility() != View.VISIBLE)
                mScreenSheetBinding.getRoot().setVisibility(View.VISIBLE);
            if(mBottomsheetBinding.getRoot().getVisibility() != View.VISIBLE)
                mBottomsheetBinding.getRoot().setVisibility(View.VISIBLE);
           // waiting for the threads to finish processing
            while(!isProcessDone)
            {}
            state = 10;
            stop = false;
            if(!hasBeenRunning)
                startClipMode();
            else
                finishClipMode(ClipRegioBubble);
        });

        mSelectionBarBinding.ap2Btn.setOnClickListener(v ->
        {
            stop = true;
            current_view = 1;
            SegmentRunner =
                    TensorFlowQUSRunnerSegment.create(
                            QEL, //TODO: WILL THIS WORK???
                            getAssets(),
                            SEGNET_FILENAMES[current_view],
                            SEGNET_MAGNITUDE,
                            SEGNET_INPUT_NAME,
                            SEGNET_OUTPUT_NAME,
                            LANDMARK_OUTPUT_NAME,
                            SEGNET_INPUT_DIMS,
                            SEGNET_OUTPUT_DIMS);
            //Needs proper screen cleanup
            if(mSelectionBarBinding.getRoot().getVisibility() == View.VISIBLE)
                mLayoutBottomSheetBinding.getRoot().setVisibility(View.GONE);
            if(mLayoutBottomSheetBinding.getRoot().getVisibility() == View.VISIBLE)
                mLayoutBottomSheetBinding.getRoot().setVisibility(View.GONE);
            if(mScreenSheetBinding.getRoot().getVisibility() != View.VISIBLE)
                mScreenSheetBinding.getRoot().setVisibility(View.VISIBLE);
            if(mBottomsheetBinding.getRoot().getVisibility() != View.VISIBLE)
                mBottomsheetBinding.getRoot().setVisibility(View.VISIBLE);
            // waiting for the threads to finish processing
            while(!isProcessDone)
            {}
            state = 1;
            stop = false;
            if(!hasBeenRunning)
                startClipMode();
    });
}

    public WindowManager getWindowManager() {
        if (mWindowManager == null) {
            mWindowManager = (WindowManager) getSystemService(WINDOW_SERVICE);
        }
        return mWindowManager;
    }

    //==================================================================== This is the function to decide whether trash layout disappears or not==========================================================================
// It also allows you to stop the Service
    public void checkInCloseRegion(float x, float y) {
        if (closeRegion == null) {
            int[] location = new int[2];
            View v = mTrashLayoutBinding.getRoot();
            v.getLocationOnScreen(location);
            closeRegion = new int[]{location[0], location[1],
                    location[0] + v.getWidth()+200 , location[1] + v.getHeight()+200};
        }
        if (Float.compare(x, closeRegion[0]) >= 0 &&
                Float.compare(y, closeRegion[1]) >= 0 &&
                Float.compare(x, closeRegion[2]) <= 0 &&
                Float.compare(3, closeRegion[3]) <= 0) {
            stop = true;
            finalRelease();
            if(QUSRunner != null)
                QUSRunner.close();
            if(SegmentRunner != null)
                SegmentRunner.close();
            stopSelf();
        } else {
            mTrashLayoutBinding.getRoot().setVisibility(View.GONE);
        }
    }
    //============================================================================ Makes the trash layout show up ======================================================================================================
    public void updateViewLayout(View view, WindowManager.LayoutParams params) {
        mTrashLayoutBinding.getRoot().setVisibility(View.VISIBLE);
        getWindowManager().updateViewLayout(view, params);
    }
    //==================================================================== This is the function to decide whether trash layout disappears or not==========================================================================
    public void startClipMode() {
        mSelectionBarBinding.getRoot().setVisibility(View.GONE);
        stop = true;
        mTrashLayoutBinding.getRoot().setVisibility(View.GONE);
        isClipMode = true;
        if (mClipLayoutBinding == null) {
            LayoutInflater layoutInflater = LayoutInflater.from(this);
            mClipLayoutBinding = ClipLayoutBinding.inflate(layoutInflater);
 //         mClipLayoutBinding.setHandler(new ClipHandler(this));
        }
        WindowManager.LayoutParams mClipLayoutParams = buildLayoutParamsForClip();
        ((ClipView) mClipLayoutBinding.getRoot()).updateRegion(0, 0, 0, 0);
        mClipLayoutBinding.setHandler(new ClipHandler(this));
        mBubbleLayoutBinding.getRoot().setVisibility(View.GONE);
        //mBubbleLayoutBinding.bubble1.setBackground(getDrawable(R.drawable.stop_recording));
        //This is so that startClipMode does not throw an exception for the add.view method next time
        if(!hasBeenRunning) {
            getWindowManager().addView(mClipLayoutBinding.getRoot(), mClipLayoutParams);
        }
        else
            mClipLayoutBinding.getRoot().setVisibility(View.VISIBLE);
        hasBeenRunning = true;
        Toast.makeText(this, "Choose the frame", Toast.LENGTH_SHORT).show();
    }

    public void finishClipMode(int[] clipRegion) {
        mBubbleLayoutBinding.getRoot().setVisibility(View.VISIBLE);
        isClipMode = false;
        stop = false;
        ClipRegioBubble = clipRegion;

        // Init output resize transform
        // The method needs clip region data so it needs to be called here

        if (clipRegion[2] < 50 || clipRegion[3] < 50) {
            Toast.makeText(this, "Region is too small. Try Again", Toast.LENGTH_SHORT).show();
            mClipLayoutBinding.getRoot().setVisibility(View.GONE);
            mBubbleLayoutBinding.getRoot().setVisibility(View.GONE);
            mScreenSheetBinding.getRoot().setVisibility(View.GONE);
            mBottomsheetBinding.getRoot().setVisibility(View.GONE);
            mLayoutBottomSheetBinding.getRoot().setVisibility(View.GONE);
            return;
        } else {
            if(state != 100)
            {
                outputToResizeTransform = ImageUtils.getTransformationMatrix(
                        SEGNET_INPUT_WIDTH, SEGNET_INPUT_HEIGHT,
                        ClipRegioBubble[2], ClipRegioBubble[3],
                        0, false);
                if (mScreenSheetBindingParams == null) {
                    mScreenSheetBindingParams = layoutBuilder.buildLayoutParamsForSheet(clipRegion[0],clipRegion[1], clipRegion);
                    getWindowManager().addView(mScreenSheetBinding.getRoot(), mScreenSheetBindingParams);
                }
                iV = mScreenSheetBinding.ImageView;
                mBottomsheetBinding.getRoot().setVisibility(View.VISIBLE);
            } else {
                mLayoutBottomSheetBinding.getRoot().setVisibility(View.VISIBLE);
            }
            screenshot(clipRegion);
            mClipLayoutBinding.getRoot().setVisibility(View.GONE);
        }
        mBubbleLayoutBinding.bubble1.setBackground(getDrawable(R.drawable.video_camera));
    }
    //TODO remove this useless method
    public void screenshot(int[] clipRegion){
        mLayoutBottomSheetBinding.setResult(display_results);
        shotScreen(clipRegion);
    }
    @SuppressLint("CheckResult")
    private void shotScreen(int[] clipRegion) {
        getScreenShot(clipRegion);
    }

// ============================================================================================ Cleaning Up ==============================================================================================

    private void finalRelease() {
        if (virtualDisplay != null) {
            virtualDisplay.release();
            virtualDisplay = null;
        }
        if (imageReader != null) {
            imageReader.close();
            imageReader = null;
        }
    }

    @Override
    public void onConfigurationChanged(Configuration newConfig) {
        if (isClipMode) {
            isClipMode = false;
            getWindowManager().removeView(mClipLayoutBinding.getRoot());
            Toast.makeText(this,"Configuration changed, stop clip mode.",
                    Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    public void onDestroy() {
        finalRelease();
        if (mWindowManager != null) {
            if (mBubbleLayoutBinding != null) {
                mWindowManager.removeView(mBubbleLayoutBinding.getRoot());
            }
            if (mTrashLayoutBinding != null) {
                mWindowManager.removeView(mTrashLayoutBinding.getRoot());
            }
            if (mClipLayoutBinding != null) {
                mWindowManager.removeView(mClipLayoutBinding.getRoot());
            }
            if (mBottomsheetBinding != null) {
                mWindowManager.removeView(mBottomsheetBinding.getRoot());
            }
            if (mLayoutBottomSheetBinding != null) {
                mWindowManager.removeView(mLayoutBottomSheetBinding.getRoot());
            }
            mWindowManager = null;
        }
        if (sMediaProjection != null) {
            sMediaProjection.stop();
            sMediaProjection = null;
        }
        if (mScreenSheetBinding != null) {
            mScreenSheetBinding.getRoot().setVisibility(View.GONE);
        }
        super.onDestroy();
    }
    protected WindowManager.LayoutParams buildLayoutParamsForClip() {
        WindowManager.LayoutParams params;
        if (Build.VERSION.SDK_INT <= 22) {
            //noinspection deprecation
            params = new WindowManager.LayoutParams(
                    WindowManager.LayoutParams.MATCH_PARENT,
                    WindowManager.LayoutParams.MATCH_PARENT,
                    WindowManager.LayoutParams.TYPE_TOAST,
                    WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE |
                            WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN |
                            WindowManager.LayoutParams.FLAG_TRANSLUCENT_NAVIGATION,
                    PixelFormat.TRANSPARENT);
        } else {
            Display display = getWindowManager().getDefaultDisplay();
            /*
            The real size may be smaller than the physical size of the screen
            when the window manager is emulating a smaller display (using adb shell wm size).
             */
            Point sizeReal = new Point();
            display.getRealSize(sizeReal);
            /*
            If requested from activity
            (either using getWindowManager() or (WindowManager) getSystemService(Context.WINDOW_SERVICE))
            resulting size will correspond to current app window size.
            In this case it can be smaller than physical size in multi-window mode.
             */
            Point size = new Point();
            display.getSize(size);
            int screenWidth, screenHeight, diff;
            if (size.x == sizeReal.x) {
                diff = sizeReal.y - size.y;
            } else {
                diff = sizeReal.x - size.x;
            }
            screenWidth = sizeReal.x + diff;
            screenHeight = sizeReal.y + diff;

            Log.d("kanna", "get screen " + screenWidth + " " + screenHeight
                    + " " + sizeReal.x + " " + size.x
                    + " " + sizeReal.y + " " + size.y);
            if (Build.VERSION.SDK_INT >= 26) {
                params = new WindowManager.LayoutParams(
                        screenWidth,
                        screenHeight,
                        WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                        WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE |
                                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                        PixelFormat.TRANSPARENT);
            } else {
                //noinspection deprecation
                params = new WindowManager.LayoutParams(
                        screenWidth,
                        screenHeight,
                        WindowManager.LayoutParams.TYPE_PHONE,
                        WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE |
                                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                        PixelFormat.TRANSPARENT);
            }
        }
        return params;
    }

    //https://stackoverflow.com/questions/14341041/how-to-get-real-screen-height-and-width
    private void getScreenShot(int[] clipRegion) {

        Log.i("Nima", "getScreenShot: run");

        if (imageReader == null) {
            final Point screenSize = new Point();
            final DisplayMetrics metrics = getResources().getDisplayMetrics();
            Display display = getWindowManager().getDefaultDisplay();
            display.getRealSize(screenSize);
            imageReader = ImageReader.newInstance(screenSize.x, screenSize.y,
                    PixelFormat.RGBA_8888, 3);
            virtualDisplay = sMediaProjection.createVirtualDisplay("cap", screenSize.x, screenSize.y,
                    metrics.densityDpi, DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                    imageReader.getSurface(), null, null);
            ImageReader.OnImageAvailableListener mImageListener =
                    new ImageReader.OnImageAvailableListener() {
                        Image image;
                        @Override
                        public void onImageAvailable(ImageReader reader) {
                            try {
                                image = imageReader.acquireLatestImage();
                                if (image == null) {
                                    Log.d("Nima", "No image => Freak out");
                                }
                                if (state == 100) {
//                                    finalTime = System.currentTimeMillis() -initialTime;
//                                    initialTime = System.currentTimeMillis();
//                                    Log.d("Nima", "onImageAvailable: time between frames = " + finalTime);
                                    count++;
                                    bitmapCut = createBitmap(image, clipRegion);
                                    Log.e("nima", "Count:" + count);
                                    // The GLOBAL.stop is checked there to avoid the error when terminating the application
                                    if (!stop && isProcessDone) {
                                        processImage(bitmapCut);
                                    }
                                    displayResults(filt_mean, filt_std);
                                }
                                else
                                {
//                                  getWindowManager().updateViewLayout(iV, mScreenSheetBindingParams);
                                    //Needs to be done here to avoid errors
                                    resizedOutputBitmap = Bitmap.createBitmap(ClipRegioBubble[2], ClipRegioBubble[3], Bitmap.Config.ARGB_8888);
                                    tV.setText("EF: "+ EF_string );
//                                finalTime = System.currentTimeMillis() -initialTime;
//                                initialTime = System.currentTimeMillis();
//                                        Log.d("Nima", "onImageAvailable: time between frames = " + finalTime);

                                    Image.Plane[] planes = image.getPlanes();
                                    ByteBuffer buffer = planes[0].getBuffer();
                                    int pixelStride = planes[0].getPixelStride();
                                    int rowStride = planes[0].getRowStride();
                                    int rowPadding = rowStride - pixelStride * image.getWidth();
                                    // create bitmap
                                    bitmap = Bitmap.createBitmap(image.getWidth() + rowPadding / pixelStride,
                                            image.getHeight(), Bitmap.Config.ARGB_8888);
                                    bitmap.copyPixelsFromBuffer(buffer);
                                    bitmapCut = Bitmap.createBitmap(bitmap,
                                            clipRegion[0], clipRegion[1], clipRegion[2], clipRegion[3]);
//                                            Log.e("nima", "Count:" + count);
                                    count++;
                                    // The GLOBAL.stop is checked there to avoid the error when terminating the application

                                    if (!stop && isProcessDone) {
                                        processImage(bitmapCut, bitmap, frame_counter);
                                        frame_counter++;
                                        //TODO: Check what you need to do with the frame counter once it overflows
                                        if (frame_counter >= SEGMENT_RECORD_LENGTH)
                                            frame_counter = 0;
                                        final Canvas canvas = new Canvas(resizedOutputBitmap);
                                        Bitmap finalBitmap = detectEdges(outputBitmap);
                                        canvas.drawBitmap(finalBitmap, outputToResizeTransform, null);
                                        iV.setImageBitmap(finalBitmap);
                                        displaySegment();
                                    }
//                                    long downTime = SystemClock.uptimeMillis();
//                                    long eventTime = SystemClock.uptimeMillis() + 100;
//                                    float x = 20.0f;
//                                    float y = 20.0f;
//// List of meta states found here: developer.android.com/reference/android/view/KeyEvent.html#getMetaState()
//                                    int metaState = 0;
//                                    MotionEvent motionEvent = MotionEvent.obtain(
//                                            downTime,
//                                            eventTime,
//                                            MotionEvent.ACTION_DOWN,
//                                            x,
//                                            y,
//                                            metaState
//                                    );
// Dispatch touch event to view
//                                    mScreenSheetBinding.getRoot().dispatchTouchEvent(motionEvent);
                                }
                                image.close();
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        }
                    };
            imageReader.setOnImageAvailableListener(mImageListener, null);
        }
    }
    private Bitmap createBitmap(Image image, int[] clipRegion) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer buffer = planes[0].getBuffer();
        int pixelStride = planes[0].getPixelStride();
        int rowStride = planes[0].getRowStride();
        int rowPadding = rowStride - pixelStride * image.getWidth();
        // create bitmap
        bitmap = Bitmap.createBitmap(image.getWidth() + rowPadding / pixelStride,
                image.getHeight(), Bitmap.Config.ARGB_8888);
        bitmap.copyPixelsFromBuffer(buffer);
        bitmapCut = Bitmap.createBitmap(bitmap,
                clipRegion[0], clipRegion[1], clipRegion[2], clipRegion[3]);
        buffer.rewind();
        bitmap.recycle();
        return bitmapCut;
    }
    protected synchronized void processImage(Bitmap bitmap) {
        runInBackground(
                () -> {
                    if (QUSRunner != null) {
                        // get proper transformation matrix
                        Matrix prev2cnn = ImageUtils.getTransformationMatrix(bitmap.getWidth(), bitmap.getHeight(), INPUT_WIDTH, INPUT_HEIGHT, 0, false);
                        bit = Bitmap.createBitmap(INPUT_WIDTH, INPUT_HEIGHT, Bitmap.Config.ARGB_8888); // creates a new bitmap to draw the scaled bitmapCut onto.
                        canvas.setBitmap(bit); // Sets the bitmap to canvas to draw onto
                        canvas.drawBitmap(bitmap, prev2cnn, null); // Draws the scaled version of bitmap onto bit
// ======================================================================================================================= Run network ========================================================================================================
                        if (QUSRunner == null) {
                            Log.e("Nima", "QUALITY RUNNER DELETED!");
                            return;
                        }
                        double initial = System.currentTimeMillis();
                        QUSRunner.scoreImage(bit);
                        isProcessDone = true;
                        double processTime = System.currentTimeMillis() - initial;
                        Log.i("Nima", "ScoreTime = " + processTime);
// ===================================================================================================================== Sets the results ===================================================================================================
                    }
                });
    }
    private void displayResults (float filt_mean, float filt_std){
        synchronized (results_mtx) {
            if (res_mean_vec.size() >= FILTER_LENGTH) {
                res_mean_vec.remove(0);
                res_std_vec.remove(0);
            }
            res_mean_vec.addElement(lastResults[0]);
            res_std_vec.addElement(lastResults[1]);

//                    float filt_mean = 0, filt_std = 0;
            //Log.i(TAG,"res size = "+results_vector.size());
            for (int i = 0; i < res_mean_vec.size(); i++) {
                filt_mean = filt_mean + (float) res_mean_vec.elementAt(i);
                filt_std = filt_std + (float) res_std_vec.elementAt(i);
            }
            filt_mean = filt_mean / FILTER_LENGTH;
            filt_std = filt_std / FILTER_LENGTH;
        }
        progressBar.setUncertainProgress(filt_mean, filt_std);
    }
    protected void processImage(Bitmap bitmapCut, Bitmap bitmap, int frame_counter) {
        EFCalculated = false;
        // get proper transformation matrix
        Matrix prev2seg = ImageUtils.getTransformationMatrix(bitmapCut.getWidth(), bitmapCut.getHeight(), SEGNET_INPUT_WIDTH, SEGNET_INPUT_HEIGHT, 0, false);
        bit = Bitmap.createBitmap(SEGNET_INPUT_WIDTH, SEGNET_INPUT_HEIGHT, Bitmap.Config.ARGB_8888); // creates a new bitmap to draw the scaled bitmapCut onto.
        canvas.setBitmap(bit); // Sets the bitmap to canvas to draw onto
        canvas.drawBitmap(bitmapCut, prev2seg, null); // Draws the scaled version of bitmap onto bit
// ======================================================================================================================= Run network ========================================================================================================
//            runInBackground(() -> {
        if (SegmentRunner != null) {
            // Add a frame counter to count how many frames were segmented?????
            SegmentRunner.scoreImage(bit, bitmapCut, frame_counter);

            isProcessDone = true;
            //                   ++;
//                        double processTime = System.currentTimeMillis() - initial;
//                        Log.i("Nima", "ScoreTime = " + processTime);
// ===================================================================================================================== Sets the results ===================================================================================================
        }
//            });
    }
    protected void runInBackground(final Runnable r) {
        if (handler != null) {                                          // handler.post, posts a message to the handler
            handler.post(r);                                            // .post is used to when you want to run some unknown code on UI thread
        }
    }

    public void trashLayoutRemove()
    {
        mTrashLayoutBinding.getRoot().setVisibility(View.GONE);
    }

    public void runEFCalculator() {
        Log.i(TAG,"buffer filled, running EF calculation");
        //synchronized (FSM_sync){currentState = STATE_PREVIEW;}

        // Invalidate all recorded segmentations
        for (int i = 0; i < SEGMENT_RECORD_LENGTH; i++)
            valid_segment_frames[i] = false;

        // Run LV volume estimation and calc EF
        new Thread(() -> {
            mEFCalculator.setRecordedData(recorded_segment_data, recorded_landmark_data);
            mEFCalculator.setLVAreas(current_view);
            mEFCalculator.updateResults();
        }).start();

        // TODO: remove this for realtime
        EFCalculated = true;
    }

    public synchronized void updateSegmentEvent(byte[] output_data, byte[] landmark_data, Bitmap full_res, int fc) {

        int r, g;
        for (int i = 0; i < SEGNET_INPUT_WIDTH * SEGNET_INPUT_HEIGHT; i++) {
            // Construct bitmap
            //32 bits = 24A - 16R - 8G - B;
            //r = (int) landmark_data[i] * 255;
            g = (int) output_data[i] * 255;
            output_pixels[i] = (g != 0) ?
                    (0x66 << 24)  | (g << 8) :
                    0x00000000;
        }
        outputBitmap.setPixels(output_pixels, 0, SEGNET_INPUT_WIDTH, 0, 0, SEGNET_INPUT_WIDTH, SEGNET_INPUT_HEIGHT);

        System.arraycopy(output_data, 0,
                recorded_segment_data, fc * SEGNET_INPUT_WIDTH * SEGNET_INPUT_HEIGHT,
                SEGNET_INPUT_WIDTH * SEGNET_INPUT_HEIGHT);
        System.arraycopy(landmark_data, 0,
                recorded_landmark_data, fc * SEGNET_INPUT_WIDTH * SEGNET_INPUT_HEIGHT,
                SEGNET_INPUT_WIDTH * SEGNET_INPUT_HEIGHT);

        // Validate this frame
        valid_segment_frames[fc] = true;

        // 4. If all segments are valid, finish!
        for (int i = 0; i < SEGMENT_RECORD_LENGTH; i++) {
            if (!valid_segment_frames[i]) return;
        }
//        runInBackground(new Runnable() {
//            @Override
//            public void run() {
        if (!EFCalculated) runEFCalculator();
//            }
//        });
    }

    public synchronized void updateEFOutputEvent(float ESVol, float EDVol, float EF, boolean bi){
        DecimalFormat nf = new DecimalFormat("#0.0");
        final String str = (ESVol == -1)? "N/A" : nf.format(Math.round(EF * 100.0f))+"%";
//        tV.setText(str);
        EF_string = str;
        Log.d(TAG, "updateEFOutputEvent: str is = " + str);
    }
//TODO remember to fix the status bar

    //  public int getStatusBarHeight() {
//        int height = 0;
//        int resourceId = getResources().getIdentifier("status_bar_height", "dimen", "android");
//        if (resourceId > 0) {
//            height = getResources().getDimensionPixelSize(resourceId);
//        }
//        return height;
//    }
    private Bitmap detectEdges(Bitmap bitmap) {
        Mat rgba = new Mat();
        Utils.bitmapToMat(bitmap, rgba);
        Mat edges = new Mat(rgba.size(), CvType.CV_8UC1);
        Bitmap resultBitmap = Bitmap.createBitmap(edges.cols(), edges.rows(), Bitmap.Config.ARGB_8888);
        int[] pixels = new int[resultBitmap.getHeight()*resultBitmap.getWidth()];
        Imgproc.cvtColor(rgba, edges, Imgproc.COLOR_RGB2GRAY, 4);
        Imgproc.Canny(edges, edges, 80, 100);

        Utils.matToBitmap(edges, resultBitmap);
        resultBitmap.getPixels(pixels, 0, resultBitmap.getWidth(), 0, 0, resultBitmap.getWidth(), resultBitmap.getHeight());

        //Post processing to make the black and white output, transparent and green
        for (int i=0; i < resultBitmap.getWidth()*resultBitmap.getHeight(); i++)
        {
            if (pixels[i]== Color.BLACK)
                pixels[i] = 0x00000000;
            else if (pixels[i] == Color.WHITE)
                pixels[i] = 0x8000FF00;
        }
        resultBitmap.setPixels(pixels, 0, resultBitmap.getWidth(), 0, 0, resultBitmap.getWidth(), resultBitmap.getHeight());
        return resultBitmap;
    }
    private boolean displaySegment()
    {
        if(displaySegment)
            mScreenSheetBinding.getRoot().setVisibility(View.VISIBLE);
        else
            mScreenSheetBinding.getRoot().setVisibility(View.GONE);
        return true;
    }
    //==========================================================================================SeekBar Listener==================================================================================

    public void addListenerOnDepthSeekBar() {
        mSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                depth = progress/10.0f;
                DecimalFormat nf = new DecimalFormat("#0.0");
                Log.i(TAG,"New progress = "+progress+" \t depth = "+depth);
                depth_textview.setText(nf.format(depth)+"cm");
                mEFCalculator.setDepth(depth);
            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });
    }
    public synchronized void updateResultEvent(float[] view_probs, float[] qual_results) {
        // Update EMA
        int pv = UNINIT_IDX;
        float max_prob = 0;
        for (int i = 0; i < NUM_VIEW_CLASSES; i++) {
            view_ema_arr[i] = alpha * view_probs[i] + (1 - alpha) * view_ema_arr[i];
            if (view_ema_arr[i] > max_prob) {
                max_prob = view_ema_arr[i];
                pv = i;
            }
            System.arraycopy(qual_results,0,lastResults,0,2);
            //Log.i("Nima","View = "+VIEW_NAMES[pv]+" with a prob of = "+max_prob);
            display_results.setRes1(VIEW_NAMES[pv]);
        }
    }
}

