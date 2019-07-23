package com.segmentation.nima;

import android.annotation.SuppressLint;
import android.app.Service;
import android.content.Intent;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.PixelFormat;
import android.graphics.Point;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
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
import android.widget.Toast;

import com.segmentation.nima.databinding.BubbleLayoutBinding;
import com.segmentation.nima.databinding.ClipLayoutBinding;
//import com.segmentation.nima.databinding.LayoutBottomSheetBinding;
import com.segmentation.nima.databinding.TrashLayoutBinding;
import com.segmentation.nima.env.ImageUtils;

import java.nio.ByteBuffer;
import java.text.DecimalFormat;

import static com.segmentation.nima.MainActivity.sMediaProjection;

public class BubbleService extends Service
        implements QUSEventListener{

        private static WindowManager mWindowManager;
        private BubbleLayoutBinding mBubbleLayoutBinding;
        private WindowManager.LayoutParams mBubbleLayoutParams;
        private TrashLayoutBinding mTrashLayoutBinding;
        private WindowManager.LayoutParams mTrashLayoutParams;
        private ClipLayoutBinding mClipLayoutBinding;
        private int[] closeRegion = null;//left, top, right, bottom
        private boolean isClipMode;
        private ImageReader imageReader;
        private VirtualDisplay virtualDisplay;
//        private LayoutBottomSheetBinding mLayoutBottomSheetBinding;
        private WindowManager.LayoutParams mLayoutBottomSheetParams;
        private Results display_results = new Results();
        private Handler handler;
        private HandlerThread handlerThread;
        private Canvas canvas = new Canvas();
        private Bitmap bitmapCut;
        private Bitmap bitmap;
        private Bitmap bit;
        //====================================================================================== QUS  Variables ===================================================================================
        // ---------------------------------- Network Constants ----------------------------------------
        // Shared Net Constants
        public static final String[] VIEW_NAMES = {"AP2", "AP3", "AP4", "AP5", "PLAX",
                "RVIF", "SUBC4", "SUBC5", "SUBCIVC", "PSAXA",
                "PSAXM", "PSAXPM", "PSAXAPIX", "SUPRA", "OTHER",
                "UNINIT"}; // always add unint at the ned but don't include it in the length
        public static final int AP2_IDX = 0;
        public static final int AP4_IDX = 2;
        public static final int OTHER_IDX = 14;

        // Quality Net Constants
        private static final String CNN_FILENAME = "CNN";
        private static final String RNN_FILENAME = "RNN";
        private static final String CNN_INPUT_NAME = "input";
        private static final String CNN_OUTPUT_NAME = "output_node0";
        private static final String RNN_INPUT_NAME = "RNN_input";
        private static final String RNN_QUALITY_NAME = "output_node0";
        private static final String RNN_VIEW_NAME = "output_node1";
        private static final int NUM_QUAL_CLASSES = 1;
        public static final int NUM_VIEW_CLASSES = 15; // = length(VIEW_NAMES) - 1
        private static final int CNN_INPUT_WIDTH = 120;
        private static final int CNN_INPUT_HEIGHT = 120;
        private static final int RNN_INPUT_FRAMES = 10;
        private static final long[] CNN_INPUT_DIMS = {CNN_INPUT_WIDTH, CNN_INPUT_HEIGHT};
        private static final long[] RNN_INPUT_DIMS = {RNN_INPUT_FRAMES, 14, 14, 34};

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
        private final int ON_COLOUR = 0xFFD0D0D0;
        private final int OFF_COLOUR = 0xFF4A4A4A;
        private final int GREEN_COLOUR = 0xFF00DD00;
        private static final int FRAME_RATE = 10;
        private static final int PREVIEW_WIDTH = 640;
        private static final int PREVIEW_HEIGHT = 640;
        private static final String[] CINE_NAMES = {"AP4_mo2","AP2_mo2"};
        public static final int AP4_VIEW_IDX = 0;
        public static final int AP2_VIEW_IDX = 1;
        private static final int[] CIRCUIT = {AP4_VIEW_IDX, AP2_VIEW_IDX};

        // Quality Constants
        private static final int FILTER_LENGTH = 20;
        private static final float ALPHA = 0.1f;

        // Segment Constants
        private static final int SEGMENT_RECORD_LENGTH = 74; // NOTE: THIS MUST BE <= # of .bmps
        // Max heart rate before aliasing = 90 bmp?
        // Min heart rate capturable = 45 bpm
        private static final int INITIAL_DEPTH_SETTING = 100; // [mm]
        public static final float THRESHOLD = 0.5f;

        // -------------------------------------- App Variables ----------------------------------------
        // Shared Variables
        private Toast mToast;
        private int current_view = 0; // 0 -> AP4, 1 -> AP2
        private static float depth = 0.0f;

        // Quality Variables
        private Classifier QualityRunner;
        private static float[] view_ema_arr = new float[NUM_VIEW_CLASSES];
        private static float lastResult;
        private static int lastReqView;
        private static int lastPredView;

        // Segment Variables
        private Classifier SegmentRunner;
        private EFCalculator mEFCalculator;
        private Bitmap outputBitmap, resizedOutputBitmap;           // raw and resized output colormaps
        private int[] output_pixels;                                // pixel vector used to create output Bitmap
        private byte[] recorded_segment_data;                       // recorded 128x128 segmentations
        private byte[] recorded_landmark_data;                     // recorded 128x128 landmark
        private android.graphics.Matrix outputToResizeTransform;    // resizes the 128x128 output map to the preview dims
        private static boolean[] valid_segment_frames;
        private boolean EFCalculated;

        // FSM variables
        public static final int STATE_DISCONNECTED = -1;
        public static final int STATE_PREVIEW = 0;
        public static final int STATE_QUALITY = 1;
        public static final int STATE_SEGMENT = 2;
        private int currentState = STATE_DISCONNECTED;
        private final Object FSM_sync = new Object();
        private QUSEventListener QEL;

// ======================================================================================= Timing Analysis Vars ==================================================================================

        private static double initialTime = 0;
        private static double finalTime = 0;
        public static boolean stop = true;
        public static boolean hasBeenRunning = false;
        public static int count = 0;
        public static boolean isProcessDone = true;

        public BubbleService() {
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
            // TODO: Should I change this??
            QEL = this;
            recorded_segment_data = new byte[SEGNET_INPUT_WIDTH * SEGNET_INPUT_HEIGHT * SEGMENT_RECORD_LENGTH];   // 128*128*100*4 bytes = 6.25 MB
            recorded_landmark_data = new byte[SEGNET_INPUT_WIDTH * SEGNET_INPUT_HEIGHT * SEGMENT_RECORD_LENGTH]; // 128*128*100*4 bytes = 6.25 MB
            valid_segment_frames = new boolean[SEGMENT_RECORD_LENGTH];
            mEFCalculator = new EFCalculator(this, SEGNET_INPUT_WIDTH, SEGNET_INPUT_HEIGHT, SEGMENT_RECORD_LENGTH);

            // Init output resize transform
            outputToResizeTransform = ImageUtils.getTransformationMatrix(
                    SEGNET_INPUT_WIDTH, SEGNET_INPUT_HEIGHT,
                    PREVIEW_WIDTH, PREVIEW_HEIGHT,
                    0, false);


            // Create output bitmaps
            outputBitmap = Bitmap.createBitmap(SEGNET_INPUT_WIDTH, SEGNET_INPUT_HEIGHT, Bitmap.Config.ARGB_8888);
            output_pixels = new int[SEGNET_INPUT_WIDTH * SEGNET_INPUT_HEIGHT];
            resizedOutputBitmap = Bitmap.createBitmap(PREVIEW_WIDTH, PREVIEW_HEIGHT, Bitmap.Config.ARGB_8888);
            initial();
// ======================================================================= From TF Lite app to enable background image processing ========================================================
            handlerThread = new HandlerThread("Inference");
            handlerThread.start();
//
            handler = new Handler(handlerThread.getLooper());
//        handler = new Handler();
// =======================================================================================================================================================================================
            return super.onStartCommand(intent, flags, startId);
        }

        private void initial() {
//      Log.d("kanna", "initial");
//==================================================================== Inflate all the views and set their proper parameters and Handlers===============================================================================
            LayoutInflater layoutInflater = LayoutInflater.from(this);
            mTrashLayoutBinding = TrashLayoutBinding.inflate(layoutInflater);
//            mLayoutBottomSheetBinding = LayoutBottomSheetBinding.inflate(layoutInflater);
            if (mTrashLayoutParams == null) {
                mTrashLayoutParams = buildLayoutParamsForBubble(0, 200);
                mTrashLayoutParams.gravity = Gravity.BOTTOM | Gravity.CENTER_HORIZONTAL;
            }
            getWindowManager().addView(mTrashLayoutBinding.getRoot(), mTrashLayoutParams);
            mTrashLayoutBinding.getRoot().setVisibility(View.GONE);
            mBubbleLayoutBinding = BubbleLayoutBinding.inflate(layoutInflater);
            if (mBubbleLayoutParams == null) {
                mBubbleLayoutParams = buildLayoutParamsForBubble(60, 60);
                mBubbleLayoutBinding.getRoot().setBackground(getDrawable(R.drawable.video_camera));
            }
            if (mLayoutBottomSheetParams == null) {
                mLayoutBottomSheetParams = buildLayoutParamsForBottomSheet(0, 0);
            }
            mBubbleLayoutBinding.setHandler(new BubbleHandler(this));
            getWindowManager().addView(mBubbleLayoutBinding.getRoot(), mBubbleLayoutParams);
//            getWindowManager().addView(mLayoutBottomSheetBinding.getRoot(), mLayoutBottomSheetParams);
//            mLayoutBottomSheetBinding.getRoot().setVisibility(View.GONE);
            //mLayoutBottomSheetBinding.setHandler(new BottomSheetHandler(this));
            if(SegmentRunner == null)
            {
                //initialize qus runner here
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
        }

        private WindowManager getWindowManager() {
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
                SegmentRunner.close();
                stopSelf();
            } else {
                mTrashLayoutBinding.getRoot().setVisibility(View.GONE);
            }
        }
        //============================================================================ Makes the trashlayout to show up ======================================================================================================
        public void updateViewLayout(View view, WindowManager.LayoutParams params) {
            mTrashLayoutBinding.getRoot().setVisibility(View.VISIBLE);
            getWindowManager().updateViewLayout(view, params);
        }
        //==================================================================== This is the function to decide whether trash layout disappears or not==========================================================================
        public void startClipMode() {
            stop = true;
            mTrashLayoutBinding.getRoot().setVisibility(View.GONE);
            isClipMode = true;
            if (mClipLayoutBinding == null) {
                LayoutInflater layoutInflater = LayoutInflater.from(this);
                mClipLayoutBinding = ClipLayoutBinding.inflate(layoutInflater);
                mClipLayoutBinding.setHandler(new ClipHandler(this));
            }
            WindowManager.LayoutParams mClipLayoutParams = buildLayoutParamsForClip();
            ((ClipView) mClipLayoutBinding.getRoot()).updateRegion(0, 0, 0, 0);
            //mBubbleLayoutBinding.getRoot().setVisibility(View.INVISIBLE);    //This is when you are taking the screenshot. You can set the visibility to Gone to get rid of the Bubble
            mBubbleLayoutBinding.getRoot().setBackground(getDrawable(R.drawable.stop_recording));
//        if(QUSRunner != null) {
//            QUSRunner.clearLastResult();
//            QUSRunner.close();
//        }
            //This is so that startClipMode does not throw an exception for the add.view method next time
            if(!hasBeenRunning) {
                getWindowManager().addView(mClipLayoutBinding.getRoot(), mClipLayoutParams);
            }
            else
                mClipLayoutBinding.getRoot().setVisibility(View.VISIBLE);
            hasBeenRunning = true;
            Toast.makeText(this, "Start clip mode.", Toast.LENGTH_SHORT).show();
        }

        public void finishClipMode(int[] clipRegion) {
            isClipMode = false;
            stop = false;
            // Initialize the QUS runner after the clip layout is chosen
            // This helps when you want to reselect the clip layout

            //getWindowManager().removeView(mClipLayoutBinding.getRoot());    //This is the clip region view where you choose to take the screenshot.
            // By not removing the view the box will stay on the screen indefinitely
            if (clipRegion[2] < 50 || clipRegion[3] < 50) {
                Toast.makeText(this, "Region is too small. Try Again", Toast.LENGTH_SHORT).show();
                mBubbleLayoutBinding.getRoot().setVisibility(View.VISIBLE);
                mClipLayoutBinding.getRoot().setVisibility(View.GONE);
                mBubbleLayoutBinding.getRoot().setVisibility(View.GONE);
//                mLayoutBottomSheetBinding.getRoot().setVisibility(View.GONE);
                return;
//            finalRelease();
//            GLOBAL.stop = true;
//            stopSelf();
            } else {
                screenshot(clipRegion);
                mClipLayoutBinding.getRoot().setVisibility(View.GONE);
            }
            mBubbleLayoutBinding.getRoot().setBackground(getDrawable(R.drawable.video_camera));

        }
        public void screenshot(int[] clipRegion){
//            mLayoutBottomSheetBinding.getRoot().setVisibility(View.VISIBLE);
//        mLayoutBottomSheetBinding.setHandler(new BottomSheetHandler(this));
//            mLayoutBottomSheetBinding.setResult(display_results);
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
            if(stop)
            {
//                mLayoutBottomSheetBinding.getRoot().setVisibility(View.GONE);
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
//                if (mLayoutBottomSheetBinding != null) {
//                    mWindowManager.removeView(mLayoutBottomSheetBinding.getRoot());
//                }
                mWindowManager = null;
            }
            if (sMediaProjection != null) {
                sMediaProjection.stop();
                sMediaProjection = null;
            }
            super.onDestroy();
        }

        private WindowManager.LayoutParams buildLayoutParamsForBubble(int x, int y) {
            WindowManager.LayoutParams params;
            if (Build.VERSION.SDK_INT >= 26) {
                params = new WindowManager.LayoutParams(
                        WindowManager.LayoutParams.WRAP_CONTENT,
                        WindowManager.LayoutParams.WRAP_CONTENT,
                        WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                        WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
                        PixelFormat.TRANSPARENT);
            } else if(Build.VERSION.SDK_INT >= 23) {
                //noinspection deprecation
                params = new WindowManager.LayoutParams(
                        WindowManager.LayoutParams.WRAP_CONTENT,
                        WindowManager.LayoutParams.WRAP_CONTENT,
                        WindowManager.LayoutParams.TYPE_PHONE,
                        WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
                        PixelFormat.TRANSPARENT);
            } else {
                //noinspection deprecation
                params = new WindowManager.LayoutParams(
                        WindowManager.LayoutParams.WRAP_CONTENT,
                        WindowManager.LayoutParams.WRAP_CONTENT,
                        WindowManager.LayoutParams.TYPE_TOAST,
                        WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
                        PixelFormat.TRANSPARENT);
            }
            params.gravity = Gravity.TOP | Gravity.START;
            params.x = x;
            params.y = y;
            return params;
        }
        private WindowManager.LayoutParams buildLayoutParamsForBottomSheet(int x, int y) {
            WindowManager.LayoutParams params;
            if (Build.VERSION.SDK_INT >= 26) {
                params = new WindowManager.LayoutParams(
                        WindowManager.LayoutParams.MATCH_PARENT,
                        WindowManager.LayoutParams.WRAP_CONTENT,
                        WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                        WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
                        PixelFormat.TRANSPARENT);
            } else if(Build.VERSION.SDK_INT >= 23) {
                //noinspection deprecation
                params = new WindowManager.LayoutParams(
                        WindowManager.LayoutParams.MATCH_PARENT,
                        WindowManager.LayoutParams.WRAP_CONTENT,
                        WindowManager.LayoutParams.TYPE_PHONE,
                        WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
                        PixelFormat.TRANSPARENT);
            } else {
                //noinspection deprecation
                params = new WindowManager.LayoutParams(
                        WindowManager.LayoutParams.MATCH_PARENT,
                        WindowManager.LayoutParams.WRAP_CONTENT,
                        WindowManager.LayoutParams.TYPE_TOAST,
                        WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
                        PixelFormat.TRANSPARENT);
            }
            params.gravity = Gravity.BOTTOM | Gravity.START;
            params.x = x;
            params.y = y;
            return params;
        }

        private WindowManager.LayoutParams buildLayoutParamsForClip() {
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
                        PixelFormat.RGBA_8888, 2);
                virtualDisplay = sMediaProjection.createVirtualDisplay("cap", screenSize.x, screenSize.y,
                        metrics.densityDpi, DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                        imageReader.getSurface(), null, null);
                ImageReader.OnImageAvailableListener mImageListener =
                        new ImageReader.OnImageAvailableListener() {
                            Image image;

                            @Override
                            public void onImageAvailable(ImageReader reader) {
                                try {
                                    finalTime = System.currentTimeMillis() -initialTime;
                                    initialTime = System.currentTimeMillis();
                                    Log.d("Nima", "onImageAvailable: time between frames = " + finalTime);
                                    image = imageReader.acquireLatestImage();
                                    if (image == null) {
                                        Log.d("Nima", "No image => Freak out");
                                    } else {
                                        count++;
//                                        bitmapCut = createBitmap(image, clipRegion);
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
                                        Log.e("nima", "Count:" + count);
                                        // The GLOBAL.stop is checked there to avoid the error when terminating the application
                                        if (!stop && isProcessDone) {
                                            //Insert image process call here
                                            processImage(bitmapCut, bitmap);
                                        }

                                        image.close();
                                    }
                                } catch (Exception e) {
                                    e.printStackTrace();
                                }
                            }
                        };
                imageReader.setOnImageAvailableListener(mImageListener, null);
            }
        }
        protected synchronized void processImage(Bitmap bitmapCut, Bitmap bitmap) {
            runInBackground(new Runnable() {
                @Override
                public void run() {
                    if (SegmentRunner != null) {
                        // get proper transformation matrix
                        Matrix prev2seg = ImageUtils.getTransformationMatrix(bitmapCut.getWidth(), bitmapCut.getHeight(), SEGNET_INPUT_WIDTH, SEGNET_INPUT_HEIGHT, 0, false);
                        bit = Bitmap.createBitmap(SEGNET_INPUT_WIDTH, SEGNET_INPUT_HEIGHT, Bitmap.Config.ARGB_8888); // creates a new bitmap to draw the scaled bitmapCut onto.
                        canvas.setBitmap(bit); // Sets the bitmap to canvas to draw onto
                        canvas.drawBitmap(bitmapCut, prev2seg, null); // Draws the scaled version of bitmap onto bit
// ======================================================================================================================= Run network ========================================================================================================
                        if (SegmentRunner == null) {
                            Log.e("Nima", "SEGNET RUNNER DELETED!");
                            return;
                        }
                        double initial = System.currentTimeMillis();
                        // Add a frame counter to count how many frames were segmented?????
                        SegmentRunner.scoreImage(bit, bitmapCut, 0);
                        isProcessDone = true;
                        double processTime = System.currentTimeMillis() - initial;
                        Log.i("Nima", "ScoreTime = " + processTime);
// ===================================================================================================================== Sets the results ===================================================================================================
                    }
                }
            });
        }
        protected synchronized void runInBackground(final Runnable r) {
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
            new Thread(new Runnable() {
                public void run() {
                    mEFCalculator.setRecordedData(recorded_segment_data, recorded_landmark_data);
                    mEFCalculator.setLVAreas(current_view);
                    mEFCalculator.updateResults();
                }
            }).start();

            // TODO: remove this for realtime
            EFCalculated = true;
        }

    public synchronized void updateSegmentEvent(byte[] output_data, byte[] landmark_data, Bitmap full_res, int fc) {
               runInBackground(new Runnable() {
                   @Override
                   public void run() {
                       int r,g;
                       for (int i = 0; i < SEGNET_INPUT_WIDTH * SEGNET_INPUT_HEIGHT; i++) {
                           // Construct bitmap
                           //32 bits = 24A - 16R - 8G - B;
                           r = (int)landmark_data[i]*255;
                           g = (int)output_data[i]*255;
                           output_pixels[i] = (r!=0 | g!=0)?
                                   (0x66 << 24) | (r << 16) | (g << 8) :
                                   0x00000000;
                       }
                       outputBitmap.setPixels(output_pixels, 0, SEGNET_INPUT_WIDTH, 0, 0, SEGNET_INPUT_WIDTH, SEGNET_INPUT_HEIGHT);

                       // 2. Transform to preview dimensions
                       final Canvas canvas = new Canvas(resizedOutputBitmap);
                       // Must be run on main thread????
                               canvas.drawBitmap(full_res, 0, 0, null);
                               canvas.drawBitmap(outputBitmap, outputToResizeTransform, null);

                               // 3. Set imageview bitmap
                               // Commented out for now until we find a way to display it
                               // imageView.setImageBitmap(resizedOutputBitmap);
                           }
                       });

                       // If we are outside of the segment record buffer, stop here
//                       if(fc >= SEGMENT_RECORD_LENGTH) return;

                       // 3. Only record if we are in sync with bitmaps for synchronization purposes
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
                       if(!EFCalculated) runEFCalculator();
    }

    public synchronized void updateEFOutputEvent(float ESVol, float EDVol, float EF, boolean bi){
                DecimalFormat nf = new DecimalFormat("#0.0");
                final String str = (ESVol == -1)? "N/A" : nf.format(Math.round(EF * 100.0f))+"%";
            Log.d(TAG, "updateEFOutputEvent: str is = " + str);
    }




}
