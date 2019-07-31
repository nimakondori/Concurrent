package com.segmentation.nima;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.databinding.DataBindingUtil;
import com.segmentation.nima.databinding.ActivityMainBinding;

import com.segmentation.nima.BubbleService;

import org.opencv.android.OpenCVLoader;

public class MainActivity extends AppCompatActivity {
    public static MediaProjection sMediaProjection;
    private static final int MY_PERMISSIONS_REQUEST_READ_CONTACTS = 5566;
    private static final int REQUEST_CODE = 55566;
    private ActivityMainBinding binding;
    private MediaProjectionManager mProjectionManager;
    private Animation toBottom;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        OpenCVLoader.initDebug();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            Window w = getWindow();
            w.setFlags(WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS, WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS);
        }
        super.onCreate(savedInstanceState);

        binding = DataBindingUtil.setContentView(this, R.layout.activity_main);
        checkDrawOverlayPermission();
        checkWritePermission();

        mProjectionManager = (MediaProjectionManager) getSystemService(Context.MEDIA_PROJECTION_SERVICE);

        binding.test.setOnClickListener(view -> {
            if(!checkDrawOverlayPermission()){
                checkDrawOverlayPermission();
                return;
            }
            if(!checkWritePermission()){
                checkWritePermission();
                return;
            }
            BubbleService.stop = true;
            BubbleService.hasBeenRunning = false;
            AnimationUtils.loadAnimation(this, R.anim.to_bottom);
            startMediaProjection();
        });
    }

    private boolean checkWritePermission() {
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    MY_PERMISSIONS_REQUEST_READ_CONTACTS);
            return false;
        }
        return true;
    }

    private void startMediaProjection() {
        startActivityForResult(mProjectionManager.createScreenCaptureIntent(), REQUEST_CODE);
    }

    private boolean checkDrawOverlayPermission() {
        if (Build.VERSION.SDK_INT >= 23) {
            if (!Settings.canDrawOverlays(this)) {
                Intent intent = new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                        Uri.parse("package:" + getPackageName()));
                startActivity(intent);
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch (requestCode) {
            case MY_PERMISSIONS_REQUEST_READ_CONTACTS:
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Log.d("kanna","get write permission");
                }
                break;
        }
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_CODE) {
            sMediaProjection = mProjectionManager.getMediaProjection(resultCode, data);
            if (sMediaProjection == null) {
                Log.d("kanna", "not get permission of media projection");
                Toast.makeText(this, "Need MediaProjection", Toast.LENGTH_LONG).show();
                startMediaProjection();
            } else {
                startBubble();
            }
        }
    }

    @Override
    protected void onResume(){
//        if (sMediaProjection == null) {
        if(BubbleService.hasBeenRunning) {
            binding.test.setVisibility(View.VISIBLE);
// ============================================================== When the app resumes just stop any possible services running previously ===================================================================
// ===================================================== Since the binding button is visible, the user can click and start the service all over again =======================================================
// ===================================================================== remember it only happens if the MediaPorjection is Null ============================================================================
            Intent intent = new Intent(this, BubbleService.class);
            stopService(intent);
            BubbleService.stop = true;
        }
//      }
        super.onResume();

    }

    private void startBubble() {
        binding.test.setVisibility(View.GONE);
        Intent intent = new Intent(this, BubbleService.class);
        stopService(intent);
        startService(intent);
    }
}

