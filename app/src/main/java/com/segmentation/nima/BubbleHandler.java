package com.segmentation.nima;

import android.animation.Animator;
import android.animation.ValueAnimator;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;

import androidx.databinding.Bindable;

public class BubbleHandler {
    private final BubbleService service;
    private int initialX;
    private int initialY;
    private float initialTouchX;
    private float initialTouchY;
    private float moveDistance;

    BubbleHandler(BubbleService service) {
        this.service = service;
    }
    public boolean onTouch(View view, MotionEvent motionEvent) {
        WindowManager.LayoutParams params = (WindowManager.LayoutParams) view.getLayoutParams();
        switch (motionEvent.getAction()) {
            case MotionEvent.ACTION_DOWN:
                moveDistance = 0;
                initialX = params.x;
                initialY = params.y;
                initialTouchX = motionEvent.getRawX();
                initialTouchY = motionEvent.getRawY();
                service.mSelectionBarBinding.getRoot().setVisibility(View.GONE);
                break;
            case MotionEvent.ACTION_UP:
                view.performClick();
                Log.d("Nima", "onTouch: GlobalStop " + BubbleService.stop);
                if (Float.compare(moveDistance, 100f) >= 0) {
                    service.checkInCloseRegion(motionEvent.getRawX(), motionEvent.getRawY());
                    service.updateViewLayout(view, params);
                    service.trashLayoutRemove();

                } else {
                    animate(0,0 ,params);
//                    service.startClipMode();

                }
                break;
            case MotionEvent.ACTION_MOVE:
                params.x = initialX + (int) (motionEvent.getRawX() - initialTouchX);
                params.y = initialY + (int) (motionEvent.getRawY() - initialTouchY);
                float distance = motionEvent.getRawX() - initialTouchX
                        + motionEvent.getRawY() - initialTouchY;
                moveDistance += Math.abs(distance);
                service.updateViewLayout(view, params);
                break;
        }

        return true;
    }
    public void animate(
            final float x,
            final float y,
            WindowManager.LayoutParams params) {
        final float startX = params.x;
        final float startY = params.y;
        ValueAnimator animator = ValueAnimator.ofInt(0, 5)
                .setDuration(100);
        animator.addUpdateListener(valueAnimator -> {
            try {
                float currentX = startX + ((x - startX) *
                        (Integer) valueAnimator.getAnimatedValue() / 5);
                float currentY = startY + ((y - startY) *
                        (Integer) valueAnimator.getAnimatedValue() / 5);
                    params.x = (int) currentX;
                    params.x = params.x < 0 ? 0 : params.x;
                    params.y = (int) currentY;
                    params.y = params.y < 0 ? 0 : params.y;

                service.updateViewLayout(service.mBubbleLayoutBinding.getRoot(), params);
                service.trashLayoutRemove();

            } catch (Exception exception) {
                Log.e("NKO", exception.getMessage());
            }
        });
        animator.addListener(new Animator.AnimatorListener() {
            @Override
            public void onAnimationStart(Animator animation) {
            }

            @Override
            public void onAnimationEnd(Animator animation) {
                service.mSelectionBarBinding.getRoot().setVisibility(View.VISIBLE);
            }

            @Override
            public void onAnimationCancel(Animator animation) {

            }

            @Override
            public void onAnimationRepeat(Animator animation) {

            }
        });
        animator.start();
    }
}

