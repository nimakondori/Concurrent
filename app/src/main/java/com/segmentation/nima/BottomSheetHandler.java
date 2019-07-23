package com.segmentation.nima;

import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;

public class BottomSheetHandler {

    private final BubbleService service;
    private int initialX;
    private int initialY;
    private float initialTouchX;
    private float initialTouchY;
    private float moveDistance;
    BottomSheetHandler(BubbleService service) {
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
                break;
            case MotionEvent.ACTION_UP:
                view.performClick();
                service.trashLayoutRemove();
                break;
            case MotionEvent.ACTION_MOVE:
                // params.x = initialX + (int) (motionEvent.getRawX() - initialTouchX);
                // no need for above statement if the textbox is already matching the parent size
                // I don't get it why but changing a + to - solves the weird Y effect of Y parameter
                // It it highly possible that the Gravity has some effect here
                // Yes it is due to the fact that the bottom sheet is gravitated at the bottom but the bubble starts at the top
                params.y = initialY - (int) (motionEvent.getRawY() - initialTouchY);
                float distance = motionEvent.getRawX() - initialTouchX
                        + motionEvent.getRawY() - initialTouchY;
                moveDistance += Math.abs(distance);
                service.updateViewLayout(view, params);
                break;

        }
        return true;
    }
}