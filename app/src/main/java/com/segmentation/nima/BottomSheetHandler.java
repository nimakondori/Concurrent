package com.segmentation.nima;

import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.SeekBar;
import android.widget.TextView;

import androidx.databinding.BaseObservable;
import androidx.databinding.Bindable;

public class BottomSheetHandler extends BaseObservable {

    private final BubbleService service;
    private int initialX;
    private int initialY;
    private float initialTouchX;
    private float initialTouchY;
    private TextView tV1;
    private SeekBar seekBar;

    BottomSheetHandler(BubbleService service) {
        this.service = service;
    }

    @Bindable
    public TextView gettV1() {
        return tV1;
    }

    public void settV1(TextView tV1) {
        this.tV1 = tV1;
        notifyPropertyChanged(BR.tV1);
    }
    @Bindable
    public SeekBar getSeekBar() {
        return seekBar;
    }

    public void setSeekBar(SeekBar seekBar) {
        this.seekBar = seekBar;
        notifyPropertyChanged(BR.seekBar);
    }

    public boolean onTouch(View view, MotionEvent motionEvent) {
        WindowManager.LayoutParams params = (WindowManager.LayoutParams) view.getLayoutParams();
        switch (motionEvent.getAction()) {
            case MotionEvent.ACTION_DOWN:
                initialX = params.x;
                initialY = params.y;
                initialTouchX = motionEvent.getRawX();
                initialTouchY = motionEvent.getRawY();
                service.displaySegment=true;

                break;
            case MotionEvent.ACTION_UP:
                view.performClick();
                service.displaySegment=false;

                break;
            case MotionEvent.ACTION_MOVE:
                // params.x = initialX + (int) (motionEvent.getRawX() - initialTouchX);
                // no need for above statement if the textbox is already matching the parent size
                // I don't get it why but changing a + to - solves the weird Y effect of Y parameter
                // It it highly possible that the Gravity has some effect here
                // Yes it is due to the fact that the bottom sheet is gravitated at the bottom but the bubble starts at the top
                params.y = initialY - (int) (motionEvent.getRawY() - initialTouchY);
                service.updateViewLayout(view, params);
                service.trashLayoutRemove();
                service.displaySegment=true;

                break;

        }
        return true;
    }


}
