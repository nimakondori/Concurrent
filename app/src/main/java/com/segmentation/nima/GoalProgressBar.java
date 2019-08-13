package com.segmentation.nima;

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Bundle;
import android.os.Parcelable;
import android.util.AttributeSet;
import android.view.View;

public class GoalProgressBar extends View{

        private Paint progressPaint;
        private float primaryProgress, meanProgress, secondaryProgress;
        private int primaryColour, meanColour, secondaryColour;
        private int unfilledColour;

        private float goalIndicatorHeight;
        private float goalIndicatorThickness;
        private int barThickness;
        //private IndicatorType indicatorType;
        //private ValueAnimator barAnimator;

        //public enum IndicatorType {
        //    Line, Circle, Square
        //}

        public GoalProgressBar(Context context, AttributeSet attrs) {
            super(context, attrs);
            init(attrs);
        }

        private void init(AttributeSet attrs) {
            progressPaint = new Paint();
            progressPaint.setStyle(Paint.Style.FILL_AND_STROKE);

            TypedArray typedArray = getContext().getTheme().obtainStyledAttributes(attrs, R.styleable.GoalProgressBar, 0, 0);
            try {
                setGoalIndicatorHeight(typedArray.getDimensionPixelSize(R.styleable.GoalProgressBar_goalIndicatorHeight, 10));
                setGoalIndicatorThickness(typedArray.getDimensionPixelSize(R.styleable.GoalProgressBar_goalIndicatorThickness, 5));
                //setGoalReachedColor(typedArray.getColor(R.styleable.GoalProgressBar_goalReachedColor, Color.BLUE));
                //setGoalNotReachedColor(typedArray.getColor(R.styleable.GoalProgressBar_goalNotReachedColor, Color.BLACK));
                setUnfilledSectionColour(typedArray.getColor(R.styleable.GoalProgressBar_unfilledSectionColor, Color.RED));
                setBarThickness(typedArray.getDimensionPixelOffset(R.styleable.GoalProgressBar_barThickness, 4));

                //int index = typedArray.getInt(R.styleable.GoalProgressBar_indicatorType, 0);
                //setIndicatorType(IndicatorType.values()[index]);
            } finally {
                typedArray.recycle();
            }
        }

        @Override
        protected Parcelable onSaveInstanceState() {
            Bundle bundle = new Bundle();

            // save our added state - progress and goal
            bundle.putFloat("min", primaryProgress);
            bundle.putFloat("mean", meanProgress);
            bundle.putFloat("max", secondaryProgress);

            // save super state
            bundle.putParcelable("superState", super.onSaveInstanceState());

            return bundle;
        }

        @Override
        protected void onRestoreInstanceState(Parcelable state) {
            if (state instanceof Bundle) {
                Bundle bundle = (Bundle) state;

                // restore our added state - progress and goal
                this.primaryProgress = (bundle.getFloat("min"));
                this.meanProgress = (bundle.getFloat("mean"));
                this.secondaryProgress = (bundle.getFloat("max"));

                // restore super state
                state = bundle.getParcelable("superState");
            }

            super.onRestoreInstanceState(state);
        }

        @Override
        protected void onDraw(Canvas canvas) {
            int halfHeight = getHeight() / 2;
            int progressMinX = (int) (getWidth() * primaryProgress / 100f);
            int progressMaxX = (int) (getWidth() * secondaryProgress / 100f);

            // draw the primary portion of the bar
            progressPaint.setStrokeWidth(barThickness);
            progressPaint.setColor(primaryColour);
            canvas.drawLine(0, halfHeight, progressMinX, halfHeight, progressPaint);

            // draw the secondary portion of the bar
            progressPaint.setColor(secondaryColour);
            canvas.drawLine(progressMinX, halfHeight, progressMaxX, halfHeight, progressPaint);

            // draw the unfilled portion of the bar
            progressPaint.setColor(unfilledColour);
            canvas.drawLine(progressMaxX, halfHeight, getWidth(), halfHeight, progressPaint);

            // draw goal indicator
            int indicatorPosition = (int) (getWidth() * meanProgress / 100f);
            progressPaint.setColor(meanColour);
            progressPaint.setStrokeWidth(goalIndicatorThickness);
            canvas.drawLine(
                    indicatorPosition,
                    halfHeight - (goalIndicatorHeight / 2),
                    indicatorPosition,
                    halfHeight + (goalIndicatorHeight / 2),
                    progressPaint);
        }
        @Override
        protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
            int width = MeasureSpec.getSize(widthMeasureSpec);

            int specHeight = MeasureSpec.getSize(heightMeasureSpec);
            int height;
            switch (MeasureSpec.getMode(heightMeasureSpec)) {

                // be exactly the given specHeight
                case MeasureSpec.EXACTLY:
                    height = specHeight;
                    break;

                // be at most the given specHeight
                case MeasureSpec.AT_MOST:
                    height = (int) Math.min(goalIndicatorHeight, specHeight);
                    break;

                // be whatever size you want
                case MeasureSpec.UNSPECIFIED:
                default:
                    height = specHeight;
                    break;
            }

            // must call this, otherwise the app will crash
            setMeasuredDimension(width, height);
        }

        public void setUncertainProgress(float mean, float std){
            float min, mid, max;
            //Log.i(TAG,"mean = "+mean+"\t std = "+std);
            if(mean == -1){
                min = 0;
                mid = 0;
                max = 0;
            } else {
                min = Math.max(0,Math.min(95,100f*(mean - std)));
                mid = Math.max(3,Math.min(97,100f *mean));
                max = Math.max(6,Math.min(99,100f*(mean + std)));
            }

            int G = Math.round(0xFF * (max+min)/200);
            int R = 0xFF - G;
            int B = 0x3F;
            this.primaryProgress = min;
            this.primaryColour = (0xDD << 24) + (R << 16) + (G << 8) + B;

            this.meanProgress = mid;
            this.meanColour = (0xFF << 24) + (R << 16) + (G << 8) + B;

            this.secondaryProgress = max;
            this.secondaryColour = (0xBB << 24) + (R << 16) + (G << 8) + B;
            postInvalidate();
        }

        public void setGoalIndicatorHeight(float goalIndicatorHeight) {
            this.goalIndicatorHeight = goalIndicatorHeight;
            postInvalidate();
        }

        public void setGoalIndicatorThickness(float goalIndicatorThickness) {
            this.goalIndicatorThickness = goalIndicatorThickness;
            postInvalidate();
        }

        public void setBarThickness(int barThickness) {
            this.barThickness = barThickness;
            postInvalidate();
        }

        public void setUnfilledSectionColour(int unfilledSectionColour) {
            this.unfilledColour = unfilledSectionColour;
            postInvalidate();
        }
}
