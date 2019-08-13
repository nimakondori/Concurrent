package com.segmentation.nima;

import android.widget.ImageView;

import androidx.databinding.BaseObservable;
import androidx.databinding.Bindable;

public class ResultsBinding extends BaseObservable
{
    ImageView iV;
    String res1;
    GoalProgressBar prog1;

    @Bindable
    public ImageView getiV() {
        return iV;
    }

    public void setiV(ImageView iV) {
        this.iV = iV;
        notifyPropertyChanged(BR.iV);
    }
    @Bindable
    public String getRes1() {
        return res1;
    }

    public void setRes1(String res1) {
        this.res1 = "Predicted View: " + res1;
        notifyPropertyChanged(BR.res1);
    }
    @Bindable
    public GoalProgressBar getProg1() {
        return prog1;
    }

    public void setBar(GoalProgressBar bar) {
        this.prog1 = bar;
        notifyPropertyChanged(BR.prog1);
    }
}

