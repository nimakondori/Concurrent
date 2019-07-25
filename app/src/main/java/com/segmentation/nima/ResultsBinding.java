package com.segmentation.nima;

import android.widget.ImageView;

import androidx.databinding.BaseObservable;
import androidx.databinding.Bindable;

public class ResultsBinding extends BaseObservable
{
    ImageView iV;

    @Bindable
    public ImageView getiV() {
        return iV;
    }

    public void setiV(ImageView iV) {
        this.iV = iV;
        notifyPropertyChanged(BR.iV);
    }
}

