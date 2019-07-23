package com.segmentation.nima;

import androidx.databinding.BaseObservable;
import androidx.databinding.Bindable;


public class Results extends BaseObservable {
    String res1;

    // Making it Bindable will let the app update the UI every time it is updated
    @Bindable
    public String getRes1() {
        return res1;
    }

    public void setRes1(String res1) {
        this.res1 = "Predicted View: " + res1;
        notifyPropertyChanged(BR.res1);
    }
}
