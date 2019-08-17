package com.segmentation.nima;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

/**
 * Created by nathanvw on 10/4/18.
 */

public class EFCalculator {

    private static String TAG = "nvw-EF";
    private QUSEventListener QEL;
    private int width, height, length, frame_size;
    private float depth;
    private byte[] segments, landmarks;
    private int[] pixels;
    private int last_view;

    // For biplane measurement
    private ArrayList<Frame> saved_frames;
    private static final String[] FRAME_IDS = {"ED4", "ES4", "ED2", "ES2"};
    private static final int ED4_idx = 0;
    private static final int ES4_idx = 1;
    private static final int ED2_idx = 2;
    private static final int ES2_idx = 3;
    private float biplane_vol_ED, biplane_vol_ES;
    private boolean biplane_valid;

    public EFCalculator(QUSEventListener _QEL, int _width, int _height, int _length){
        width = _width;
        height = _height;
        length = _length;
        QEL = _QEL;
        frame_size = width*height;
        depth = 0;
        last_view = -1;

        segments = new byte[width*height*length];
        landmarks = new byte[width*height*length];
        pixels = new int[width*height*length];

        saved_frames = new ArrayList<>();
        for (int i=0;i<4;i++)
            saved_frames.add(new Frame(width, height, FRAME_IDS[i]));
    }

    public void clearResults(){
        Log.i(TAG,"Clearing results");
        biplane_valid = false;
        //for(Frame f:saved_frames) f.valid = false;
        last_view = -1;
        updateResults();
    }

    // TODO: delete _fullres once working
    public void setRecordedData(byte[] _segments, byte[] _landmarks){
        System.arraycopy(_segments,0,segments,0,width*height*length);
        System.arraycopy(_landmarks,0,landmarks,0,width*height*length);
        //for(Frame f:saved_frames) f.valid = false;
        //biplane_valid = false;
        //writeData(folder_path, counter, data_bytes);
    }

    public void setDepth(float _depth){
        //Log.i(TAG,"Depth changed = "+_depth);
        depth = _depth;
        biplane_valid = false;
        updateResults();
    }

    public void updateResults() {
        float px_density = (width*height)/(depth*depth); // px / cm^2
        float px_resoltn = (height)/(depth);             // px / cm

        Frame ED, ES;
        Log.i(TAG,"Last view = "+last_view);
        switch(last_view) {
            case BubbleService.AP4_VIEW_IDX:
                ED = saved_frames.get(ED4_idx);
                ES = saved_frames.get(ES4_idx);
                break;
            case BubbleService.AP2_VIEW_IDX:
                ED = saved_frames.get(ED2_idx);
                ES = saved_frames.get(ES2_idx);
                break;
            default:
                Log.e(TAG, "No view selected yet");
                QEL.updateEFOutputEvent(-1, -1, -0.01f, false);
                return;
        }
        if(!ED.valid | !ES.valid) return;
        float ED_vol = 0.85f*(float)Math.pow((double)(ED.area/px_density),2)/(ED.L/px_resoltn);
        float ES_vol = 0.85f*(float)Math.pow((double)(ES.area/px_density),2)/(ES.L/px_resoltn);
        float EF = (ED_vol - ES_vol)/ED_vol;
        QEL.updateEFOutputEvent(ED_vol, ES_vol, EF, false);

        if(biplane_valid){
            EF = (biplane_vol_ED - biplane_vol_ES)/biplane_vol_ED;
            QEL.updateEFOutputEvent(biplane_vol_ED, biplane_vol_ES, EF,true);
            //QEL.updateEFOutputEvent(biplane_vol_ED/(float)Math.pow(px_resoltn,3), biplane_vol_ES/(float)Math.pow(px_resoltn,3), EF,true);
        }
    }

    public void setLVAreas(int view_idx){
        // Update last requested view for updateResults()
        last_view = view_idx;

        // Find ED and ES indices
        int[] idxs = new int[2];
        findMinMaxFrames(idxs);

        for (int i = 0; i < 2; i++) {
            // Select current frame
            Frame curr_frame = saved_frames.get(i + ((view_idx == BubbleService.AP4_VIEW_IDX)?0:2));
            curr_frame.setData(segments, landmarks, frame_size * idxs[i], FRAME_IDS[i + ((view_idx == BubbleService.AP4_VIEW_IDX)?0:2)]);
            curr_frame.setArea();
            curr_frame.findLandmarks();
            curr_frame.setL();
            curr_frame.setAngle();
            curr_frame.setDepth(depth);
            curr_frame.valid = true;
        }

        // If both AP4 and AP2 are valid...
        for (Frame f : saved_frames)
            if(!f.valid) return;

        // ...set biplane volumes!
        for (Frame f : saved_frames)
            f.rotate(); // Note: rotating has now increased WIDTH and HEIGHT

        // Now scale up the smaller of AP4 and AP2 (for both ED and ES)
        Frame ED4 = saved_frames.get(ED4_idx),
              ED2 = saved_frames.get(ED2_idx),
              ES4 = saved_frames.get(ES4_idx),
              ES2 = saved_frames.get(ES2_idx);

        biplane_vol_ED = calculateBiplaneVolume(ED4, ED2);
        biplane_vol_ES = calculateBiplaneVolume(ES4, ES2);
        biplane_valid = true;
    }

    private float calculateBiplaneVolume(Frame AP4, Frame AP2){
        if (AP4.L > AP2.L) AP2.upscaleTo(AP4.L, AP4.depth);
        else AP4.upscaleTo(AP2.L, AP2.depth);
        ArrayList<Integer> a_s = AP4.getFunkSums();
        ArrayList<Integer> b_s = AP2.getFunkSums();
        Log.i(TAG, "length(a_s) = "+a_s.size()+"\t b_s = "+b_s.size());
        if(Math.abs(a_s.size()-b_s.size()) > 6) Log.e(TAG,"Funk maps are a little too funky...");
        ArrayList<Integer> longer, shorter;
        if(a_s.size() > b_s.size()){
            longer = a_s;
            shorter = b_s;
        } else {
            longer = b_s;
            shorter = a_s;
        }
        // Remove pixels from the start/end alternatingly until a.length == b.length
        boolean first_last = false;
        while(longer.size() > shorter.size()){
            first_last = !first_last;
            longer.remove((first_last)? 0 : longer.size()-1);
        }
        float sum = 0;
        for(int i = 0; i < Math.min(a_s.size(),b_s.size()); i++) sum += a_s.get(i)*b_s.get(i);

        Log.i(TAG,"AP4 depth was = "+AP4.depth);
        Log.i(TAG,"AP2 depth was = "+AP2.depth);
        Log.i(TAG,"APb depth was = "+((AP4.L > AP2.L)?AP4.depth:AP2.depth));
        Log.i(TAG,"L_4 / L_2 = "+(AP4.L/AP2.L));
        Log.i(TAG,"d_2 / d_4 = "+(AP2.depth/AP4.depth));

        return (float)Math.PI / 4.0f * sum * AP4.depth * AP2.depth * ((AP4.L > AP2.L)?AP4.depth:AP2.depth) / (float)Math.pow(width,3);
    }

    private void findMinMaxFrames(int[] idxs){
        // Step 1: Find ED and ES frame, max -> ED & min -> ES
        // TODO: Use median of top 90% and bottom 10% instead of max
        // TODO: What if there are 2?
        // TODO: this loop only needs to be run once
        int frame_size = width * height;
        int[] px_sums = new int[length];
        for (int k = 0; k < length; k++) {
            px_sums[k] = 0;
            for (int i = 0; i < frame_size; i++)
                px_sums[k] += segments[frame_size * k + i];
        }
        ArrayIndexComparator comparator = new ArrayIndexComparator(px_sums);
        Integer[] indexes = comparator.createIndexArray();
        Arrays.sort(indexes, comparator);
        int ED_idx_sorted = (int)Math.round(length*0.95); // Median of top 90 percentile?
        int ES_idx_sorted = (int)Math.round(length*0.05); // Median of bottom 10 percentile?
        idxs[0] = indexes[ED_idx_sorted];
        idxs[1] = indexes[ES_idx_sorted];
        Log.i(TAG, "Max frame found at k = " + idxs[0] + " with sum = " + px_sums[idxs[0]]);
        Log.i(TAG, "Min frame found at k = " + idxs[1] + " with sum = " + px_sums[idxs[1]]);
    }

    public boolean writeData(String folder_name, int counter, byte[] data_ptr) {
        Log.v(TAG,"Writing to: "+folder_name);

        // Make folder if it doesn't exist
        File folder = new File(folder_name);
        if (!folder.exists()) {
            Log.v(TAG, "Creating new folder: " + folder.getAbsolutePath());
            if (!folder.mkdirs())
                return false;//Log.e(TAG, "mkdirs returned false :(");
        } else Log.v(TAG, "Folder already exists: " + folder.getAbsolutePath());

        // Save tensor
        File file = new File(folder_name+"/"+counter+".bin");
        try {
            //Log.i(TAG, "filepath = " + file.getAbsolutePath());
            FileOutputStream os = new FileOutputStream(file);
            os.write(data_ptr);
            os.flush();
            os.close();
        } catch (Exception e) {
            e.printStackTrace();
            Log.e(TAG, "SAVE FILE ERROR");
            return false;
        }
        return true;
    }

    // TODO: ALL THE BIPLANE STUFF
    // Step 4. Resize frames so that L_ap2 = L_ap4
    // aka zoom in,

}

class Frame {
    private static String TAG = "nvw-EF";
    public byte[] segmentation, landmarks;
    public float area, L, angle, depth;
    public float[] apex, valve;
    private int w, h;
    public String id;
    public boolean valid = false;
    private Bitmap funk_bmp;

    public Frame(int _w, int _h, String _id) {
        w = _w;
        h = _h;
        id = _id;
        apex = new float[2];
        valve = new float[2];
        segmentation = new byte[w*h];
        landmarks = new byte[w*h];
    }

    public void setData(byte[] all_segments, byte[] all_landmarks, int offset, String id){
        //Log.i(TAG, "Setting data for frame: "+id);
        System.arraycopy(all_segments, offset, segmentation, 0, w*h);
        System.arraycopy(all_landmarks, offset, landmarks, 0, w*h);
    }

    public void setArea(){
        area = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                area += segmentation[y * w + x];
            }
        }
        Log.i(TAG, id+" - Area = " + area);
    }

    public void setL(){
        L = (float) Math.sqrt(Math.pow(apex[0] - valve[0], 2) + Math.pow(apex[1] - valve[1], 2));
        //Log.i(TAG, id+" - L = " + L);
    }

    public void setAngle(){
        angle = (float) (180.0d / Math.PI * Math.tan((valve[0] - apex[0]) / (valve[1] - apex[1])));
        //Log.i(TAG, id+" - angle = "+angle);
    }

    public void setDepth(float _d){depth = _d;}

    public void rotate(){
        // Note: rotating has now increased WIDTH and HEIGHT
        int[] pixels = new int[w*h];
        for (int i = 0; i < w*h; i++)
            pixels[i] = (0xFF << 24) | ((int) (segmentation[i] * 255) << 8);
        Bitmap src_bmp = Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888);
        Matrix rot_mat = new Matrix();
        rot_mat.setRotate(angle);
        funk_bmp = Bitmap.createBitmap(src_bmp, 0, 0, src_bmp.getWidth(), src_bmp.getHeight(), rot_mat, true);
    }

    public void upscaleTo(float biggerL, float smallerD){
        if(L > biggerL) {
            Log.i(TAG,"L should def be smaller than biggerL... L="+L+" \tbigL="+biggerL);
            return;
        }
        Log.i(TAG,id+" - Attempting to upscale by "+(depth/smallerD*100.0f)+"%");
        //Log.i(TAG,id+" - Before: w = "+funk_bmp.getWidth()+"\t h = "+funk_bmp.getHeight());
        // TODO: combine this with rotate? might not be worth the confusion
        Matrix scale_mat = new Matrix();
        scale_mat.postScale(depth/smallerD, depth/smallerD);
        funk_bmp = Bitmap.createBitmap(funk_bmp, 0, 0, funk_bmp.getWidth(), funk_bmp.getHeight(), scale_mat, true);
        //Log.i(TAG,id+" - After:  w = "+funk_bmp.getWidth()+"\t h = "+funk_bmp.getHeight());
    }

    public ArrayList<Integer> getFunkSums(){
        int[] px = new int[funk_bmp.getWidth()*funk_bmp.getHeight()];
        funk_bmp.getPixels(px,0,funk_bmp.getWidth(),0,0,funk_bmp.getWidth(),funk_bmp.getHeight());
        ArrayList<Integer> horz_sums = new ArrayList<>();
        for(int y = 0; y < funk_bmp.getHeight(); y++) {
            int sum = 0;
            for (int x = 0; x < funk_bmp.getWidth(); x++)
                if (((px[y*funk_bmp.getWidth()+x] & 0x0000FF00)>>8) >= (int)(0.75f*255.0f)) sum++;
            if(sum > 0) horz_sums.add(sum);
            //Log.i(TAG,id+" - sum["+y+"] = "+sum);
        }
        return horz_sums;
    }

    public void findLandmarks() {
        ArrayList<ConnectedComponent> ccs = new ArrayList<>();
        int cc_count = 0;
        // Find all connected components in frame
        for(int y = 0; y < h; y++){
            for(int x = 0; x < w; x++) {
                // First pixel in the next connected component
                if(landmarks[y*w+x] == 1.0f) {
                    ConnectedComponent new_cc = new ConnectedComponent(w, h, landmarks, x, y, ++cc_count, 1.0f);
                    ccs.add(new_cc); // Add it to the list
                }
            }
        }

        // Sort the cc list
        Collections.sort(ccs, new Comparator<ConnectedComponent>() {
            @Override
            public int compare(ConnectedComponent o1, ConnectedComponent o2) {
                return (o2.getCount() - o1.getCount());
            }
        });

//         Select the largest two
        for (ConnectedComponent cc: ccs) {
            Log.i(TAG,id+" - CC x = "+cc.getCOMx()+"\t y = "+cc.getCOMy()+" \t count = "+cc.getCount());
        }

        // Set the top one to be apex and the bottom valve
        int apex_idx, valve_idx;
        if (ccs.get(1)==null)
            return;
        apex_idx = (ccs.get(0).getCOMy() < ccs.get(1).getCOMy())? 0 : 1;
        valve_idx = 1 - apex_idx;
        apex[0] = ccs.get(apex_idx).getCOMx();
        apex[1] = ccs.get(apex_idx).getCOMy();
        valve[0] = ccs.get(valve_idx).getCOMx();
        valve[1] = ccs.get(valve_idx).getCOMy();

        //Log.i(TAG,id+" - APEX x = "+apex[0]+" y = "+apex[1]);
        //Log.i(TAG,id+" - VALV x = "+valve[0]+" y = "+valve[1]);
    }
}

class ConnectedComponent {
    private final String TAG = "nvw-CC";
    private byte[] frame;
    private float target;
    private int w, h;
    private int x_sum = 0, y_sum = 0;
    private int value, count = 0;

    public ConnectedComponent(int _w, int _h, byte[] _frame, int x_o, int y_o, int _value, float _target){
        w = _w;
        h = _h;
        //frame = new float[w][h];
        //System.arraycopy(_frame, 0, frame, 0, w*h);
        frame = _frame;
        target = _target;
        value = _value+1; // always go +1 so we dont go infinite
        if(value == target){
            Log.e(TAG, "This CC's value == 1, this will cause an infinite loop");
            return;
        }
        grow(x_o, y_o);
    }

    private void grow(int x, int y){
        // Add pixel!
        if(frame[index(x,y)]!=target) {Log.e(TAG,"growing on invalid pix. pix("+x+","+y+") = "+frame[index(x,y)]); return;}
        //Log.i(TAG,"Growing on valid pix. x = "+x+"\t y = "+y);
        frame[index(x,y)] = (byte)value; // 2 means we've added it to this CC
        x_sum+=x;
        y_sum+=y;
        count++;

        // Recurse!
        if(x-1 >= 0) {if(frame[index(x-1,y)] == target) grow(x-1,y);}
        if(x+1 <  w) {if(frame[index(x+1,y)] == target) grow(x+1,y);}
        if(y-1 >= 0) {if(frame[index(x,y-1)] == target) grow(x,y-1);}
        if(y+1 <  h) {if(frame[index(x,y+1)] == target) grow(x,y+1);}
    }

    public float getCOMx(){ return x_sum/(float)count;}
    public float getCOMy(){ return y_sum/(float)count;}
    public int getValue(){return value;}
    public int getCount(){return count;}
    public int index(int x, int y) {return y*w+x;}
}

class ArrayIndexComparator implements Comparator<Integer> {
    private final int[] array;
    public ArrayIndexComparator(int[] array) {
        this.array = array;
    }
    public Integer[] createIndexArray() {
        Integer[] indexes = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {
            indexes[i] = i; // Autoboxing
        }
        return indexes;
    }
    @Override
    public int compare(Integer index1, Integer index2) {
        // Autounbox from Integer to int to use as array indexes
        return array[index1] - array[index2];
    }
}