package com.lightgbm.lgbUtil;

public interface EarlyStopFunction {

  boolean callback(double[] d, int i);

}
