package com.lightgbm.lgbUtil;

import java.util.List;

public abstract class PredictFunction {

  abstract List<Double> predict(SparseVector vector);

}
