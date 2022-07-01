package com.lightgbm.lgbUtil;

public abstract class Decision<T> {

  abstract boolean decision(T fval, T threshold);

}
