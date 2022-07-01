package com.lightgbm.score;

import com.lightgbm.cache.LgbCache;
import lombok.extern.slf4j.Slf4j;

import java.util.HashMap;
import java.util.Map;

@Slf4j
public class LgbScore {

  /**
   * lgb model 打分
   *
   * user profile
   * product profile
   */
  public static Double getScoreByLgbModel(LgbCache lgbCache) {
    if (lgbCache == null) {
      log.info("lgb cache is null");
      return 0.0;
    }

    try {
      Map<Integer, Float> featureIndices = new HashMap<>();

      //feature process here
      //item,user,cross,etc;

      if (featureIndices.isEmpty()) {
        log.info("feature indices is empty");
        return 0.0;
      }

      int[] indices = new int[featureIndices.size()];
      float[] values = new float[featureIndices.size()];
      int index = 0;
      for (Integer key : featureIndices.keySet()) {
        indices[index] = key;
        values[index] = featureIndices.get(key);
        index++;
      }
      double score = lgbCache.predict(indices, values);
      return score;
    } catch (Exception e) {
      log.error("Lgbm score fail");
      return 0.0;
    }
  }
}
