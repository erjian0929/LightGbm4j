package com.lightgbm.cache;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.lightgbm.lgbUtil.LgbParser;
import com.lightgbm.lgbUtil.Predictor;
import com.lightgbm.lgbUtil.SparseVector;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import lombok.extern.slf4j.Slf4j;

/**
 * lightgbm model & feature idmap cache,用于定期刷新模型
 */
@Slf4j
public class LgbCache {

  private static final int CACHE_REFRESH_HOUR = 2; // 缓存刷新时间

  private String KEY;
  private LoadingCache<String, LgbObj> cache;

  ScheduledExecutorService ses = Executors.newSingleThreadScheduledExecutor();

  public static final String LGB_MODEL_BASE_HDFS_PATH = "lgb.model.path";
  private static final String LGB_MODEL_NAME = "lgb.model.txt";
  private static final String LGB_FEATURE_ID_MAP_NAME = "lgb.feature.idmap.txt";

  public LgbCache() {

    String path = "";
    this.init("LGBMTuner", path, false);
  }

  public void init(String key, String modelPathPrefix, boolean localMode) {

    this.KEY = key;
    log.info("LgbCache key = {},modelPath = {},localMode = {}.", key,
        modelPathPrefix, localMode);

    cache = CacheBuilder.newBuilder().refreshAfterWrite(CACHE_REFRESH_HOUR, TimeUnit.HOURS)
        .build(new CacheLoader<String, LgbObj>() {
          @Override
          public LgbObj load(String key) throws Exception {
            LgbObj lgb = null;
            try {
              String modelHdfsPath = LGB_MODEL_NAME;
              String featureIdMapHdfsPath = LGB_FEATURE_ID_MAP_NAME;
              Predictor lgbModel = loadLgbModelFromPath(modelHdfsPath);
              Map<String, Integer> featureIdMap = loadFeatureIdMapFromPath(featureIdMapHdfsPath);
              lgb = new LgbObj(lgbModel, featureIdMap);
            } catch (Exception e) {
              log.error("LgbCache error:{}", e);
            }
            return lgb;
          }
        });

    // warm-up
    ses.scheduleWithFixedDelay(new Runnable() {
      @Override
      public void run() {
        try {
          LgbObj lgb = cache.get(KEY);
          if (lgb == null) {
            log.error("lgb = null,error!");
          }

        } catch (Exception e) {
          log.error("LgbCache warm-up error:{}", e);
        }
        log.info("cache key:{}", cache.asMap().keySet());
      }
    }, 0, CACHE_REFRESH_HOUR, TimeUnit.HOURS);

  }

  private Predictor loadLgbModelFromPath(String modelPath) {

    Predictor lgbModel = null;
    Configuration conf = new Configuration();
    try {
      InputStream in;
      in = new FileInputStream(modelPath);
      String modelStr = IOUtils.toString(in, StandardCharsets.UTF_8);
      in.close();
      LgbParser parser = new LgbParser();
      lgbModel = parser.parse(modelStr);
      log.info("load lgb model from path = {},succeed!", modelPath);
    } catch (Exception e) {
      log
          .error("loadLgbModelFromPath from path = {},error:{}", modelPath, e);
    }
    return lgbModel;

  }

  private Map<String, Integer> loadFeatureIdMapFromPath(String featureIdMapPath) {
    Configuration conf = new Configuration();
    Map<String, Integer> featureIdMap = new HashMap<>(10240);
    try {
      BufferedReader reader;
      reader = new BufferedReader(new InputStreamReader(new FileInputStream(featureIdMapPath)));
      String line;
      while ((line = reader.readLine()) != null) {
        String[] splits = line.trim().split("\t");
        if (splits.length == 3) {
          String featureName = splits[1];
          int featureId = Integer.parseInt(splits[0]);
          featureIdMap.put(featureName, featureId);
        }
      }
      reader.close();
      log
          .info("load featureIdMap from path = {},succeed,size = {}", featureIdMapPath, featureIdMap.size());
    } catch (Exception e) {
      log
          .error("loadFeatureIdMapFromPath from path = {},error:{}", featureIdMapPath, e);
    }
    return featureIdMap;
  }

  public Map<String, Integer> getFeatureIdMap(String key) {
    Map<String, Integer> featureIdMap = new HashMap<>();
    try {
      LgbObj lgb = null;
      lgb = cache.get(KEY);
      if (lgb != null) {
        featureIdMap = lgb.getFeatureIdMap();
      }
    } catch (Exception e) {
      log.error("getFeatureIdMap error", e);
    }
    return featureIdMap;
  }

  public double predict(int[] indices, float[] values) {

    double score = 0;
    try {
      LgbObj lgb = cache.get(KEY);
      if (lgb != null) {
        Predictor lgbModel = lgb.getLgbModel();
        if (lgbModel != null) {
          score = scoreByLgbModel(lgbModel, indices, values);
        }
      }
    } catch (Exception e) {
      log.error("LgbCache predict error", e);
    }
    return score;
  }

  private double scoreByLgbModel(Predictor lgbModel, int[] indices, float[] values) {
    double score = 0.0;
    try {
      List<Double> scores = lgbModel.predict(new SparseVector(values, indices));
      score = scores.get(0);
    } catch (Exception e) {
      log.error("scoreByLgbModel error", e);
    }
    return score;
  }


  class LgbObj {

    private Predictor lgbModel;
    private Map<String, Integer> featureIdMap;

    public LgbObj(Predictor lgbModel, Map<String, Integer> featureIdMap) {
      this.lgbModel = lgbModel;
      this.featureIdMap = featureIdMap;
    }

    public Predictor getLgbModel() {
      return lgbModel;
    }

    public void setLgbModel(Predictor lgbModel) {
      this.lgbModel = lgbModel;
    }

    public Map<String, Integer> getFeatureIdMap() {
      return featureIdMap;
    }

    public void setFeatureIdMap(Map<String, Integer> featureIdMap) {
      this.featureIdMap = featureIdMap;
    }

  }

  public void close() {
    if (cache != null) {
      cache.cleanUp();
    }
    if (ses != null) {
      ses.shutdownNow();
    }
  }
}
