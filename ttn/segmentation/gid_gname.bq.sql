SELECT
  GID,
  REGEXP_REPLACE(GOODS_NAME, "[\\t]+", " ") AS GOODS_NAME,
  UPDATE_TIME
FROM
  %s_unima.goods_%s
