WITH
  t_gid2categs AS (
  SELECT
    gid,
    category_code
  FROM
    %s_unima.goods_category_flatten
  GROUP BY
    gid,
    category_code),
  t_categ2cnt AS (
  SELECT
    category_code,
    COUNT(*) AS cnt
  FROM (
    SELECT
      gid,
      category_code
    FROM
      %s_unima.goods_category_flatten
    GROUP BY
      gid,
      category_code)
  GROUP BY
    category_code)
SELECT
  "%s" AS code_name,
  gid,
  category_code
FROM (
  SELECT
    gid,
    STRING_AGG(category_code, ",") OVER (PARTITION BY gid) AS category_code
  FROM (
    SELECT
      *
    FROM (
      SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY gid ORDER BY diff_searchSize ASC) num
      FROM (
        SELECT
          t_gid2categs.gid,
          t_gid2categs.category_code,
          ABS(3000 - cnt) AS diff_searchSize
        FROM
          t_gid2categs
        INNER JOIN
          t_categ2cnt
        ON
          t_gid2categs.category_code = t_categ2cnt.category_code) )
    WHERE
      num <= 2) t_topSearchCategs)
GROUP BY
  gid,
  category_code

