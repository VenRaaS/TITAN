SELECT
  "%s" AS code_name,
  gid,
  a_category_code
FROM (
  SELECT
    gid,
    STRING_AGG(p_category_code, ",") OVER (PARTITION BY gid) AS a_category_code
  FROM (
    SELECT
      gid,
      p_category_code
    FROM (
      SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY gid ORDER BY dist2SearchWin ASC) AS row_num
      FROM (
        SELECT
          *,
          ABS(1000 - size) AS dist2SearchWin
        FROM (
          SELECT
            *,
            COUNT(DISTINCT gid) OVER (PARTITION BY p_category_code) AS size
          FROM (
            SELECT
              t_g2c.GID AS gid,
              t_g2c.CATEGORY_CODE AS category_code,
              t_c.P_CATEGORY_CODE AS p_category_code
            FROM
              %s_unima.unima_goods_cate_code AS t_g2c
            INNER JOIN (
              SELECT
                CATEGORY_CODE,
                P_CATEGORY_CODE
              FROM
                %s_unima.unima_category) AS t_c
            ON
              t_g2c.CATEGORY_CODE = t_c.CATEGORY_CODE )))) t_sortByIdeaSearchSize
    WHERE
      row_num <= 3) AS t_top2searchAncestor)
GROUP BY
  gid,
  a_category_code
