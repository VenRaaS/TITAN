WITH
  t_cc AS (
  SELECT
    gid
  FROM
    {cn}_Unima.GoodsCategoryFlatten_{dt}
  WHERE
    category_code = '{ccode}')

SELECT 
  goods_img_url 
FROM (
    SELECT 
      goods_img_url
    FROM
      {cn}_Unima.Goods_{dt} AS t_goods
    INNER JOIN
      t_cc
    ON
      t_goods.gid = t_cc.gid
)
GROUP BY
  goods_img_url
