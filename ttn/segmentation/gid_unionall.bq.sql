WITH
  t_lterms AS (
  SELECT
    gid,
    term
  FROM
    %s.gid_lterms_Rdict ),
  t_brands AS (
  SELECT
    gid,
    brand AS term
  FROM
    %s.gid_brands )
SELECT
  *
FROM (
  SELECT
    *
  FROM
    t_lterms
  UNION ALL
  SELECT
    *
  FROM
    t_brands )
GROUP BY
  gid,
  term
