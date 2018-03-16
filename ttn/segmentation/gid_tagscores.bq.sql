WITH
  gid_unionall AS (
  SELECT
    gid,
    term,
    1.0 AS score
  FROM
    %s)
SELECT
  "%s" AS code_name,
  gid,
  terms
FROM (
  SELECT
    gid,
    STRING_AGG(ts, ",") OVER (PARTITION BY gid) AS terms
  FROM (
    SELECT
      gid,
      CONCAT(term, ":", CAST(ROUND(score, 1) AS string)) AS ts
    FROM
      gid_unionall
    ORDER BY
      gid ))
GROUP BY
  gid,
  terms
