SELECT
  gid,
  term
FROM
  %s
WHERE
  term IN (
  SELECT
    term
  FROM
    %s.r_dict )
