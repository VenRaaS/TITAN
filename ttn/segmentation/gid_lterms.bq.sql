SELECT
  *
FROM (
  SELECT
    gid,
    term,    
    cnt_subterm
  FROM (
    SELECT
      *,
      SUM(is_subterm) OVER (PARTITION BY gid, term) AS cnt_subterm
    FROM (
      SELECT
        *,
        if (jointerm = term, 0, IF(1 <= STRPOS(jointerm, term), 1, 0)) AS is_subterm
      FROM (
        SELECT
          t1.gid gid,
          t1.term term,          
          t2.term jointerm
        FROM
          %s AS t1
        INNER JOIN (
          SELECT
            gid,
            term
          FROM
            %s) AS  t2
        ON
          t1.gid = t2.gid)))
  GROUP BY
    gid,
    term,    
    cnt_subterm )
WHERE
  0 = cnt_subterm
  OR term IN ("nb",
    "s",
    "m",
    "l",
    "xl",
    "xxl" )
ORDER BY
  gid

