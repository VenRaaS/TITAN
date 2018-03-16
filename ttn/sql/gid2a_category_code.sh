
SCRIPT_DIR="$(cd $(dirname $0) && pwd)"

[ -z ${CODE_NAME} ] && CODE_NAME="$1"

outTb="${CODE_NAME}_tmp.gid2a_category_code"
bqSql=$(printf "$(cat ${SCRIPT_DIR}/gid2a_category_code.bq.sql)" ${CODE_NAME} ${CODE_NAME} ${CODE_NAME})
#echo "$bqSql"
bqCmd="bq query --use_legacy_sql=false --allow_large_results -n 0 --replace=true --destination_table '${outTb}' '${bqSql}'"
bash -c "${bqCmd}"


