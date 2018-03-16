#!/bin/bash
SCRIPT_DIR="$(cd $(dirname $0) && pwd)"
dt="$(date +%Y%m%d)"

[ -z ${VenRaaS_PROJECT_NAME} ] && VenRaaS_PROJECT_NAME="$1"
CODENAME=${VenRaaS_PROJECT_NAME}
if [ -z ${CODENAME} ]
then
    echo "codename is required"
    exit 1
fi

TDB_BQ="${CODENAME}_tmp"
RDB_BQ="${CODENAME}_results"
TDFS_GCS="gs://ven-cust-${CODENAME}/tmp"
RDFS_GCS="gs://ven-cust-${CODENAME}/results"

table_name="gid_gname"


outTb="${TDB_BQ}.${table_name}"
bqSql_gid_gname=$(printf "$(cat ${SCRIPT_DIR}/gid_gname.bq.sql)" ${CODENAME} ${dt})
bqCmd="bq query --use_legacy_sql=false --allow_large_results -n 0 --replace=true --destination_table '${outTb}' '${bqSql_gid_gname}'"
bash -c "${bqCmd}"

srcTb="${outTb}"
outGCS="${TDFS_GCS}/${table_name}.tsv"
bqCmd="bq extract --noprint_header -F \"\t\" ${srcTb} ${outGCS}"
bash -c "${bqCmd}"

localPath="/tmp/${CODENAME}_tag"
rm -rf ${localPath}
mkdir -p ${localPath}
cpCmd="gsutil cp ${outGCS} ${localPath}"
bash -c "${cpCmd}"

g2tFN='gid_terms.tsv'
srcFP="${localPath}/${table_name}.tsv"
outFP="${localPath}/${g2tFN}"
source `which virtualenvwrapper.sh`
workon i2t
python gid_terms.py -o "${outFP}" "${srcFP}"
deactivate

g2tGCS="${TDFS_GCS}/${g2tFN}"
uploadCmd="gsutil cp ${outFP} ${g2tGCS}"
bash -c "${uploadCmd}"

g2tTb="${TDB_BQ}.gid_terms"
bqCmd_load="bq load --source_format CSV -F '\t' --replace ${g2tTb} ${g2tGCS} gid:string,term:string"
bash -c "${bqCmd_load}"

g2ltermsTb="${TDB_BQ}.gid_lterms"
bqSql=$(printf "$(cat ${SCRIPT_DIR}/gid_lterms.bq.sql)" ${g2tTb} ${g2tTb})
bqCmd="bq query --use_legacy_sql=false --allow_large_results -n 0 --replace=true --destination_table '${g2ltermsTb}' '${bqSql}'"
bash -c "${bqCmd}"

g2ltermsRdictTb="${TDB_BQ}.gid_lterms_Rdict"
bqSql=$(printf "$(cat ${SCRIPT_DIR}/gid_lterms_Rdict.bq.sql)" ${g2ltermsTb} ${TDB_BQ})
bqCmd="bq query --use_legacy_sql=false --allow_large_results -n 0 --replace=true --destination_table '${g2ltermsRdictTb}' '${bqSql}'"
bash -c "${bqCmd}"

g2unionallTb="${TDB_BQ}.gid_unionall"
bqSql=$(printf "$(cat ${SCRIPT_DIR}/gid_unionall.bq.sql)" ${TDB_BQ} ${TDB_BQ})
bqCmd="bq query --use_legacy_sql=false --allow_large_results -n 0 --replace=true --destination_table '${g2unionallTb}' '${bqSql}'"
bash -c "${bqCmd}"

g2tagscores="${RDB_BQ}.gid_tagscores"
bqSql=$(printf "$(cat ${SCRIPT_DIR}/gid_tagscores.bq.sql)" ${g2unionallTb} ${CODENAME})
bqCmd="bq query --use_legacy_sql=false --allow_large_results -n 0 --replace=true --destination_table '${g2tagscores}' '${bqSql}'"
bash -c "${bqCmd}"

srcTb="${g2tagscores}"
outGCS="${RDFS_GCS}/keywords.tsv"
bqCmd="bq extract --noprint_header -F \"\t\" ${srcTb} ${outGCS}"
bash -c "${bqCmd}"

#-- gid 2 ancestor categories
g2a_category_code="${RDB_BQ}.gid2a_category_code"
bqSql=$(printf "$(cat ${SCRIPT_DIR}/gid2search_categs.bq.sql)" ${CODENAME} ${CODENAME} ${CODENAME})
bqCmd="bq query --use_legacy_sql=false --allow_large_results -n 0 --replace=true --destination_table '${g2a_category_code}' '${bqSql}'"
bash -c "${bqCmd}"

srcTb="${g2a_category_code}"
outGCS="${RDFS_GCS}/gid2ancestorCodes.tsv"
bqCmd="bq extract --noprint_header -F \"\t\" ${srcTb} ${outGCS}"
bash -c "${bqCmd}"

