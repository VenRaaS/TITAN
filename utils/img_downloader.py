import os
import argparse
import subprocess
from urllib import urlretrieve
from multiprocessing import Pool


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cn', required=True, help='codename')
    parser.add_argument('-dt', required=True, help='datetime, e.g. 20180222')
    parser.add_argument('-cc', required=True, help='category code')
    args = parser.parse_args()

    sql = ''
    with open('query_imgurl.bq.sql', 'r') as f:
        sql = f.read()
    sql = sql.format(cn=args.cn, dt=args.dt, ccode=args.cc)

    name = 'imgurls'
    tmpTb = '{cn}_tmp.{tb}'.format(cn=args.cn, tb=name)
    cmd = 'bq query -n 0 --replace --use_legacy_sql=False --destination_table={} {}'.format(tmpTb, sql)
    print cmd
    subprocess.call(cmd.split(' '))

    gsTmpFP = os.path.join('gs://ven-cust-{cn}/tmp/{fn}.tsv'.format(cn=args.cn, fn=name))
    cmd = 'bq extract --noprint_header -F ''\t'' {} {}'.format(tmpTb, gsTmpFP)
    print cmd
    subprocess.call(cmd.split(' '))

    cmd = 'sudo gsutil cp {} {}'.format(gsTmpFP, '.')
    print cmd
    subprocess.call(cmd.split(' '))

    imgDir = 'image'
    cmd = 'rm -rf {}'.format(imgDir)
    print cmd
    subprocess.call(cmd.split(' '))
    os.makedirs(imgDir)

    def downloader(url):
        h, t = os.path.split(url)
        if t:
            print 'download {} ...'.format(url)
            urlretrieve(url, os.path.join(imgDir, t))

    with open('{}.tsv'.format(name), 'r') as f:
        urls = [ url.rstrip() for url in f ]
    pool = Pool(processes=4) 
    pool.map(downloader, urls)
