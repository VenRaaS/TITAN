from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search


def query_by_gids(gids):
    s = Search(using=Elasticsearch('miniconda:9200'))
    r = {}

    if len(gids) <= 0:
        return r

    s = s.filter('terms', gid=gids)
    s = s[0:50]
    for hit in s:
        r[hit.gid] = {
            'name': hit.goods_name, #.encode('utf-8'),
            'sale_price': hit.sale_price 
        }
    
    return r

if '__main__' == __name__:
    r = query_by_gids( ['4809792', '5341533'] )
    print r['4809792']['name'] #.decode('utf-8')
