# coding=utf-8

#-- Jieba, Chinese text segmentation
#   usage and detail are able to be found by following links. 
#   https://github.com/fxsjy/jieba
#   http://blog.fukuball.com/ru-he-shi-yong-jieba-jie-ba-zhong-wen-fen-ci-cheng-shi/

import os
import sys
import getopt
import logging
import csv
import getopt

import jieba


formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(filename)s(%(lineno)s) %(name)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(ch)


if '__main__' == __name__ :
    try:
        opts, args = getopt.getopt(sys.argv[1:], "o:")
        if not '-o' in [opt for opt, arg in opts]:
            raise getopt.GetoptError('-o ouput is not specified.')

        if len(args) <= 0:
            raise getopt.GetoptError('input is not specified.')

    except getopt.GetoptError as e:
       logger.error(str(e))
       logger.error('usage:')
       logger.error('   {} -o <output full path> <input full path>'.format(__file__))
       exit()
    
    inFP = args[0]
    outFP = ''
    for opt, arg in opts:
        if '-o' == opt:
             outFP = arg

    logger.info('input fullpath: %s', inFP)
    logger.info('output fullpath: %s', outFP)

    #-- load test_userdict
    jieba.set_dictionary('./r.dic')

    outPath = os.path.split(outFP)[0]
    if outPath and not os.path.exists(outPath) :
        os.makedirs(outPath)

    fh = logging.FileHandler(outFP, mode='w') 
    fh.setFormatter(logging.Formatter('%(message)s'))
    loggerTag = logging.getLogger('tag')
    loggerTag.setLevel(logging.INFO)
    loggerTag.addHandler(fh)

    with open(inFP, 'rb') as tsv :
        tsvReader = csv.reader(tsv, delimiter='\t')

        logger.info('gid 2 terms is generating ...')
        for row in tsvReader:
            tags = ( jieba.cut(row[1]) )
            for t in tags:
                if t.strip():
                    loggerTag.info('%s\t%s', row[0], t)

