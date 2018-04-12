import csv


i2iList_dic = {}

def load_csv(i2iModelFP) :
    with open(i2iModelFP, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if 2 == len(row):
                i2iList_dic[ row[0] ] = row[1].split(',')

def i2iList(k, modelFP) :
    global i2iList_dic

    if len(i2iList_dic) <= 0: load_csv(modelFP)

    return  i2iList_dic[k] if k in i2iList_dic  else []
    

if '__main__' == __name__ :
#    load_csv('FeaVCT_cc21000_flat/image_i2i.tsv')
    print i2iList('2853125_L', 'FeaVCT_cc21000_flat/image_i2i.tsv')

