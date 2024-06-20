

import hanlp
import os
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
from nltk.util import pr
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# HanLP = hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE)
# predictor = Predictor.from_path("/data1/gl/project/coref/coref-spanbert-large-2021.03.10.tar.gz")
# 判断一个字符串中是否有http链接
def check_http(str):
    str_lst=str.split()
    s=[]
    for word in str_lst:
        # print(word)
        if word.startswith(('http://','https://')):
            if not s:
                continue
            else:
                if s[-1]=='-lsb-':
                    s=s[:-1]
        else:
            s.append(word)
    if len(s)==0 or len(s)==1:
        return None
    else:
        return ' '.join(s)

def read_sents(sents):
    document=''
    end_flag=[]
    count=0
    for sent in sents:
        sent=sent.strip()
        sent=check_http(sent)
        if not sent:
            continue
        # if sent.endswith(r' "'):
            # sent=sent+r' .'
        sent=sent.replace('-lrb-','(')
        sent=sent.replace('-rrb-',')')
        pattern1='('+"\\s{0,}"+')'
        start_pos=0
        while '( )' in sent:
            pos=sent.find('( )')
            sent=sent[:pos]+sent[pos+3:]
        if count!=0:
            last_sent_word=document[-1]
        document+=' '+sent
        start_word=sent.split()[0]
        if '-' in start_word and not start_word.startswith('-'):
            pos_1=start_word.find('-')
            # pos_2=start_word.rfind('-')
            start_word=start_word[:pos_1]
        if start_word.endswith('.') and len(start_word)>1:
            start_word=start_word[:-1]
        if count!=0:
            end_flag.append(last_sent_word+' '+start_word)
        count+=1
    return document[1:]+' ',end_flag

def coref_sents(document,end_flag):
    # document='''john " jack " reynolds -lrb- 21 february 1869 -- 12 march 1917 -rrb- was a footballer who played for , among others , west bromwich albion , aston villa and celtic . as an international he played five times for ireland before it emerged that he was actually english and he subsequently played eight times for england . he is the only player , barring own goals , to score for and against england and is the only player to play for both ireland and england . he won the fa cup with west bromwich albion in 1892 and was a prominent member of the successful aston villa team of the 1890s , winning three english league titles and two fa cups , including a double in 1897 . reynolds was noted as a highly competitive player with some remarkable ball skills and exceptionally brilliant footwork . he was regarded as one of the great footballers of the 1890s and was one of the highest paid players of his generation . however he also gained a reputation for drinking and womanising and as result much of the money he earned disappeared . he fathered at least one illegitimate child and in 1899 he appeared in court for non-payment of child maintenance . his heavy drinking blighted his latter career and after brief spells at celtic and then southampton , he became a semi-professional journeyman . towards the end of his life he worked as a miner in sheffield and he died alone in a boarding house at the age of 48 . reynolds and his career have been the subject of several lectures , including one entitled " how to play football , win friends and die young : the life of john reynolds " , given by dr. neal garnham at the university of ulster . '''

    res=predictor.predict(document=document)
    document=res['document']
    print(end_flag)
    clusters=res['clusters']
    # end_flag=['. as','. he','. he','. reynolds','. he','. however','. he','. his','. towards','. reynolds']
    end_pos=[]
    start_pos=0
    for c in end_flag:
        print(c,'++++++++++++++++++',document)
        c1,c2=c.split()
        pos_1=document.index(c1,start_pos)
        pos_2=document.index(c2,pos_1+1)
        if (pos_1+1)==pos_2:
            end_pos.append(pos_1)
            start_pos=pos_2+1
            continue
        else:
            while (pos_1+1)!=pos_2:
                start_pos=pos_1+1
                # print(document)
                pos_1=document.index(c1,start_pos)
                pos_2=document.index(c2,pos_1+1)
            else:
                end_pos.append(pos_1)
                start_pos=pos_2+1
            

    # print(end_pos)
    sents=[]
    sents_lst=[]
    start=0
    for i in range(len(end_flag)):
        sents.append(' '.join(document[start:end_pos[i]+1]))
        sents_lst.append(document[start:end_pos[i]+1])
        start=end_pos[i]+1
        if i==len(end_flag)-1:
            sents.append(' '.join(document[start:]))
            sents_lst.append(document[start:])
    return sents,sents_lst,clusters,end_pos,document


def get_pos(tree):
    pos=0
    leaves=[]
    try:
        for i in range(len(tree)):
            if tree[i].label()=='NP':
                leaves=tree[i].leaves()
                break
            else:
                pos+=len(tree[i].leaves())
        if pos !=len(tree.leaves()):
            return pos,leaves
        else:
            return -1,None
    except:
        return -2,'del'

def get_correct(sents):    
    correct=[]
    for sent in sents:
        # sent = 'sen{}'.format(i)
        doc = HanLP(sent)
        # doc.pretty_print()
        con=doc['con']
        pos,subject=get_pos(con[0])
        if pos==-1:
            new_pos=0
            new_subject=[]
            for i in range(len(con[0])):
                new_pos,new_subject=get_pos(con[0][i])
                if new_pos!=len(con[0][i].leaves()):
                    pos=new_pos
                    subject=new_subject
                    break
            
        # print(pos,subject)
        if subject!=None:
            if subject=='del':
                correct.clear()
                break
            print(pos,subject)
            correct.append([pos,' '.join(subject)])

    # print(correct)
    return correct


def get_final_sents(clusters,correct,end_pos,sents_lst,document):
    if not correct:
        sents=[]
    else:
        match={}
        for cluster in clusters:
            mid=[]
            for c in cluster:
                s,e=c
                k=0
                for i,pos in enumerate(end_pos):
                    k=i
                    if s<end_pos[i]:
                        break
                if k!=0:
                    new_s,new_e=s-end_pos[k-1]-1,e-end_pos[k-1]-1
                else:
                    new_s,new_e=s,e
                mid.append([' '.join(document[s:e+1]),(new_s,new_e+1),k])
            match[mid[0][0]]=mid

        # print(match)
        dai=['he','his','she','her']
        for i,subject_info in enumerate(correct):
            start,subject=subject_info
            subject_lst=subject.split()
            if subject_lst[0] in dai:
                for key,value in match.items():
                    for v in value:
                        item,pos,sen_id=v
                        if item==subject_lst[0] and pos[0]==start:
                            if subject_lst[0] in ['his','her']:
                                sents_lst[i]=sents_lst[i][:start]+[key]+[r"'s"]+sents_lst[i][pos[1]:]
                            else:
                                sents_lst[i]=sents_lst[i][:start]+[key]+sents_lst[i][pos[1]:]
                            break
        sents=[' '.join(sent) for sent in sents_lst]
    return sents

sents=[]
with open('sents.txt','r',encoding='utf-8') as rstream:
    while True:
        sen=rstream.readline()
        if not sen:
            break
        sents.append(sen[:-1])

# document,end_flag=read_sents(sents)
# sents,sents_lst,clusters,end_pos,document=coref_sents(document,end_flag)
# correct=get_correct(sents)
# sents=get_final_sents(clusters,correct,end_pos,sents_lst,document)
# print(sents)
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
# HanLP=hanlp.load(hanlp.pretrained.dep.CTB9_DEP_ELECTRA_SMALL)
text=''.join(["广", "西", "壮", "族", "自", "治", "区", "柳", "城", "县", "人", "民", "检", "察", "院", "指", "控", "，", "2", "0", "1", "5", "年", "6", "月", "3", "日", "晚", "，", "被", "告", "人", "谢", "某", "容", "留", "李", "某", "乙", "、", "李", "某", "甲", "、", "覃", "某", "甲", "、", "彭", "某", "、", "张", "某", "甲", "等", "人", "在", "柳", "城", "县", "东", "泉", "镇", "森", "堡", "k", "t", "v", "贵", "宾", "l", "0", "号", "包", "厢", "内", "吸", "食", "毒", "品", "氯", "胺", "酮", "（", "俗", "称", "k", "粉", "）", "。"])
doc = HanLP(text, tasks='dep')
print(doc)
print(len(doc['dep']))
print(len(doc['tok/fine']))
print(len(doc['pos/ctb']))

# seg=HanLP.segment(text)







