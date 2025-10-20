# coding: utf-8
# _*_ coding: utf-8 _*_
# @Time : 2023/6/22 15:41 
# @Author : wz.yang 
# @File : data_process.py
# @desc :
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import en_core_web_sm
from flair.data import Sentence
from flair.models import SequenceTagger
import nltk
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
import json
import nltk
# 加载预训练模型
tagger = SequenceTagger.load('ner')
# 加载预训练模型
nlp = en_core_web_sm.load()

def process_content(document):
    words = nltk.word_tokenize(document)
    tagged = nltk.pos_tag(words)
    namedEnt = nltk.ne_chunk(tagged, binary=True)
    return namedEnt


# train_file = 'D:\Study\pycharm\workspace\Phd\Relation-Extraction\PR\Experiment\SemEval-2010-task8\Atomic_SemEval2010task8_tf1.0_noBERT\data\SemEval2010_task8_all_data\SemEval2010_task8_training\TRAIN_FILE.TXT'
# test_file = 'D:\Study\pycharm\workspace\Phd\Relation-Extraction\PR\Experiment\SemEval-2010-task8\Atomic_SemEval2010task8_tf1.0_noBERT\data\SemEval2010_task8_all_data\SemEval2010_task8_testing_keys\TEST_FILE_FULL.TXT'
#
# save_train = 'D:\Study\pycharm\workspace\Phd\Relation-Extraction\PR\Experiment\SemEval-2010-task8\Atomic_SemEval2010task8_tf1.0_noBERT\data\SemEval2010_task8_all_data\SemEval2010_task8_training\process_TRAIN_FILE.json'
# save_test = 'D:\Study\pycharm\workspace\Phd\Relation-Extraction\PR\Experiment\SemEval-2010-task8\Atomic_SemEval2010task8_tf1.0_noBERT\data\SemEval2010_task8_all_data\SemEval2010_task8_testing_keys\process_TEST_FILE_FULL.json'


train_file = 'data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
test_file = 'data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'

save_train = 'data/SemEval2010_task8_all_data/SemEval2010_task8_training/process_TRAIN_FILE.json'
save_test = 'data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/process_TEST_FILE_FULL.json'


def get_POS(text):
    text_list = nltk.word_tokenize(text)
    # POS = nltk.pos_tag(text_list)[0][1]  # 打标签
    POS = nltk.pos_tag(text_list)[0][1]
    return POS # 返回字符串

def get_entity_type(line, e, e_type, so_type):
    # 从句子里识别实体类型
    pure_sentence = line.split('\t')[1].replace('"', '').replace('<e1>', '').replace('</e1>', '').replace('<e2>','').replace('</e2>', '')
    # 使用nltk识别实体类型
    namedEnt = process_content(pure_sentence)
    for tagged_tree in namedEnt:
        if hasattr(tagged_tree, 'label'):
            entity_name = ' '.join(c[0] for c in tagged_tree.leaves())
            entity_type = tagged_tree.label()
            e_type[entity_name] = entity_type

    # 使用spacy识别实体类型
    doc = nlp(pure_sentence)
    for ent in doc.ents:
        e_type[ent.text] = ent.label_

    # 使用flair识别实体
    sentence = Sentence(pure_sentence)
    # 使用模型预测句子
    tagger.predict(sentence)
    # 打印实体
    for entity in sentence.get_spans('ner'):
        entity_name = entity.text
        entity_type = entity.tag
        e_type[entity_name] = entity_type

        # 从标记实体e1\e2里识别实体类型
        mark_sentence = line.split('\t')[1].replace('"', '')
        for word in mark_sentence.split():
            # 使用nltk识别实体类型
            namedEnt = process_content(e)
            for tagged_tree in namedEnt:
                if hasattr(tagged_tree, 'label'):
                    entity_name = ' '.join(c[0] for c in tagged_tree.leaves())
                    entity_type = tagged_tree.label()
                    e_type[entity_name] = entity_type

            # 使用spacy识别实体类型
            doc = nlp(e)
            for ent in doc.ents:
                e_type[ent.text] = ent.label_

            # 使用flair识别实体
            sentence = Sentence(e)
            # 使用模型预测句子
            tagger.predict(sentence)
            # 打印实体
            for entity in sentence.get_spans('ner'):
                entity_name = entity.text
                entity_type = entity.tag
                e_type[entity_name] = entity_type

            if e in e_type:
                print('e:', e, "实体识别：", e_type[e])
                return e_type[e], e_type
    if so_type == 1:
        return 'Subject', e_type
    if so_type == 2:
        return 'Object', e_type



def read_and_save(read_file, save_file):
    e_type = dict()
    with open(read_file, 'r', encoding='utf-8') as pr, open(save_file, 'w', encoding='utf-8') as pw:
        lines = pr.readlines()
        relation_instance = dict()
        num = 0
        for line_id, line in enumerate(lines[0::4]):
            sentence = line.split('\t')[1].replace('"', '')
            word_list = sentence.split()
            for id, word in enumerate(word_list):
                if '<e1>' in word:
                    e1_start = id
                if '</e1>' in word:
                    e1_end = id
                if '<e2>' in word:
                    e2_start = id
                if '</e2>' in word:
                    e2_end = id

            left_list = word_list[0:e1_start]
            e1_list = word_list[e1_start:e1_end + 1]
            middle_list = word_list[e1_end + 1:e2_start]
            e2_list = word_list[e2_start:e2_end + 1]
            e2_list[-1] = e2_list[-1].split('</e2>')[0]
            right_list = sentence.split('</e2>')[1].strip()

            left = ' '.join(left_list)
            e1 = ' '.join(e1_list)
            if '<e1>' in e1:
                e1 = e1.replace('<e1>', '')
            if '</e1>' in e1:
                e1 = e1.replace('</e1>', '')
            middle = ' '.join(middle_list)
            e2 = ' '.join(e2_list)
            if '<e2>' in e2:
                e2 = e2.replace('<e2>', '')
            if '</e2>' in e2:
                e2 = e2.replace('</e2>', '')
            right = ''.join(right_list)

            relation = lines[line_id * 4 + 1].strip()
            entity_position = '0'
            if '(e1,e2)' in relation:
                entity_position = '1'
            elif '(e2,e1)' in relation:
                entity_position = '2'
            relation_instance['id'] = line_id
            relation_instance['entity_position'] = entity_position
            relation_instance['left'] = left
            relation_instance['e1'] = e1
            relation_instance['middle'] = middle
            relation_instance['e2'] = e2
            relation_instance['right'] = right
            relation_instance['e1_type'], e_type = get_entity_type(line, e1, e_type, 1)
            relation_instance['e2_type'], e_type = get_entity_type(line, e2, e_type, 2)
            if middle == '':
                relation_instance['RightPOSofE1'] = get_POS(e2.split()[0])
                relation_instance['LeftPOSofE2'] = get_POS(e1.split()[-1])
            else:
                relation_instance['RightPOSofE1'] = get_POS(middle.split()[0])
                relation_instance['LeftPOSofE2'] = get_POS(middle.split()[-1])

            if left == '':
                relation_instance['LeftPOSofE1'] = ''
            else:
                relation_instance['LeftPOSofE1'] = get_POS(left.split()[-1])
            if right == '':
                relation_instance['RightPOSofE2'] = ''
            else:
                relation_instance['RightPOSofE2'] = get_POS(right.split()[0])

            relation_instance['sentence'] = sentence
            relation_instance['relation'] = relation
            json.dump(relation_instance, pw, indent=4)
            num += 1
            print('写入数据：', num, relation_instance)

read_and_save(train_file, save_train)
read_and_save(test_file, save_test)
