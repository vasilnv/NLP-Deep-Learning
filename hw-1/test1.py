#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################

import model
import nltk
from nltk.corpus import PlaintextCorpusReader
import importlib.util
import os
import os.path
import sys
import signal
import a1

def handler(signum, frame):
    raise Exception("end of time")

signal.signal(signal.SIGALRM, handler)

L1 = ['','заявката','заявката','заявката','заявката','заявката','заявката','заявката','zaazzaazzaazzaazazazazzzaazzzaza','abcd'*150]
L2 = ['','заявката','язвката','заявьата','завякатва','заявкатаа','вя','','zazazazzzazzzaazaazzazazazaz','badce'*100]
C = [0,0,3,1,2,1,7,8,9,350]
D = [0.0,0.0,8.0,2.5,5.25,3.0,20.25,24.0,23.75,887.5]

sna = {'ноа', 'нл', 'нн', 'нюа', 'фа', 'н-а', 'нао', 'ха', 'ыа', 'нй', 'нъ', 'нра', 'нта', 'н а', 'бна', 'нае', 'ни', '-а', 'а', 'ма', 'наю', 'ъна', 'она', 'нм', 'ра', 'над', 'ъа', 'ына', 'най', 'юа', 'ан', 'няа', 'ньа', 'наъ', 'вна', 'нь', 'наэ', 'тна', 'на-', 'ня', 'н-', 'наз', 'наа', 'нца', 'ца', 'аа', 'нэа', 'неа', 'са', 'щна', 'нг', 'нз', 'га', 'эна', 'нга', 'нъа', 'наы', 'нза', 'нда', 'ва', 'кна', 'да', 'наб', 'нх', 'эа', 'ба', ' на', 'нап', 'наг', 'нуа', 'нфа', 'наи', 'нща', 'мна', 'ча', 'нб', 'нш', 'нах', 'на ', 'нха', ' а', 'нша', 'нд', 'юна', 'шна', 'нар', 'ша', 'нам', 'нак', 'уна', 'ьа', 'фна', 'наф', 'гна', 'ана', 'иа', 'ща', 'нан', 'ьна', 'ниа', 'рна', 'пна', 'не', 'цна', 'ныа', 'нва', 'нка', 'ну', 'зна', 'оа', 'нау', 'нйа', 'наш', 'ена', 'яна', 'нба', 'нт', 'ню', 'н', 'нэ', 'нжа', 'нла', 'нпа', 'но', 'яа', 'нна', 'нж', 'еа', 'нав', 'нац', 'нса', 'нщ', 'нас', 'жна', 'нал', 'нц', 'нр', 'ина', 'лна', 'па', 'нп', 'нф', 'нс', 'нащ', 'та', 'чна', 'нча', 'дна', 'йна', 'уа', 'нат', 'нв', 'нач', '-на', 'ка', 'сна', 'нк', 'нма', 'жа', 'наь', 'нч', 'хна', 'ная', 'ны', 'н ', 'наж', 'за', 'йа', 'ла'}

print('Прочитане на корпуса от текстове...')
corpus_root = 'JOURNALISM.BG/C-MassMedia'
myCorpus = PlaintextCorpusReader(corpus_root, '.*\.txt')
fullSentCorpus = [ [model.startToken] + [w.lower() for w in sent] + [model.endToken] for sent in myCorpus.sents()]
print('Готово.')

print('Трениране на Марковски езиков модел...')
M2 = model.MarkovModel(fullSentCorpus,2)
print('Готово.')

#############################################################################
#### Начало на тестовете
#### ВНИМАНИЕ! Тези тестове са повърхностни и тяхното успешно преминаване е само предпоставка за приемането, но не означава задължително, че програмата Ви ще бъде приета. За приемане на заданието Вашата програма ще бъде подложена на по-задълбочена серия тестове.
#############################################################################


#### Тест на editDistance
try:
    for s1,s2,d in zip(L1,L2,C):
        signal.alarm(60)
        assert a1.editDistance(s1,s2) == d, "Разстоянието между '{}' и '{}' следва да е '{}'".format(s1,s2,d)
        signal.alarm(0)
    print("Функцията editDistance премина теста.")
except Exception as exception:
    print("Функцията editDistance не премина теста -- "+str(exception))
signal.alarm(0)

#### Тест на editWeight
try:
    for s1,s2,d in zip(L1,L2,D):
        signal.alarm(60)
        assert a1.editWeight(s1,s2) == d, "Теглото между '{}' и '{}' следва да е '{}'".format(s1,s2,d)
        signal.alarm(0)
    print("Функцията editWeight премина теста.")
except Exception as exception:
    print("Функцията editWeight не премина теста -- "+str(exception))
signal.alarm(0)

#### Тест на generate_edits
try:
    signal.alarm(60)
    assert len(set(a1.generateEdits("тест"))) == 305, "Броят на елементарните редакции \"тест\"  следва да е 305"
    assert len(set(a1.generateEdits("ааа"))) == 234, "Броят на елементарните редакции \"ааа\"  следва да е 234"
    assert set(a1.generateEdits("на")) == sna, "Елементарните редакции на \"на\" не съвпадат"
    signal.alarm(0)
    print("Функцията generateEdits премина теста.")
except Exception as exception:
    print("Функцията generateEdits не премина теста -- "+str(exception))
signal.alarm(0)

#### Тест на generate_candidates
try:
    signal.alarm(60)
    assert len(set(a1.generateCandidates("светвоно пофутбол",M2.kgrams[tuple()]))) == 2, "Броят на генерираните кандидати следва да е 2"
    signal.alarm(0)
    signal.alarm(60)
    assert len(set(a1.generateCandidates("подигот",M2.kgrams[tuple()]))) == 46, "Броят на генерираните кандидати следва да е 46"
    signal.alarm(0)
    print("Функцията generateCandidates премина теста.")
except Exception as exception:
    print("Функцията generateCandidates не премина теста -- "+str(exception))
signal.alarm(0)

#### Тест на correct_spelling
try:
    signal.alarm(60)
    assert a1.correctSpelling("светвоно пофутбол",M2,1.0) == 'световно по футбол', "Коригираната заявка следва да е 'световно по футбол'"
    assert a1.correctSpelling("подигот",M2,1.0) == 'полигон', "Коригираната заявка следва да е 'полигон'"
    signal.alarm(0)
    print("Функцията correctSpelling премина теста.")
except Exception as exception:
    print("Функцията correctSpelling не премина теста -- "+str(exception))
signal.alarm(0)

