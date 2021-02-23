#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Домашно задание 2 -- партидно стохастично спускане по градиента
###
#############################################################################

import numpy as np

def stochasticGradientDescend(data, U0, V0, contextFunction, lossAndGradientFunction, batchSize=1000, epochs=1, alpha=1., printInterval=10000):
    ###  Специализирана функция за стохастично спускане по градиента за модела word2vec skip-gram negative-sampling.
    ###  data -- списък от двойки (целева дума, контекстна дума);
    ###  U0, V0 -- начални матрици на влагания;
    ###  contextFunction -- функция, която по контекстна дума връща контекста, състоящ се от контекстната дума и негативните примери;
    ###  lossAndGradientFunction -- функция, която за цяла партида връща грешката и градиентите.
    epoch=0
    U=U0
    V=V0
    idx = np.arange(len(data))
    while epoch<epochs:
        np.random.shuffle(idx)
        for b in range(0,len(idx),batchSize):
            batchData = []
            for k in range(b,min(b+batchSize,len(idx))):
                w,c = data[idx[k]]
                batchData.append((w,contextFunction(c)))
            u_w = np.stack( [ U[w] for w,_ in batchData ] )
            Vt  = np.stack( [ V[context] for _,context in batchData ] )
            J, du_w, dVt = lossAndGradientFunction(u_w,Vt)
            for i,(w,context) in enumerate(batchData):
                #############################################################################
                ###  Тук следва да се имплементира презаписването на параметрите U и V,
                ###  като се извадят за всяко наблюдение от партидата alpha по съответните градиенти
                #############################################################################
                #### Начало на Вашия код. На мястото на pass се очакват 2-4 реда

                U[w] -= alpha * du_w[i]
                V[context] -= alpha * dVt[i]

                #### Край на Вашия код
                #############################################################################
            if b % printInterval == 0:
                print('Epoch:',epoch,'Sample:',b,'/',len(idx),'Loss:',J)
        epoch += 1
    return U,V


