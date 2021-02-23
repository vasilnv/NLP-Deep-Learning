#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Домашно задание 3
###
#############################################################################

import numpy as np
import torch

def generateText(model, char2id, startSentence, limit=1000, temperature=1.):
    # model е инстанция на обучен LSTMLanguageModelPack обект
    # char2id е речник за символите, връщащ съответните индекси
    # startSentence е началния низ стартиращ със символа за начало '{'
    # limit е горна граница за дължината на поемата
    # temperature е температурата за промяна на разпределението за следващ символ
    
    result = startSentence[1:]

    #############################################################################
    ###  Тук следва да се имплементира генерацията на текста
    #############################################################################
    #### Начало на Вашия код.
    id2char = {i: c for i, c in enumerate(char2id)}
    ch = startSentence[0]
    X = model.preparePaddedBatch([ch])
    E = model.embed(X)

    outputPacked, (h0, c0) = model.lstm(torch.nn.utils.rnn.pack_padded_sequence(E, [1], enforce_sorted=False))
    output, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked)

    for ch in startSentence[1:]:
        X = model.preparePaddedBatch([ch])
        E = model.embed(X)

        outputPacked, (h0, c0) = model.lstm(torch.nn.utils.rnn.pack_padded_sequence(E, [1], enforce_sorted=False), (h0,c0))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked)
    dropped = model.dropout(output)

    L = model.projection(dropped.flatten(0, 1))
    p = torch.nn.functional.softmax(L / temperature)
    p = p.detach().numpy().squeeze()
    charid = np.random.choice(np.arange(len(char2id)), p=p)
    result += id2char[charid]

    i=0
    while i < limit and result[-1] != model.endToken:
        ch = result[-1]
        X = model.preparePaddedBatch([ch])
        E = model.embed(X)

        outputPacked, (h0, c0) = model.lstm(torch.nn.utils.rnn.pack_padded_sequence(E, [1], enforce_sorted=False),(h0,c0))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked)
        dropped = model.dropout(output)

        L = model.projection(dropped.flatten(0, 1))
        p = torch.nn.functional.softmax(L / temperature)
        p = p.detach().numpy().squeeze()
        charid = np.random.choice(np.arange(len(char2id)), p=p)
        result += id2char[charid]
        i+=1

    if result[-1] == model.endToken:
        result = result[:-1]
    #### Край на Вашия код
    #############################################################################

    return result
