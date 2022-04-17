import re
import copy
import sys
from itertools import zip_longest

bpss = ("bps", "kbps", "Mbps", "Gbps")
volts = ("V", "mV", "uV", "nV", "pV", "fV", "aV")
ampers = ("A", "mA", "uA", "nA", "pA", "fA", "aA")
seconds = ("s", "ms", "us", "ns", "ps", "fs", "as")
units = (volts + ampers + seconds + bpss)

filepath = '/Users/fedor/PycharmProjects/MC_simulation/input.txt'
measures = []
junk = {}

inp_file = open(filepath, "r")

for line in inp_file:
    measures.append(line)
inp_file.close

meas_count = 0  # ??????????????????????????????????
ln = -1
previous_test = -1
previous_site = -1

failed_pat = False
data_zabrata = False
measure_started = False
treasure_started = False
decisure_started = False


fails = [[]]

print(len(measures))
#                               ГЛАВНЫЙ ОБРАБАТЫВАЮЩИЙ ЦИКЛ
for line in measures:
    ln += 1  # номер строки кроч
    lm = copy.deepcopy(line)  # сделали копию строки
    lm = lm.split()  # чтобы её разбить
    if len(lm) == 0:  # (если конечно в ней есть что разбивать)
        continue
    else:
        if line.split() != lm:  # чтобы взглянуть на неё подразными углами
            print("Error (original and splitted lines mismatch) Line: " + str(
                ln) + "\n" + line)  # но вообще это вряд ли пригодится

        # print("01")
        if lm == ['Datalog', 'report']:  # если видим заголовок этикетки
            measure_started = True  # значит начинается измерение
            treasure_started = True  # и начинается этикетка
            try:
                measures[ln + 1].split  # и следующей строкой должна быть дата
            except Exception:  # но может произойти любая хуйня
                continue  # но не хотелось бы
            else:  # если всё норм
                nextline = measures[ln + 1].split()  # забираем дату
                buffer = ["0", nextline[0], nextline[1], []]  # добавляем её в буфер
                data_zabrata = True  # сообщаем об этом в следующую строчку
        # print("02")
        if data_zabrata == True:
            data_zabrata = False
            continue  # которую соответственно пропускаем

        # print("03")
        if lm == ['Site', 'Number:']:  # этой строкой
            treasure_started = False  # этикетка заканчивается
            sites = re.split(r'[ ,]+', measures[ln + 1])
            try:
                re.split(r'[ ,]+', measures[ln + 1])  # и из следующей можно сразу выдернуть номера сайтов
            except Exception:  # но может произойти любая хуйня
                print("Error (while sites numbers reading) Line: " + str(ln) + "\n" + line)
                continue  # но не хотелось бы
            else:  # если всё норм
                sites = re.split(r'[ ,]+', measures[ln + 1])  # забираем номера сайтов
                sites.pop(0)
                sites[-1] = sites[-1][:-1]
                continue

        # print("04")
        if treasure_started == True:  # если этикетка ещё не кончилась:
            if lm[0][-1] == ":" or lm[1][-1] == ":":  # если в конце первого или второго слова есть двоеточие
                buffer[3].append(
                    line.strip())  # то это похоже на инфу из этикетки и мы её запихиваем в отдельный подсписок четвёртым элементом буфера, шоб не мешалася
                continue

        # print("05")
        if lm[0] == "Device#:" and len(
                sites) == 1:  # если строка начинается с Device#: а в списке сайтов у нас всего один сайт, то
            buffer.append(sites[0])  # тупо дописываем его номер в буфер
            buffer[0] = lm[1]
            buffer = [buffer]
            continue
        # print("06")
        if lm[0] == "Device#:" and len(sites) > 1:  # если сайтов несколько, то тут немного сложнее
            if lm[1].find("-") == -1:  # неплохо бы сразу проверить, указан один device number или диапазон
                print(
                    "Error (single device number for sites " + str(sites) + " is given) Line: " + str(ln) + "\n" + line)
                sys.exit()
            else:
                if int(lm[1][lm[1].index("-") + 1:]) - int(lm[1][:lm[1].index("-")]) + 1 != len(
                        sites):  # и если диапазон, то согласуется ли он с числом сайтов
                    print(int(lm[1][lm[1].index("-"):]) + " - " + int(lm[1][:lm[1].index("-")]) + " ?= " + len(sites))
                    print("Sites numbers : " + str(sites))

                    print("Error (device numbers and sites count mismatch) Line: " + str(ln) + "\n" + line)
                    sys.exit()
            multisite = True
            buffer = [buffer + [sites_element] for sites_element in
                      sites]  # размножаем буффер, путём предложенным Никсом, добавляя в него заодно номера сайтов
            s = 0
            for sites_element in sites:
                buffer[s][0] = int(lm[1][:lm[1].index("-")]) + s
                s += 1
            continue

        # print("07")
        if lm == ['Number', 'Site', 'Test', 'Name', 'Pin', 'Channel', 'Low', 'Measured', 'High', 'Force', 'Loc']:
            par_test = True  # шапка предвещающая параметрический тест
            continue
        # print("08")
        if lm == ['Number', 'Site', 'Test', 'Name', 'Pattern', '1st', 'Failed', 'Cycle', 'Total', 'Failed', 'Cycles']:
            par_test = False  # шапка предвещающая функциональный тест
            continue
        # print("09")
        if lm == ['Site', 'Failed', 'tests/Executed', 'tests']:  # шапка, с которой начинается
            decisure_started = True  # заключение (сорт, бин, сколько тестов пройдено итп), тоже своего рода финальная этикетка
            failed_pat = False  # может служить завершением векторной простыни
            continue

        # print("10")
        if lm == ['------------------------------------'] or lm == [
            '=========================================================================']:
            continue
        # print("11")
        if lm[0][0] == "<" and lm[0][-1] == ">":
            if failed_pat == True:
                failed_pat = False
            test_group_name = lm[0][1:-1]
            continue
        # print("12")
        if len(lm) > 3 and measure_started == True:
            if lm[0].isdigit == True and lm[1].isdigit == True:
                if previous_test == lm[0] - 1 or (previous_test == lm[0] and previous_site == lm[0] - 1):
                    failed_pat == False
                    # здесь continue не должно быть!!!
                    # это просто последний шанс отличить новый тест от простыней векторов, при переходе на новую строку

        if failed_pat == True:
            print("простынка")
            continue

        '''
            чо сюда нужно
            скидывание failed_pat (эта ваще похуй, ваще потом))

            вынание (F) и сразу же их удаление чтобы не мешались
            сопсна распознание min meas max и force

        '''

        if len(lm) > 3 and measure_started == True:
            if lm[0].isdigit == True and lm[1].isdigit == True:
                if previous_test + 1 == int(lm[0]) or (previous_test == lm[0] and previous_site == lm[0] - 1):
                    print("заходик совершен")
                    if par_test == True:
                        # здесь мы проверяем, есть ли в строке (F) или (А)
                        if "(F)" in lm:
                            lm.remove("(F)")
                            fail_mark = True

                        if "(A)" in lm:
                            lm.remove("(A)")
                            mode_alarm = True

                        en = 0
                        while en < 20:
                            # вариант без ограничений
                            if lm[en] == "N/A" and re.fullmatch('r[-]{0,1}[0-9]{1,25}\.[0-9]{2,4}', str(lm[en + 1])) and \
                                    lm[en + 2] in units and lm[en + 3] == "N/A":
                                print(lm[0])
                                print(lm[en + 1] + lm[en + 2])
                            # вариант с нижним ограничением
                            # вариант с верхним ограничением
                            # вариант с нижним и верхним ограничением
                            en += 1

                previous_test = lm[0]
                previous_test = lm[1]

'''
        if measure_started != True and line!='Datalog report' and lm[0]!='Device#:':   #если вдруг какой мусор: print(мусор)
            junk[str(ln)]=line
        if len(lm)==2 and 

        #if line[0]!='Datalog'
'''

sys.exit()

# _______________________________________________________________________________________
########################################################################################


for measure in file1:  # повынули букав из файла
    if wait == 0 and (measure.startswith("Datalog report") or measure.startswith("    Device#:")):
        meas_count += 1
        # print(meas_count)
        measures.append([])
        if measure.startswith("Datalog report"):
            wait = 25
    if wait != 0:
        wait -= 1
    measure = measure.split()
    measures[meas_count].append(measure)

file1.close()

for measure in measures[1:]:
    buffer = [0]
    buffer.append(measure[1][0])
    buffer.append(measure[1][1])
    buffer.append("tablichka")  # measure[2:10])

    multisite = False  # разбираемся, мультисайт у нас тут или что
    sites = measure[measure.index(['Site', 'Number:']) + 1]
    if len(sites) == 1:
        buffers = [buffer]
    else:
        sites = [site.replace(",", "") for site in sites]
        buffers = [buffer + [site] for site in sites]
        multisite = True

    testname = ""
    testnunmber = 0
    sitenumber = 0
    for line in measure:  # забрали номера девайсов в этом измерении и проверили их на корректность
        if line != []:
            if line[0] == 'Device#:':
                device_number_s = copy.deepcopy(line[1])
                if multisite == False:
                    if device_number_s.find("-") != -1:
                        print("one site and many devices")
                else:
                    if device_number_s.find("-") == -1:
                        print("one device and many sites")
                    device_number_s.split("-")
                    if int(device_number_s[1]) - int(device_number_s[0]) + 1 != len(sites):
                        print("devices and sites count mismatch")
            if line == ['Number', 'Site', 'Test', 'Name', 'Pin', 'Channel', 'Low', 'Measured', 'High', 'Force',
                        'Loc']: par_test = True  # сообщает какие будут дальнейшие тесты, парамерические
            if line == ['Number', 'Site', 'Test', 'Name', 'Pattern', '1st', 'Failed', 'Cycle', 'Total', 'Failed',
                        'Cycles']: par_test = False  # или всё ж таки функциаональные
            if line[0][0] == "<":
                test_name = line[0]
                # print((test_name,par_test))
            if len(line) > 3:
                if line[0].isdigit() and line[1].isdigit():
                    if int(line[0]) - 1 == testnunmber or int(line[1]) - 1 == sitenumber:
                        testnunmber = int(line[0])
                        sitenumber = int(line[1])
                    else:
                        pass

    '''
    for site in sites:
        buffers.append([[x] for x in buffer+site])
        '''

    # print(buffers)

'''    
#                           считываем все строки
lines = file1.readlines()
lines="".join(lines)
print("ENTER SEPARATOR")
separator=input()
if separator=="":separator="Datalog report\n|    Device#: |\n\n\n"
lines=re.split(separator,lines)
print(len(lines))
'''

# %%
filepath = '/Users/fedor/PycharmProjects/MC_simulation/input.txt'






