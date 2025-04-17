
#!/usr/bin/env python3
#coding=utf-8

'''
# 狗屁东西, 脑残才会想出这种问题
把阿拉伯数字转换成中文表示形式.
比如:
输入: 123456789
输出: 一亿二千零四十五万零七百八十九
'''

# 基础问题: 如何用中文表示一万以内的数字
def to_chinese(num):
    if num == 0:
        return "零"
    units = ["", "十", "百", "千"]
    chinese_digits = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
    digits = []
    # from LSB to MSB
    num_b = num
    while num_b != 0:
        remainder = num_b %10
        digits.append(remainder)
        num_b  = num_b // 10
    digits = [(i,d) for i,d in enumerate(digits)]
    # from MSB to LSM
    digits.reverse()
    repr = []
    forward_is_zero = False
    for idx, (unit_i, digit_i) in enumerate(digits):
        if digit_i == 0: # skip continuous 0s
            forward_is_zero = True
            continue
        if forward_is_zero:
            forward_is_zero = False
            repr.append(chinese_digits[0])
        if idx==0 and unit_i==1 and digit_i==1: # for case 10, 11, 12, ...
            pass
        else:
            repr.append(chinese_digits[digit_i])
        repr.append(units[unit_i])
    repr = "".join(repr)
    #print(f"{num}: {repr}")
    return repr


def number_to_chinese(num):
    if num == 0:
        return "零"
    iter = 0
    iteration_units = ["", "万", "亿"]
    subs = []
    # in LSB order
    num_b= num
    while num_b > 0:
        remainder = num_b % 10000
        subs.append((remainder, iteration_units[iter]))
        num_b = num_b // 10000
        iter += 1
    # LSB to MSB
    subs.reverse()
    repr = []
    forward_is_zero = False
    for idx, (n, u) in enumerate(subs):
        if n==0:
            forward_is_zero = True
            continue
        if forward_is_zero:
            forward_is_zero = False
            repr.append("零")
        elif idx!=0 and n<1000: # 狗屁的表示法, SB
            repr.append("零")
        # 解决每个子问题
        repr.append(to_chinese(n)+u)
    repr = "".join(repr)
    print(f"{num}: {repr}")
    return repr


number_to_chinese(1)
number_to_chinese(10)
number_to_chinese(11)
number_to_chinese(12)
number_to_chinese(111)
number_to_chinese(1111)
number_to_chinese(100)
number_to_chinese(1000)
number_to_chinese(1001)
number_to_chinese(1010)
number_to_chinese(1011)
number_to_chinese(110)
number_to_chinese(1100)
number_to_chinese(1110)
number_to_chinese(11100)
number_to_chinese(111000)
number_to_chinese(1110000)
number_to_chinese(100000000)
number_to_chinese(100000001)
number_to_chinese(100000111)
number_to_chinese(100000011)
number_to_chinese(100000010)
number_to_chinese(110000000)
number_to_chinese(101000000)
number_to_chinese(100010000)
number_to_chinese(100010011)
number_to_chinese(123456789)
number_to_chinese(123450789)
number_to_chinese(120450789)
