# import math


# def get_factors(n):
#     return sum(2 for i in range(1, round(math.sqrt(n) + 1)) if not n % i)


# tst = get_factors(44)

# print(tst)

# hum = {"1": 2, "3": 4}
# print(len(list(hum.keys())))


# import math


# class Circle:
#     def __init__(self, radius):
#         self.radius = radius

#     @property
#     def radius(self):
#         return self._radius

#     @radius.setter
#     def radius(self, value):
#         self._radius = float(value)

#     @property
#     def diameter(self):
#         return self.radius * 2

#     @diameter.setter
#     def diameter(self, value):
#         self.radius = value / 2


# circle = Circle(42)

# print(circle.radius)
# print(circle.diameter)
# circle.diameter = 100
# print(circle.radius)
# circle.radius = 10
# print(circle.diameter)

# import string

# print(string.ascii_letters)

# test_list = [1, 5, 3, 7, 93, 5]  # second largest is 7


# def second_largest():

#     test_sorted = sorted(test_list.copy())
#     value = test_list.index(test_sorted[-2])

#     return value


# your_val = second_largest()
# print(your_val)

# memo = "thing"

# for i in memo:
#     print(f"i: {i} ::  Index of i: {memo.index(i)}")


# x = [3, 3, 3, 1, 0, 2]
# y = sorted(list(set(x)))

# while len(y) != len(x):
#     y.insert(0, 0)
# print(y)


# list1 = ["A", "A", "B", "A", "B"]

# list2 = ["A", "B", "A", "CAN"]

# diff = []


# for item in list1:

#     if list1.count(item) > list2.count(item):
#         if diff.count(item) < (list1.count(item) - list2.count(item)):
#             diff.append(item)

# for item in list2:

#     if list2.count(item) > list1.count(item):
#         if diff.count(item) < (list2.count(item) - list1.count(item)):
#             diff.append(item)

# print("F", diff)


# x = "MCMXCIV"
# y = "III"


# def romanToInt(s: str) -> int:
#     start_value = 0
#     final_value = 0
#     roman_values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

#     for letter in s:
#         if roman_values[letter] < start_value:
#             final_value -= roman_values[letter]
#         else:
#             final_value += roman_values[letter]
#         start_value = roman_values[letter]
#     return final_value


# my_func = romanToInt(x)


# roman_values_len = {
#     "I": [1, 1],
#     "V": [5, 1],
#     "X": [10, 2],
#     "L": [50, 2],
#     "C": [100, 3],
#     "D": [500, 3],
#     "M": [1000, 4],
# }
# e = [
#     "",
#     "C",
#     "CC",
#     "CCC",
#     "CD",
#     "D",
#     "DC",
#     "DCC",
#     "DCCC",
#     "CM",
#     "",
#     "X",
#     "XX",
#     "XXX",
#     "XL",
#     "L",
#     "LX",
#     "LXX",
#     "LXXX",
#     "XC",
#     "",
#     "I",
#     "II",
#     "III",
#     "IV",
#     "V",
#     "VI",
#     "VII",
#     "VIII",
#     "IX",
# ]


# def intToRoman(num: int) -> str:
#     final_value = ""

#     roman_lists_dict = {
#         "Thousands": ["", "M", "MM", "MMM"],
#         "Hundreds": ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"],
#         "Tens": ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"],
#         "Ones": ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"],
#     }

#     M = roman_lists_dict["Thousands"][num // 1000]
#     C = roman_lists_dict["Hundreds"][(num % 1000) // 100]
#     X = roman_lists_dict["Tens"][(num % 100) // 10]
#     IV = roman_lists_dict["Ones"][num % 10]

#     final_list = [M, C, X, IV]
#     for letter in final_list:
#         if letter == "":
#             pass
#         else:
#             final_value += letter

#     return final_value


# rmn_int = intToRoman(3549)
# # print(rmn_int)


# def twoSum(nums: list[int], target: int) -> list[int]:

#     z = 0
#     start = 1
#     stop = True

#     while stop:
#         for index, num in enumerate(nums[start::]):
#             final_list = [z]

#             if nums[z] + num == target:
#                 final_list.append(index + start)

#                 return final_list

#         z += 1
#         start += 1
#         stop -= 1


# two = twoSum([2, 5, 11, 6, 5], 10)
# # print(two)


# def lengthOfLongestSubstring(s: str) -> int:
#     zero_one = len(s)
#     final_len = 0

#     if zero_one == 0:
#         return 0
#     elif zero_one == 1:
#         return 1
#     else:
#         sub_str = ""
#         for letter in s:
#             if letter not in sub_str:
#                 sub_str += letter
#                 if len(sub_str) > final_len:
#                     final_len = len(sub_str)

#             else:
#                 sub_str = sub_str[(sub_str.index(letter) + 1) : :] + letter

#     return final_len


# # test = lengthOfLongestSubstring("au")
# # print(test)


# def findMedianSortedArrays(nums1: list[int], nums2: list[int]) -> float:

#     combined = sorted(nums1 + nums2)
#     print(combined)

#     com_len = len((combined))
#     print(com_len)

#     if com_len % 2 == 0:
#         com_len = len((combined)) - 1
#         x = int(com_len / 2)
#         print(f"x: {x}")
#         y = x + 1
#         print(f"y: {y}")

#         # print(int(com_len))

#         return float((combined[x] + combined[y]) / 2)

#     else:
#         x = int(com_len / 2)
#         print(combined[x])
#         return combined[x]


# med_of_list = findMedianSortedArrays([1, 3, 5], [2, 6, 7])
# # print(med_of_list)


# def longestPalindrome(s: str) -> str:

#     # palendrome = sorted(
#     #     {
#     #         s[x:y]
#     #         for x in range(len(s))
#     #         for y in range(x + 1, len(s) + 1)
#     #         if s[x:y][0::] == s[x:y][::-1]
#     #     },
#     #     key=len,
#     # )[-1]
#     palindrome = ""

#     # loop through the input string
#     for i in range(len(s)):

#         # loop backwards through the input string
#         for j in range(len(s), i, -1):

#             # Break if out of range
#             if len(palindrome) >= j - i:
#                 break

#             # Update variable if matches
#             elif s[i:j] == s[i:j][::-1]:
#                 palindrome = s[i:j]
#                 break

#     return palindrome


# pal_check = longestPalindrome("abababaaaa")
# # print(pal_check)


# def revverse(x: int) -> int:
#     str_x = str(x)
#     if x < 0:

#         rev_str = str_x[::-1]
#         drop_neg = rev_str[:-1]
#         final = int("-" + drop_neg)
#         if final <= -2147483648:
#             return 0
#         else:
#             return int("-" + drop_neg)

#     else:
#         for_str = int(str_x[::-1])
#         if for_str >= 2147483647:
#             return 0
#         else:
#             return int(for_str)


# str_xt = revverse(2147483647)
# # print(str_xt)


# def myAtoi(s: str) -> int:
#     iter_str = ""
#     s = s.strip()

#     if s == "":
#         return 0

#     for letter in s:
#         if letter == "+":
#             if iter_str == "":
#                 print(f'"": iter: {iter_str}  letter: {letter}')
#                 if s[s.index(letter) + 1].isdigit():
#                     print(f"Next:  {s[s.index(letter) + 1]}")
#                     print(f"Digit: iter: {iter_str}  letter: {letter}")
#                     pass
#                 elif s[s.index(letter) + 1] == "-":
#                     print(f"Next2:  {s[s.index(letter) + 1]}")
#                     print(f'"-": iter: {iter_str}  letter: {letter}')
#                     pass
#                 else:
#                     return 0

#             elif iter_str[0] == "-":
#                 return 0
#             else:
#                 if iter_str[0].isdigit():
#                     break
#         if letter == ".":
#             if iter_str == "":
#                 return 0
#             else:
#                 break

#         if letter.isalpha() and (iter_str == ""):
#             return 0
#         if letter == "-" and iter_str == "":
#             iter_str += letter
#         if letter.isdigit():
#             iter_str += letter
#         if (iter_str != "") and (letter.isdigit() is False) and (iter_str != "-"):
#             break

#     if iter_str == "-":
#         return 0
#     else:
#         final = int(iter_str)

#     if final > 2147483648:
#         final = 2147483648
#     if final < -2147483648:
#         final = -2147483648

#     return final


# my_test_Atoi = myAtoi("+-12")
# # print(my_test_Atoi)


# def longestCommonPrefix(strs: list[str]) -> str:
#     final = ""
#     count = 0

#     if not strs:
#         return ""

#     if len(strs) == 1:
#         return "".join(strs)

#     while True:
#         try:
#             temp_list = []
#             for x in strs:
#                 temp_list.append(x[count])
#             if len("".join(set(temp_list))) == 1:
#                 final += x[count]
#                 count += 1
#             else:
#                 break
#         except IndexError:
#             break

#     return final


# LCP = longestCommonPrefix(["abaaa", "abababa", "abc"])
# print(f"LCP {LCP}")

# from itertools import permutations


# def findSubstring(s: str, words: list[str]) -> list[int]:
#     word_len = len("".join(words))
#     mixed_words = ["".join(x) for x in permutations(words)]
#     mixed = set(mixed_words)

#     final = [i for (i, j) in enumerate(s) if s[i : (word_len + i)] in mixed]

#     return final


# FS = findSubstring("barfoofoobarthefoobarman", ["bar", "foo", "the"])
# print(f"FS2: {FS}")


# def numMovesStones(a: int, b: int, c: int) -> list[int]:

#     temp_list = sorted([a, b, c])
#     x = temp_list[0]
#     y = temp_list[1]
#     z = temp_list[2]

#     final_list = [0, 0]

#     if (x + 1) < y:
#         final_list[0] = final_list[0] + 1
#         final_list[1] = final_list[1] + (y - (x + 1))

#     if (y + 1) < z:
#         final_list[0] = final_list[0] + 1
#         final_list[1] = final_list[1] + (z - (y + 1))

#     if ((z - (y + 1)) == 1) or ((y - (x + 1)) == 1):
#         final_list[0] = 1

#     return final_list


# NMS = numMovesStones(3, 5, 1)
# print(f"NMS: {NMS}")


# There are n bulbs that are initially off.
# You first turn on all the bulbs,
# then you turn off every second bulb.
# On the third round, you toggle every third bulb (turning on if it's off or turning off if it's on).
# For the ith round, you toggle every i bulb.
# For the nth round, you only toggle the last bulb.

# Return the number of bulbs that are on after n rounds.


# def bulbSwitch(n: int) -> int:
#     bulbs = [True] * n
#     print(bulbs)
#     for off in bulbs[0::2]:
#         bulbs[off] = False
#     print(bulbs)
#     for i, on_off in enumerate(bulbs):
#         print(i)
#         if i % 3 == 0:
#             print(i)
#             if bulbs[i] is True:
#                 bulbs[i] = False

#     print(bulbs)


# BS = bulbSwitch(3)
# print(f"BS: {BS}")


# def isValid(s: str) -> bool:
#     s_len = len(s)
#     half = s_len // 2
#     left = ["(", "[", "{"]
#     right = [")", "]", "}"]
#     test_left = []
#     test_right = []

#     if s_len % 2 == 1:
#         return False

#     if s[0] in right:
#         return False

#     if s[-1] in left:
#         return False

#     count = 0
#     for item in s[1::]:
#         if item in left:
#             count += 1


# IS = isValid("()[]{[]}")
# print(IS)
