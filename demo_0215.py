# import torch
# a = torch.tensor(([1],[1]))
# print(a.size())
# b = torch.squeeze(a,dim=1)
# print(b.size())
# c = torch.unsqueeze(b,dim=0)
# print(c.size())
#
# d = torch.randn(2,3,4,5)
# print(d)
# print(d.shape)
# d = d.reshape(-1,2,3,4)
# print(d.shape)

import collections
from collections import deque

# a = deque([1,2,3,4,5])
# # print(type(a))
# a.rotate(3)
# a.appendleft(-1)
# a.pop()
# print
# b = deque()
# a = [3,9,2,None,None,1,7]
# b.append(a)
# print(b)
# a = 'we are happy'
# res = list(a)
# res.extend([' ']*4)
# print(res)
# def del_space(a):
#
#     nums = list(a)
#     fast = 0
#     slow = 0
#     while fast < len(nums):
#         if nums[fast] != ' ':
#             nums[slow] = nums[fast]
#             slow += 1
#             if nums[fast+1] == ' ':
#                 nums [slow] = ' '
#                 slow += 1
#         fast += 1
#     return nums[:slow]
# a = '  the sky is    blue '
# def merge(a,b):
#     slow = 0
#     fast = 0
#     res = []
#     while slow < len(a) and fast < len(b):
#         if a[slow] < b[fast]:
#             res.append((a[slow]))
#             if slow == len(a)-1:
#                 res += b[len(a):len(b)]
#             slow += 1
#         elif a[slow] > b[fast]:
#             res.append((b[fast]))
#             if fast == len(b)-1 and b[fast] > a[len(b)-1]:
#                 res += a[len(b):len(a)]
#             elif fast == len(b)-1 and b[fast] < a[len(b)-1]:
#                 res += a[len(b)-1:len(a)]
#             fast += 1
#         else :
#             res. append((a[slow]))
#             if slow == len(a)-1 and a[slow] != b[len(b)-1]:
#                 res.append((b[len(b)-1]))
#             slow += 1
#             if fast == len(b)-1 and b[fast] != a[len(a)-1]:
#                 res.append((a[len(a)-1]))
#             fast += 1
#     print(res)
# a = [0,1,3,5,6,7,8,10,11]
# b = [2,3,4,5,6,7,8,10,13]
# merge(a,b)

#查找 数组中 重复的数字
# def search_dubble(nums):
#     nums.sort()
#     n = len(nums)
#     slow = 0
#     fast = 1
#     while fast < n:
#         if nums[slow] != nums[fast]:
#             fast += 1
#             slow += 1
#         else:
#             return nums[slow]
#     return None

# def remove_dubble(nums):
#     nums.sort
#     n = len(nums)
#     slow = 0
#     fast = 0
#     tmp = fast+1
#     while tmp:
#         if nums[fast] != nums[tmp]:
#             nums[slow] = nums[fast]
#             slow += 1
#         fast += 1
#     return nums[:]
#
# nums = [0,1,2,3,4,5,7,7,8,9]
# # print(search_dubble(nums))
# print(remove_dubble(nums))

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def reverseBetween(self, head:[ListNode], left: int, right: int):
        dummy = ListNode(next = head)
        p0 = dummy
        for _ in range(left-1):
            p0 = p0.next
        pre = None
        cur = p0.next
        for _ in range(right-left+1):
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        p0.next.next = cur
        p0.next = pre
        return dummy.next
print(Solution.reverseBetween(self=Solution,head = ListNode((1,2,3,4,5)),left=2,right=4))