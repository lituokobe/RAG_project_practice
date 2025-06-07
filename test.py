from collections import Counter

nums1 = [4,9,5]
nums2 =[9,4,9,8,4]
print(Counter(nums1))
output = []
for nums in nums1:
    if nums in nums2:
        nums2.remove(nums)
        output.append(nums)

print(output)