> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/qq_21201267/article/details/97308533)

### 文章目录

*   *   [1. 问题描述](#1__1)
    *   [2. 解题思路](#2__5)
    *   *   [2.1 动态规划](#21__16)
        *   [2.2 二分查找](#22__80)

1. 问题描述
-------

有一个数字序列包含 n 个不同的数字，如何求出这个序列中的最长递增子序列长度？比如 2，9，3，6，5，1，7 这样一组数字序列，它的最长递增子序列就是 2，3，5，7，所以最长递增子序列的长度是 4。  
https://leetcode-cn.com/problems/longest-increasing-subsequence/

2. 解题思路
-------

*   类似题目：  
    [山谷序列（DP）](https://blog.csdn.net/qq_21201267/article/details/108895314)  
    [LeetCode 5559. 得到山形数组的最少删除次数（最长上升子序 DP nlogn）](https://michael.blog.csdn.net/article/details/110322462)  
    [程序员面试金典 - 面试题 17.08. 马戏团人塔（最长上升子序 DP / 二分查找）](https://michael.blog.csdn.net/article/details/105360463)  
    [LeetCode 354. 俄罗斯套娃信封问题（最长上升子序 DP / 二分查找）](https://michael.blog.csdn.net/article/details/105370146)  
    [LeetCode 368. 最大整除子集（DP）](https://michael.blog.csdn.net/article/details/106816075)  
    [程序员面试金典 - 面试题 08.13. 堆箱子（DP）](https://michael.blog.csdn.net/article/details/105538496)  
    [LeetCode 673. 最长递增子序列的个数（DP）](https://michael.blog.csdn.net/article/details/106677852)  
    [LeetCode 1027. 最长等差数列（DP）](https://michael.blog.csdn.net/article/details/106820302)  
    [LeetCode 5545. 无矛盾的最佳球队（最大上升子序 DP）](https://michael.blog.csdn.net/article/details/109147451)

### 2.1 动态规划

*   假设在包含 i-1 下标数字时的最大递增子序列长度为 maxLen（i-1），那么下标为 i 时的 maxLen（i）需要考虑前面所有的状态，
*   如果 a[j] < a[i] （0 <= j < i），则 maxlen[i] = max(maxlen[j]+1 | （0 <= j < i）);
*   如果 a[j] >= a[i] （0 <= j < i），则 maxlen[i] = 1;

借一张动图说明  
![](https://img-blog.csdnimg.cn/20190803000703610.gif)  
![](https://img-blog.csdnimg.cn/20190803000235414.png)

```
class Solution 
{
public:
    int lengthOfLIS(vector<int>& nums) 
    {
        int n = nums.size();
        if(n == 0)
            return 0;
        int maxlen[n], ans;
        int i, j;
        for(i = 0; i < n; ++i)
            maxlen[i] = 1;//至少为1，自己
        for(i = 1; i < n; ++i)
        {
        	ans = 1;
            for(j = 0; j < i; ++j)
            {
            	if(nums[i] > nums[j] && maxlen[j]+1 > ans)
            	{
            		ans = maxlen[j]+1;
            		maxlen[i] = ans;
            	} 
        	}
        }
        for(ans = 1, i = 0; i < n; ++i)
        {
        	if(maxlen[i] > ans)//取最大值
        		ans = maxlen[i];
        }
        return ans;
    }
};
```

```
class Solution {	//2020.3.14
public:
    int lengthOfLIS(vector<int>& nums) {
        if(nums.size() == 0)
            return 0;
        int i, j, n = nums.size(),maxlen = 1;
        vector<int> dp(n,1);
        for(i = 1; i < n; ++i)
        {
            for(j = i-1; j >= 0; --j)
            {
                if(nums[i] > nums[j])
                    dp[i] = max(dp[i], dp[j]+1);
            }
            maxlen = max(maxlen, dp[i]);
        }
        return maxlen;
    }  
};
```

### 2.2 二分查找

*   参考官方的解答
*   dp[i] 表示长度为 i+1 的子序的最后一个元素的 最小数值
*   遍历每个 nums[i]，找到其在 dp 数组中的位置（大于等于 nums[i] 的第一个数），将他替换成较小的

以输入序列 [0, 8, 4, 12, 2] 为例：

第一步插入 0，dp = [0]

第二步插入 8，dp = [0, 8]

第三步插入 4，dp = [0, 4]

第四步插入 12，dp = [0, 4, 12]

第五步插入 22，dp = [0, 2, 12]

```
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        if(nums.size() == 0)
            return 0;
        int i, l, r, n = nums.size(), maxlen = 1, idx;
        vector<int> dp(n);
        dp[0] = nums[0];
        for(i = 1; i < n; ++i)//遍历每个数
        {
            l = 0, r = maxlen-1;
            idx = bs(dp,l,maxlen,nums[i],maxlen);
			//二分查找nums[i] 在dp中的位置
            if(idx == maxlen)//nums[i] 是最大的
            {
                dp[idx] = nums[i];
                maxlen++;
            }
            else//不是最大的，更新 dp[i] 里的数为较小的
                dp[idx] = min(dp[idx], nums[i]);
        }
        return maxlen;
    }  

    int bs(vector<int> &dp, int l, int r, int& target, int& maxlen)
    {	//二分查找nums[i] 在dp中的位置， 第一个大于等于 nums[i] 的
        int mid;
        while(l <= r)
        {
            mid = l + ((r-l)>>1);
            if(dp[mid] < target)
                l = mid+1;
            else
            {
                if(mid == 0 || dp[mid-1] < target)
                    return mid;
                else
                    r = mid-1;
            }
        }
        return maxlen;//没有找到，nums[i] 最大，放最后
    }
};
```

*   基于上面的想法，直接用 treeset 可以简化代码

```
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        if(nums.size() == 0)
            return 0;
        set<int> s;
        for(auto& n : nums)
        {
            if(s.count(n))
                continue;
            else
            {
                auto it = s.upper_bound(n);//n的上界
                if(it == s.end())//没有比我大的
                    s.insert(n);
                else//有比我大的
                {
                    s.erase(it);//删除比我大的
                    s.insert(n);//换成我
                }
            }
        }
        return s.size();
    }
};
```

![](https://img-blog.csdnimg.cn/20200314180540360.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxMjAxMjY3,size_16,color_FFFFFF,t_70)