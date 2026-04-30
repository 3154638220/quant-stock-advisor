#include <bits/stdc++.h>
using namespace std;

const int N = 5e5 + 10;

int a[N], l = 1, n, k, tr[N], sum, ans;

int lowbit(int x) {return x & -x;}

void add(int x, int val)
{
    for (; x <= n; x += lowbit(x))
        tr[x] += val;
}

int query(int x)
{
    int res = 0;
    for (; x > 0; x -= lowbit(x))
        res += tr[x];
    return res;
}

int solve(int k)
{
    l = 1; sum = ans = 0;
    memset(tr, 0, sizeof(tr));
    for (int r = 1; r <= n; r++) {
        add(a[r], 1);
        sum += r - l + 1 - query(a[r]);
        while (l <= r && sum > k) {
            add(a[l], -1);
            sum -= query(a[l]);
            l++;
        }
        ans += r - l + 1;
    }
    return ans;
}

int main()
{
    cin >> n >> k;
    for (int i = 1; i <= n; i++)
        cin >> a[i];
    int x = solve(k), y = solve(k - 1);
    cout << x - y;
    return 0;
}