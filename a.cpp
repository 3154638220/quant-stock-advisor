#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

ll a[30], sum = 1, n, ans, lst, tot;

int main()
{
    cin >> n;
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
        a[i] <<= 1;
    }
    for (int i = 0; i < (1 << n); i++) {
        sum = 1; tot = 0;
        for (int j = 1; j <= n; j++) {
            lst = sum;
            if ((i >> (j - 1)) & 1)
                sum += a[i];
            else
                sum -= a[i];
            if (lst * sum < 0)
                ++tot;
        }
        ans = max(ans, tot);
    }
    cout << ans;
    return 0;
}