// SegMaster/segment.cpp

#include <bits/stdc++.h>

using namespace std;

struct Node
{
	int sum, pref, suff, maxf, lazy, len;
};

const int MAXN = 100010;
vector<Node> tree(4 * MAXN);
int n;

void build(int id, int l, int r)
{
	tree[id].lazy = -1;
	tree[id].len = r - l + 1;
	tree[id].sum = 0;
	tree[id].pref = tree[id].suff = tree[id].maxf = tree[id].len;
	if (l == r)
		return;
	int m = (l + r) / 2;
	build(2 * id, l, m);
	build(2 * id + 1, m + 1, r);
}

void prop(int id, int l, int r)
{
	if (tree[id].lazy == -1)
		return;
	if (tree[id].lazy == 1)
	{
		tree[id].sum = tree[id].len;
		tree[id].pref = tree[id].suff = tree[id].maxf = 0;
	}
	else
	{
		tree[id].sum = 0;
		tree[id].pref = tree[id].suff = tree[id].maxf = tree[id].len;
	}
	if (l != r)
	{
		tree[2 * id].lazy = tree[2 * id + 1].lazy = tree[id].lazy;
	}
	tree[id].lazy = -1;
}

Node merge(const Node &a, const Node &b)
{
	Node res;
	res.len = a.len + b.len;
	res.sum = a.sum + b.sum;
	res.pref = (a.pref == a.len ? a.pref + b.pref : a.pref);
	res.suff = (b.suff == b.len ? b.suff + a.suff : b.suff);
	res.maxf = max({a.maxf, b.maxf, a.suff + b.pref});
	res.lazy = -1;
	return res;
}

void update(int id, int l, int r, int ql, int qr, int val)
{
	prop(id, l, r);
	if (qr < l || ql > r)
		return;
	if (ql <= l && r <= qr)
	{
		tree[id].lazy = val;
		prop(id, l, r);
		return;
	}
	int m = (l + r) / 2;
	update(2 * id, l, m, ql, qr, val);
	update(2 * id + 1, m + 1, r, ql, qr, val);
	tree[id] = merge(tree[2 * id], tree[2 * id + 1]);
}

int get_sum(int id, int l, int r, int ql, int qr)
{
	prop(id, l, r);
	if (qr < l || ql > r)
		return 0;
	if (ql <= l && r <= qr)
		return tree[id].sum;
	int m = (l + r) / 2;
	return get_sum(2 * id, l, m, ql, qr) +
	       get_sum(2 * id + 1, m + 1, r, ql, qr);
}

int get_maxf(int id, int l, int r)
{
	prop(id, l, r);
	return tree[id].maxf;
}

int find_first(int id, int l, int r, int k)
{
	prop(id, l, r);
	if (tree[id].maxf < k)
		return -1;
	if (l == r)
		return l;
	int m = (l + r) / 2;
	prop(2 * id, l, m);
	if (tree[2 * id].maxf >= k)
		return find_first(2 * id, l, m, k);
	prop(2 * id + 1, m + 1, r);
	int cross = tree[2 * id].suff + tree[2 * id + 1].pref;
	if (cross >= k)
		return m - tree[2 * id].suff + 1;
	return find_first(2 * id + 1, m + 1, r, k);
}

int get_val(int id, int l, int r, int pos)
{
	prop(id, l, r);
	if (l == r)
		return tree[id].sum;
	int m = (l + r) / 2;
	if (pos <= m)
		return get_val(2 * id, l, m, pos);
	return get_val(2 * id + 1, m + 1, r, pos);
}

int main()
{
	ios::sync_with_stdio(false);
	cin.tie(NULL);
	string cmd;
	while (cin >> cmd)
	{
		if (cmd == "init")
		{
			cin >> n;
			build(1, 1, n);
			cout << "OK" << endl;
		}
		else if (cmd == "alloc")
		{
			int k;
			cin >> k;
			int pos = find_first(1, 1, n, k);
			if (pos == -1)
			{
				cout << "FAIL" << endl;
			}
			else
			{
				update(1, 1, n, pos, pos + k - 1, 1);
				cout << pos << endl;
			}
		}
		else if (cmd == "free")
		{
			int L, R;
			cin >> L >> R;
			update(1, 1, n, L, R, 0);
			cout << "OK" << endl;
		}
		else if (cmd == "update")
		{
			int L, R, val;
			cin >> L >> R >> val;
			update(1, 1, n, L, R, val);
			cout << "OK" << endl;
		}
		else if (cmd == "total_alloc")
		{
			cout << get_sum(1, 1, n, 1, n) << endl;
		}
		else if (cmd == "max_free")
		{
			cout << get_maxf(1, 1, n) << endl;
		}
		else if (cmd == "find_first")
		{
			int k;
			cin >> k;
			int pos = find_first(1, 1, n, k);
			cout << (pos == -1 ? "NONE" : to_string(pos)) << endl;
		}
		else if (cmd == "dump")
		{
			for (int i = 1; i <= n; ++i)
			{
				cout << get_val(1, 1, n, i);
				if (i < n)
					cout << " ";
				else
					cout << endl;
			}
		}
		else if (cmd == "reset")
		{
			update(1, 1, n, 1, n, 0);
			cout << "OK" << endl;
		}
		else if (cmd == "exit")
		{
			break;
		}
	}
	return 0;
}
