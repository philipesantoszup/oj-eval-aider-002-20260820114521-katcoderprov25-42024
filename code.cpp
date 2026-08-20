#include <complex>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace sjtu {

class int2048 {
  // todo
public:
  // Constructors
  int2048();
  int2048(long long);
  int2048(const std::string &);
  int2048(const int2048 &);

  // The parameter types of the following functions are for reference only, you can choose to use constant references or not
  // If needed, you can add other required functions yourself
  // ===================================
  // Integer1
  // ===================================

  // Read a big integer
  void read(const std::string &);
  // Output the stored big integer, no need for newline
  void print();

  // Add a big integer
  int2048 &add(const int2048 &);
  // Return the sum of two big integers
  friend int2048 add(int2048, const int2048 &);

  // Subtract a big integer
  int2048 &minus(const int2048 &);
  // Return the difference of two big integers
  friend int2048 minus(int2048, const int2048 &);

  // ===================================
  // Integer2
  // ===================================

  int2048 operator+() const;
  int2048 operator-() const;

  int2048 &operator=(const int2048 &);

  int2048 &operator+=(const int2048 &);
  friend int2048 operator+(int2048, const int2048 &);

  int2048 &operator-=(const int2048 &);
  friend int2048 operator-(int2048, const int2048 &);

  int2048 &operator*=(const int2048 &);
  friend int2048 operator*(int2048, const int2048 &);

  int2048 &operator/=(const int2048 &);
  friend int2048 operator/(int2048, const int2048 &);

  int2048 &operator%=(const int2048 &);
  friend int2048 operator%(int2048, const int2048 &);

  friend std::istream &operator>>(std::istream &, int2048 &);
  friend std::ostream &operator<<(std::ostream &, const int2048 &);

  friend bool operator==(const int2048 &, const int2048 &);
  friend bool operator!=(const int2048 &, const int2048 &);
  friend bool operator<(const int2048 &, const int2048 &);
  friend bool operator>(const int2048 &, const int2048 &);
  friend bool operator<=(const int2048 &, const int2048 &);
  friend bool operator>=(const int2048 &, const int2048 &);

private:
  // Digits stored little-endian, each element is a single decimal digit (0-9)
  std::vector<int> digits;
  bool sign; // false = non-negative, true = negative

  void normalize();
  bool isZero() const;
  void absAdd(const int2048 &);   // this += other, both non-negative
  void absSub(const int2048 &);   // this -= other, this >= other >= 0
  bool absLess(const int2048 &) const;
  bool absEqual(const int2048 &) const;
  int2048 absMul(const int2048 &) const;
  int2048 absDiv(const int2048 &) const;  // floor division of magnitudes (truncates toward zero for positive)
};

// ==================== Helper functions (avoiding <algorithm>/<cmath>) ====================

static void my_swap(int &a, int &b) {
    int t = a; a = b; b = t;
}

static void my_reverse_digits(std::vector<int> &v) {
    int i = 0, j = (int)v.size() - 1;
    while (i < j) { my_swap(v[i], v[j]); ++i; --j; }
}

static long long my_llround(double x) {
    return x >= 0 ? (long long)(x + 0.5) : (long long)(x - 0.5);
}

// ==================== FFT helpers ====================

using Complex = std::complex<double>;
static const double PI = 3.14159265358979323846;

static void fft(std::vector<Complex> &a, bool invert) {
    int n = (int)a.size();
    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) my_swap(a[i].real(), a[j].real()), my_swap(a[i].imag(), a[j].imag());
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        Complex wlen = std::polar(1.0, ang);
        for (int i = 0; i < n; i += len) {
            Complex w(1, 0);
            for (int k = 0; k < len / 2; ++k) {
                Complex u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (invert) {
        for (int i = 0; i < n; ++i) a[i] /= n;
    }
}

static int2048 fftMultiply(const int2048 &a, const int2048 &b) {
    if (a.isZero() || b.isZero()) return int2048(0);
    const std::vector<int> &da = a.digits;
    const std::vector<int> &db = b.digits;
    int na = (int)da.size(), nb = (int)db.size();
    int n = 1;
    while (n < na + nb) n <<= 1;

    std::vector<Complex> fa(n), fb(n);
    for (int i = 0; i < na; ++i) fa[i] = Complex(da[i], 0);
    for (int i = 0; i < nb; ++i) fb[i] = Complex(db[i], 0);

    fft(fa, false);
    fft(fb, false);
    for (int i = 0; i < n; ++i) fa[i] *= fb[i];
    fft(fa, true);

    int2048 result;
    result.sign = false;
    result.digits.resize(na + nb, 0);
    long long carry = 0;
    for (int i = 0; i < n; ++i) {
        long long val = carry + my_llround(fa[i].real());
        if (i < (int)result.digits.size()) {
            result.digits[i] = (int)(val % 10);
            carry = val / 10;
        }
    }
    result.normalize();
    return result;
}

static int2048 naiveMultiply(const int2048 &a, const int2048 &b) {
    if (a.isZero() || b.isZero()) return int2048(0);
    const std::vector<int> &da = a.digits;
    const std::vector<int> &db = b.digits;
    int2048 result;
    result.sign = false;
    result.digits.resize(da.size() + db.size(), 0);
    for (size_t i = 0; i < da.size(); ++i) {
        int carry = 0;
        for (size_t j = 0; j < db.size() || carry; ++j) {
            long long cur = result.digits[i + j] + carry +
                (long long)da[i] * (j < db.size() ? db[j] : 0);
            result.digits[i + j] = (int)(cur % 10);
            carry = (int)(cur / 10);
        }
    }
    result.normalize();
    return result;
}

// ==================== int2048 implementation ====================

int2048::int2048() : sign(false) {
    digits.push_back(0);
}

int2048::int2048(long long v) : sign(false) {
    if (v < 0) { sign = true; v = -v; }
    if (v == 0) digits.push_back(0);
    else {
        while (v > 0) { digits.push_back((int)(v % 10)); v /= 10; }
    }
}

int2048::int2048(const std::string &s) : sign(false) { read(s); }

int2048::int2048(const int2048 &other) : digits(other.digits), sign(other.sign) {}

void int2048::normalize() {
    while (digits.size() > 1 && digits.back() == 0) digits.pop_back();
    if (isZero()) sign = false;
}

bool int2048::isZero() const { return digits.size() == 1 && digits[0] == 0; }

bool int2048::absLess(const int2048 &other) const {
    if (digits.size() != other.digits.size()) return digits.size() < other.digits.size();
    for (int i = (int)digits.size() - 1; i >= 0; --i) {
        if (digits[i] != other.digits[i]) return digits[i] < other.digits[i];
    }
    return false;
}

bool int2048::absEqual(const int2048 &other) const { return digits == other.digits; }

void int2048::absAdd(const int2048 &other) {
    int carry = 0;
    size_t n = digits.size() > other.digits.size() ? digits.size() : other.digits.size();
    digits.resize(n, 0);
    for (size_t i = 0; i < n || carry; ++i) {
        if (i == digits.size()) digits.push_back(0);
        int sum = digits[i] + carry + (i < other.digits.size() ? other.digits[i] : 0);
        digits[i] = sum % 10;
        carry = sum / 10;
    }
}

void int2048::absSub(const int2048 &other) {
    int borrow = 0;
    for (size_t i = 0; i < other.digits.size() || borrow; ++i) {
        int sub = digits[i] - borrow - (i < other.digits.size() ? other.digits[i] : 0);
        if (sub < 0) { sub += 10; borrow = 1; }
        else borrow = 0;
        digits[i] = sub;
    }
    normalize();
}

void int2048::read(const std::string &s) {
    sign = false;
    digits.clear();
    size_t start = 0;
    if (start < s.size() && s[start] == '-') { sign = true; ++start; }
    while (start < s.size() && s[start] == '0') ++start;
    if (start >= s.size()) { digits.push_back(0); sign = false; return; }
    for (int i = (int)s.size() - 1; i >= (int)start; --i)
        digits.push_back(s[i] - '0');
    normalize();
}

void int2048::print() {
    if (sign && !isZero()) std::cout << '-';
    for (int i = (int)digits.size() - 1; i >= 0; --i)
        std::cout << digits[i];
}

int2048 &int2048::add(const int2048 &other) {
    if (sign == other.sign) {
        absAdd(other);
    } else {
        if (absLess(other)) {
            int2048 tmp = other;
            tmp.absSub(*this);
            *this = tmp;
            sign = other.sign;
        } else {
            absSub(other);
        }
    }
    return *this;
}

int2048 add(int2048 a, const int2048 &b) { a.add(b); return a; }

int2048 &int2048::minus(const int2048 &other) {
    int2048 neg = other;
    neg.sign = !neg.sign;
    if (neg.isZero()) neg.sign = false;
    return add(neg);
}

int2048 minus(int2048 a, const int2048 &b) { a.minus(b); return a; }

int2048 int2048::operator+() const { return *this; }

int2048 int2048::operator-() const {
    int2048 r = *this;
    if (!r.isZero()) r.sign = !r.sign;
    return r;
}

int2048 &int2048::operator=(const int2048 &other) {
    digits = other.digits;
    sign = other.sign;
    return *this;
}

int2048 &int2048::operator+=(const int2048 &other) { return add(other); }

int2048 operator+(int2048 a, const int2048 &b) { return a.add(b), a; }

int2048 &int2048::operator-=(const int2048 &other) { return minus(other); }

int2048 operator-(int2048 a, const int2048 &b) { return a.minus(b), a; }

int2048 int2048::absMul(const int2048 &other) const {
    size_t thresh = 64;
    if (digits.size() <= thresh || other.digits.size() <= thresh)
        return naiveMultiply(*this, other);
    return fftMultiply(*this, other);
}

int2048 &int2048::operator*=(const int2048 &other) {
    if (isZero() || other.isZero()) { digits = {0}; sign = false; return *this; }
    sign = (sign != other.sign);
    *this = absMul(other);
    return *this;
}

int2048 operator*(int2048 a, const int2048 &b) { return a *= b, a; }

int2048 int2048::absDiv(const int2048 &other) const {
    if (other.isZero()) return int2048(0);
    if (absLess(other)) return int2048(0);
    if (absEqual(other)) return int2048(1);

    // Precompute other * 1..9 for fast quotient digit estimation
    std::vector<std::vector<int>> multiples(10);
    multiples[0] = {0};
    for (int d = 1; d <= 9; ++d) {
        std::vector<int> &cur = multiples[d];
        cur.clear();
        int c2 = 0;
        for (size_t k = 0; k < other.digits.size() || c2; ++k) {
            long long v = c2 + (long long)(k < other.digits.size() ? other.digits[k] : 0) * d;
            cur.push_back((int)(v % 10));
            c2 = (int)(v / 10);
        }
    }

    std::vector<int> q_digits;
    std::vector<int> rem;

    for (int i = (int)digits.size() - 1; i >= 0; --i) {
        // rem = rem * 10 + digits[i]
        int carry = digits[i];
        for (size_t j = 0; j < rem.size() || carry; ++j) {
            if (j == rem.size()) rem.push_back(0);
            long long cur = (long long)rem[j] * 10 + carry;
            rem[j] = (int)(cur % 10);
            carry = (int)(cur / 10);
        }
        // Remove trailing zeros from rem (which are leading zeros in big-endian view)
        while (rem.size() > 1 && rem.back() == 0) rem.pop_back();

        // Find largest d such that other*d <= rem
        int best = 0;
        for (int d = 9; d >= 1; --d) {
            const std::vector<int> &prod = multiples[d];
            if (prod.size() > rem.size()) continue;
            if (prod.size() < rem.size()) { best = d; break; }
            // same length, compare
            bool le = true;
            for (int k = (int)prod.size() - 1; k >= 0; --k) {
                if (prod[k] != rem[k]) { le = prod[k] < rem[k]; break; }
            }
            if (le) { best = d; break; }
        }
        q_digits.push_back(best);

        // rem = rem - other*best
        if (best > 0) {
            const std::vector<int> &prod = multiples[best];
            int borrow = 0;
            for (size_t k = 0; k < prod.size() || borrow; ++k) {
                int sub = rem[k] - borrow - (k < prod.size() ? prod[k] : 0);
                if (sub < 0) { sub += 10; borrow = 1; }
                else borrow = 0;
                rem[k] = sub;
            }
            while (rem.size() > 1 && rem.back() == 0) rem.pop_back();
        }
    }

    // q_digits were collected MSD-first, reverse to LSD-first
    my_reverse_digits(q_digits);

    int2048 result;
    result.sign = false;
    result.digits = q_digits;
    result.normalize();
    return result;
}

int2048 &int2048::operator/=(const int2048 &other) {
    if (isZero()) return *this;
    bool neg = (sign != other.sign);
    // Compute magnitude quotient
    int2048 self_abs = *this;
    self_abs.sign = false;
    int2048 other_abs = other;
    other_abs.sign = false;
    int2048 q = self_abs.absDiv(other_abs);

    // Compute remainder magnitude to check if division is exact
    int2048 prod = q;
    prod.sign = false;
    prod = prod.absMul(other_abs);
    int2048 rem = self_abs;
    rem.sign = false;
    rem.absSub(prod);
    bool hasRem = !rem.isZero();

    // Floor division: round toward -inf
    // If signs differ and there's a remainder, we need q+1 (then negate)
    if (neg && hasRem) {
        q.add(int2048(1));
    }
    q.sign = neg;
    *this = q;
    return *this;
}

int2048 operator/(int2048 a, const int2048 &b) { return a /= b, a; }

int2048 &int2048::operator%=(const int2048 &other) {
    int2048 q = *this;
    q /= other;
    int2048 prod = q;
    prod *= other;
    *this = minus(*this, prod);
    return *this;
}

int2048 operator%(int2048 a, const int2048 &b) { return a %= b, a; }

std::istream &operator>>(std::istream &is, int2048 &n) {
    std::string s;
    is >> s;
    n.read(s);
    return is;
}

std::ostream &operator<<(std::ostream &os, const int2048 &n) {
    n.print();
    return os;
}

bool operator==(const int2048 &a, const int2048 &b) {
    return a.sign == b.sign && a.digits == b.digits;
}

bool operator!=(const int2048 &a, const int2048 &b) { return !(a == b); }

bool operator<(const int2048 &a, const int2048 &b) {
    if (a.sign != b.sign) return a.sign;
    if (a.sign) {
        if (a.digits.size() != b.digits.size()) return a.digits.size() > b.digits.size();
        for (int i = (int)a.digits.size() - 1; i >= 0; --i) {
            if (a.digits[i] != b.digits[i]) return a.digits[i] > b.digits[i];
        }
        return false;
    } else {
        if (a.digits.size() != b.digits.size()) return a.digits.size() < b.digits.size();
        for (int i = (int)a.digits.size() - 1; i >= 0; --i) {
            if (a.digits[i] != b.digits[i]) return a.digits[i] < b.digits[i];
        }
        return false;
    }
}

bool operator>(const int2048 &a, const int2048 &b) { return b < a; }

bool operator<=(const int2048 &a, const int2048 &b) { return !(b < a); }

bool operator>=(const int2048 &a, const int2048 &b) { return !(a < b); }

} // namespace sjtu
