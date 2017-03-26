import math
from bitstring import BitString
from fractions import gcd
from collections import deque


fibonacci_list = {}


def sieve_of_atkin(limit):
    primes = [2, 3]
    is_prime = BitString(limit+1)
    for x in range(1, int(math.sqrt(limit))+1):
        for y in range(1, int(math.sqrt(limit))+1):
            n = 4*x**2 + y**2
            if n <= limit and (n % 12 == 1 or n % 12 == 5):
                is_prime[n] = not is_prime[n]
            n = 3*x**2 + y**2
            if n <= limit and n % 12 == 7:
                is_prime[n] = not is_prime[n]
            n = 3*x**2 - y**2
            if x > y and n <= limit and n % 12 == 11:
                is_prime[n] = not is_prime[n]
    for x in range(5, int(math.sqrt(limit))):
        if is_prime[x]:
            for y in range(x**2, limit+1, x**2):
                is_prime[y] = False
    for p in range(5, limit):
        if is_prime[p]:
            primes.append(p)
    print 'Found primes till %d using Sieve of Atkin' % limit
    return primes


def is_prime(n):
    if n == 1:
        return False
    elif n in [2, 3, 5, 7]:
        return True
    elif n % 2 == 0 or n % 3 == 0:
        return False
    else:
        r = int(math.sqrt(n))
        f = 5
        while f <= r:
            if n % f == 0:
                return False
            if n % (f + 2) == 0:
                return False
            f = f+6
        return True


def lcm(a, b):
    gcd, tmp = a, b
    while tmp != 0:
        gcd, tmp = tmp, gcd % tmp
    return a * b / gcd


def sum_of_squares(n):
    return (n * (n + 1) * ((2 * n) + 1)) / 6


def sum_of_numbers(n):
    return (n * (n + 1)) / 2


def sum_digits(n):
    s = 0
    while n:
        s += n % 10
        n /= 10
    return s


def sum_power_of_digits(n, p):
    s = 0
    while n:
        s += (n % 10) ** p
        n /= 10
    return s


def product_digits(n):
    p = 1
    while n:
        p *= n % 10
        n /= 10
    return p


def euclid_triplet(S):
    s = S / 2
    pt = []
    for m in range(2, int(math.sqrt(s) - 1)):
        if s % m == 0:
            sm = s / m
            while sm % 2 == 0:
                sm /= 2
            if m % 2 == 1:
                k = m + 2
            else:
                k = m + 1
            while k < 2 * m and k <= s * m:
                if sm % k == 0 and gcd(k, m) == 1:
                    d = s / (k * m)
                    n = k - m
                    a = d * (m * m - n * n)
                    b = 2 * d * m * n
                    c = d * (m * m + n * n)
                    pt.append([a, b, c])
                k += 2
    return pt


def factors(n):
        step = 2 if n % 2 else 1
        return set(reduce(list.__add__, ([i, n//i] for i in range(1, int(math.sqrt(n))+1, step) if n % i == 0)))


def prime_factors(n):
    i = 2
    factors = set()
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.add(i)
    if n > 1:
        factors.add(n)
    return factors


def multiply_2x2_matrix(A, B):
    M11 = A[0][0] * B[0][0] + A[0][1] * B[1][0]
    M12 = A[0][0] * B[0][1] + A[0][1] * B[1][1]
    M21 = A[1][0] * B[0][0] + A[1][1] * B[1][0]
    M22 = A[1][0] * B[0][1] + A[1][1] * B[1][1]
    r = [[M11, M12], [M21, M22]]
    return r


def find_matrix_power(M, p):
    if p == 1:
        return M
    if p in fibonacci_list:
        return fibonacci_list[p]
    R = find_matrix_power(M, int(p/2))
    Z = multiply_2x2_matrix(R, R)
    fibonacci_list[p] = Z
    return Z


def get_fibonacci_number(num):
    F = [[1, 1],
         [1, 0]]
    if num == 0 or num == 1:
        return 1
    powers = [int(pow(2, b)) for (b, d) in enumerate(reversed(bin(num-1)[2:])) if d == '1']
    mats = [find_matrix_power(F, p) for p in powers]
    while len(mats) > 1:
        M1 = mats.pop()
        M2 = mats.pop()
        R = multiply_2x2_matrix(M1, M2)
        mats.append(R)
    return mats[0][0][0]


def sum_of_pyramid(pyramid):
    for p in range((len(pyramid)-1), 0, -1):
        for q in range((len(pyramid[p])-2), -1, -1):
            if pyramid[p-1][q] + pyramid[p][q+1] > pyramid[p-1][q] + pyramid[p][q]:
                pyramid[p-1][q] = pyramid[p-1][q] + pyramid[p][q+1]
            else:
                pyramid[p-1][q] = pyramid[p-1][q] + pyramid[p][q]
    return pyramid[0][0]


def check_leap_year(year):
    if (year % 4 == 0):
        if (year % 100 == 0) and (year % 400 == 0):
            return True
        elif (year % 100 == 0) and (year % 400 != 0):
            return False
        else:
            return True
    else:
        return False


def get_the_day(date):
    start_date = [1, 1, 1901]
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    no_of_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    dd_mm_yyyy = date.split('-')
    for i in range(0, 3):
        dd_mm_yyyy[i] = int(dd_mm_yyyy[i])
    if dd_mm_yyyy[1] > 2 and check_leap_year(dd_mm_yyyy[2]):
        no_of_days[1] = 29
    total_days = 0
    if dd_mm_yyyy[2] > start_date[2]:
        for y in range(start_date[2], dd_mm_yyyy[2]):
            if check_leap_year(y):
                total_days += 366
            else:
                total_days += 365

    for d in range(0, (dd_mm_yyyy[1] - 1)):
        total_days += no_of_days[d]
    total_days += dd_mm_yyyy[0]
    day = total_days % 7
    if dd_mm_yyyy[0] == start_date[0] and dd_mm_yyyy[1] == start_date[1] and dd_mm_yyyy[2] == start_date[2]:
        return days[0]
    else:
        return days[day]


def get_num_digits_rotated(num):
    strnum = deque(str(num))
    for i in xrange(len(strnum)):
        yield int(''.join(strnum))
        strnum.rotate()


def is_perfect_square(n):
    sqrt = int(n ** 0.5)
    if sqrt ** 2 == n:
        return True
    else:
        return False


def is_perfect_root(n, r):
    powered = int(n ** (1. / r))
    return powered ** r == n


def is_triangular(n):
    sqrt_D = (1 + 8 * n) ** 0.5
    if int(sqrt_D) == sqrt_D and int(sqrt_D) % 2 == 1:
        return True
    else:
        return False


def is_pentagonal(n):
    sqrt_D = (1 + 24 * n) ** 0.5
    if int(sqrt_D) == sqrt_D and (int(sqrt_D) + 1) % 6 == 0:
        return True
    else:
        return False


def triangular(n):
    return (n * (n + 1)) / 2


def pentagonal(n):
    return (n * (3 * n - 1)) / 2


def polygonal(s, n):
    return (s - 2) * n * (n - 1) / 2 + n


def is_twice_square(n):
    t = math.sqrt(n / 2)
    return t == int(t)


def upperlimit(primes):
    limit = 0
    for p in primes:
        if limit + p < 1000000:
            limit += p
        else:
            print p
            return p


def n_C_r(n, r):
    return math.factorial(n)/(math.factorial(r) * math.factorial(n - r))


def phi(n):
    if(n < 1):
        return 0
    if(n == 1):
        return 1
    if is_prime(n):
        return n - 1
    pf = prime_factors(n)
    ans = n
    for f in pf:
        ans *= (f - 1) / f
    return ans
