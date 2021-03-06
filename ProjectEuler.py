import math
from fractions import gcd
import io
import re
from itertools import permutations, combinations
import eulerlib
import collections
from operator import itemgetter, mul as mulop
from decimal import getcontext, Decimal


def pe0001():
    print 'Project Euler Problem 1 https://projecteuler.net/problem=1'
    y = 0
    for x in range(1, 1000):
        if x % 3 == 0 or x % 5 == 0:
            y += x
    print 'Answer is ' + str(y)


def pe0002():
    print 'Project Euler Problem 2 https://projecteuler.net/problem=2'
    x = 1
    y = 2
    z = 0
    while y < 4000000:
        if y % 2 == 0:
            z += y
        t = x
        x = y
        y += t
    print 'Answer is ' + str(z)


def pe0003():
    print 'Project Euler Problem 3 https://projecteuler.net/problem=3'
    num = 600851475143
    pf = eulerlib.sieve_of_atkin(int(math.sqrt(num)))
    for p in reversed(pf):
        if num % p == 0:
            print 'Answer is ' + str(p)
            return


def pe0004():
    print 'Project Euler Problem 4 https://projecteuler.net/problem=4'
    largestPalindrome = 0
    a = 999
    while a >= 100:
        b = 999
        while b >= a:
            mul = a * b
            if mul <= largestPalindrome:
                break
            if mul == int(str(mul)[::-1]):
                largestPalindrome = mul
            b = b - 1
        a = a - 1
    print 'Answer is ' + str(largestPalindrome)


def pe0005():
    print 'Project Euler Problem 5 https://projecteuler.net/problem=5'
    divisors = range(11, 21)
    z = range(2, 11)
    for n in range(2, 11):
        for m in divisors:
            if m % n == 0:
                try:
                    z.remove(n)
                    break
                except ValueError:
                    pass
    divisors.extend(z)
    print 'Answer is ' + str(reduce(eulerlib.lcm, divisors))


def pe0006():
    print 'Project Euler Problem 6 https://projecteuler.net/problem=6'
    print 'Answer is ' + str(int(math.pow(eulerlib.sum_of_numbers(100), 2)) - eulerlib.sum_of_squares(100))


def pe0007():
    print 'Project Euler Problem 7 https://projecteuler.net/problem=7'
    target = 10001
    num = 1
    count = 1
    while 1:
        num = num + 2
        if eulerlib.is_prime(num):
            count += 1
        if count == target:
            break
    print 'Answer is ' + str(num)


def pe0008():
    print 'Project Euler Problem 8 https://projecteuler.net/problem=8'
    with open('pe0008.txt') as f:
        big_num = f.read().replace('\n', '')
    prod = 1
    for i in range(0, 987):
        temp = eulerlib.product_digits(int(big_num[i:i+13]))
        if temp > prod:
            prod = temp
    print 'Answer is ' + str(prod)


def pe0009():
    print 'Project Euler Problem 9 https://projecteuler.net/problem=9'
    triplets = eulerlib.euclid_triplet(1000)
    for t in triplets:
        if sum(t) == 1000:
            print 'Answer is ' + str(t[0] * t[1] * t[2])


def pe0010():
    print 'Project Euler Problem 10 https://projecteuler.net/problem=10'
    p = eulerlib.sieve_of_atkin(2000000)
    print 'Answer is ' + str(sum(p))


def pe0011():
    print 'Project Euler Problem 11 https://projecteuler.net/problem=11'
    with open('pe0011.txt') as f:
        grid = [[int(j) for j in l.replace('\n', '').split()] for l in f.readlines()]

    prod = 1
    for i in range(0, 20):
        for j in range(0, 20):
            if j < 17:
                temp = grid[i][j] * grid[i][j+1] * grid[i][j+2] * grid[i][j+3]
                if temp > prod:
                    prod = temp
            if i < 17:
                temp = grid[i][j] * grid[i+1][j] * grid[i+2][j] * grid[i+3][j]
                if temp > prod:
                    prod = temp
            if j > 2:
                temp = grid[i][j] * grid[i][j-1] * grid[i][j-2] * grid[i][j-3]
                if temp > prod:
                    prod = temp
            if i > 2:
                temp = grid[i][j] * grid[i-1][j] * grid[i-2][j] * grid[i-3][j]
                if temp > prod:
                    prod = temp
            if i > 2 and j > 2:
                temp = grid[i][j] * grid[i-1][j-1] * grid[i-2][j-2] * grid[i-3][j-3]
                if temp > prod:
                    prod = temp
            if i > 2 and j < 17:
                temp = grid[i][j] * grid[i-1][j+1] * grid[i-2][j+2] * grid[i-3][j+3]
                if temp > prod:
                    prod = temp
            if i < 17 and j > 2:
                temp = grid[i][j] * grid[i+1][j-1] * grid[i+2][j-2] * grid[i+3][j-3]
                if temp > prod:
                    prod = temp
            if i < 17 and j < 17:
                temp = grid[i][j] * grid[i+1][j+1] * grid[i+2][j+2] * grid[i+3][j+3]
                if temp > prod:
                    prod = temp
    print 'Answer is ' + str(prod)


def pe0012():
    print 'Project Euler Problem 12 https://projecteuler.net/problem=12'
    p = eulerlib.sieve_of_atkin(65500)
    n = 3
    Dn = 2
    cnt = 0
    while cnt <= 500:
        n = n + 1
        n1 = n
        if n1 % 2 == 0:
            n1 = n1 / 2
        Dn1 = 1
        for i in range(0, p[-1]):
            if p[i] * p[i] > n1:
                Dn1 *= 2
                break
            exponent = 1
            while n1 % p[i] == 0:
                exponent += 1
                n1 /= p[i]
            if exponent > 1:
                Dn1 *= exponent
            if n1 == 1:
                break
        cnt = Dn * Dn1
        Dn = Dn1
    print 'Answer is ' + str(n*(n-1)/2)


def pe0013():
    print 'Project Euler Problem 13 https://projecteuler.net/problem=13'
    with open('pe0013.txt') as f:
        num = f.read().replace('\n', '')
    print 'Answer is ' + str(sum(map(int, num.split())))[:10]


def pe0014():
    print 'Project Euler Problem 14 https://projecteuler.net/problem=14'
    c = [1, 2]
    for n in range(3, 1000000):
        count = 0
        t = n
        while t >= n:
            if t % 2 == 0:
                t /= 2
            else:
                t = (3 * t) + 1
            count += 1
        count += c[t - 1]
        c.append(count)
    print 'Answer is ' + str(c.index(max(c)) + 1)


def pe0015():
    print 'Project Euler Problem 15 https://projecteuler.net/problem=15'
    print 'Answer is ' + str(math.factorial(40)/(math.factorial(20) ** 2))


def pe0016():
    print 'Project Euler Problem 16 https://projecteuler.net/problem=16'
    n = 0
    for d in str(2 ** 1000):
        n += int(d)
    print 'Answer is ' + str(n)


def pe0017():
    print 'Project Euler Problem 17 https://projecteuler.net/problem=17'
    word_dict = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight',
                 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen',
                 16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen', 20: 'twenty', 30: 'thirty',
                 40: 'forty', 50: 'fifty', 60: 'sixty', 70: 'seventy', 80: 'eighty', 90: 'ninety',
                 100: 'hundred', 1000: 'thousand'}
    size = 0
    for i in range(1, 1001):
        if i < 21:
            size += len(word_dict[i])
        elif i < 100:
            tens = i - (i % 10)
            ones = i % 10
            if ones != 0:
                size += len(word_dict[tens]) + len(word_dict[ones])
            else:
                size += len(word_dict[tens])
        else:
            huns = i / 100
            ones = 0
            if i % 100 < 20 or (i % 100) / 10 == 0:
                tens = i % 100
            elif (i % 100) / 10 != 0:
                tens = (i % 100) - (i % 10)
                ones = i % 10
            if tens == 0 and ones == 0:
                size += len(word_dict[huns]) + len(word_dict[100])
            elif ones == 0:
                size += len(word_dict[huns]) + len(word_dict[100]) + 3 + len(word_dict[tens])
            else:
                size += len(word_dict[huns]) + len(word_dict[100]) + 3 + len(word_dict[tens]) + len(word_dict[ones])
            if i == 1000:
                size += len(word_dict[1]) + len(word_dict[1000])
    print 'Answer is ' + str(size)


def pe0018():
    print 'Project Euler Problem 18 https://projecteuler.net/problem=18'
    pyramid = []
    gridio = io.open('pe0018.txt', 'r')
    grid = gridio.readlines()
    for p in grid:
        t = p.rstrip().split(' ')
        for q in range(0, len(t)):
            t[q] = int(t[q])
        pyramid.append(t)
    print 'Answer is ' + str(eulerlib.sum_of_pyramid(pyramid))


def pe0019():
    print 'Project Euler Problem 19 https://projecteuler.net/problem=19'
    count = 0
    for y in range(1901, 2001):
        for m in range(1, 13):
            day = eulerlib.get_the_day('1-' + str(m) + '-' + str(y))
            if day == 'Sunday':
                count += 1
    print 'Answer is ' + str(count)


def pe0020():
    print 'Project Euler Problem 20 https://projecteuler.net/problem=20'
    fact_str = str(math.factorial(100))
    print 'Answer is ' + str(sum(int(c) for c in fact_str))


def pe0021():
    print 'Project Euler Problem 21 https://projecteuler.net/problem=21'
    amicable_numbers = []
    for i in range(10, 10001):
        if i not in amicable_numbers:
            t1 = eulerlib.factors(i)
            if t1:
                t1.remove(i)
            num1 = sum(t1)
            t2 = eulerlib.factors(num1)
            if t2:
                t2.remove(num1)
            num2 = sum(t2)
            if num2 == i and num1 != i:
                amicable_numbers.append(i)
                amicable_numbers.append(num1)
    print 'Answer is ' + str(sum(amicable_numbers))


def pe0022():
    print 'Project Euler Problem 22 https://projecteuler.net/problem=22'
    weight = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12,
              'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23,
              'X': 24, 'Y': 25, 'Z': 26}
    namesfh = io.open('pe0022.txt', 'r')
    names = re.sub('"', '', namesfh.read()).split(',')
    names.sort()
    namelist = []
    for name in names:
        charsum = 0
        for c in name:
            charsum += weight[c]
        namelist.append(charsum)
    namesum = 0
    for i in range(0, len(namelist)):
        namesum += (i+1) * namelist[i]
    print 'Answer is ' + str(namesum)


def pe0023():
    print 'Project Euler Problem 23 https://projecteuler.net/problem=23'
    limit = 28123
    abundant_num = []
    non_abundant_sums = range(1, limit+1)
    for n in range(1, limit + 1):
        t = eulerlib.factors(n)
        if t:
            t.remove(n)
        if sum(t) > n:
            abundant_num.append(n)
    for i in range(0, len(abundant_num)):
        for j in range(i, len(abundant_num)):
            abundant_sum = abundant_num[i] + abundant_num[j]
            if abundant_sum in non_abundant_sums:
                non_abundant_sums.remove(abundant_sum)
    print 'Answer is ' + str(sum(non_abundant_sums))


def pe0024():
    print 'Project Euler Problem 24 https://projecteuler.net/problem=24'
    perms = list(permutations(range(10)))
    ans = ''
    for d in perms[999999]:
        ans = ans + str(d)
    print 'Answer is ' + ans


def pe0025():
    print 'Project Euler Problem 25 https://projecteuler.net/problem=25'
    f, n = 0, 0
    while len(str(f)) < 1000:
        f = eulerlib.get_fibonacci_number(n)
        n += 1
    print 'Answer is ' + str(n-1)


def pe0026():
    print 'Project Euler Problem 26 https://projecteuler.net/problem=26'
    primes = eulerlib.sieve_of_atkin(1000)[::-1]
    ans = 0
    for p in primes:
        x = 1
        while (10 ** (x + 2) // p) % 1000 != (10 ** (2 * x + 2) // p) % 1000:
            x += 1
        if x + 1 == p:
            ans = p
            break
    print 'Answer is ' + str(ans)


def pe0027():
    print 'Project Euler Problem 27 https://projecteuler.net/problem=27'
    primes = eulerlib.sieve_of_atkin(1000)[::-1]
    for i in range(0, len(primes)):
        primes.append(-primes[i])
    cons_big = 0
    coeff = [0, 0]
    for b in primes:
        for a in range(-999, 1000):
            n, pc = 0, 0
            while True:
                t = (n * n) + (a * n) + b
                n += 1
                if t < 0:
                    break
                elif eulerlib.is_prime(t):
                    pc += 1
                    if cons_big < pc:
                        cons_big = pc
                        coeff[0] = a
                        coeff[1] = b
                else:
                    break
    print 'Answer is %d' % (coeff[0] * coeff[1])


def pe0028():
    print 'Project Euler Problem 28 https://projecteuler.net/problem=28'
    diag_sum = 1
    t = 1
    for i in range(2, 1001, 2):
        diag_sum += (t + i) + (t + 2 * i) + (t + 3 * i) + (t + 4 * i)
        t = t + 4 * i
    print 'Answer is %d' % diag_sum


def pe0029():
    print 'Project Euler Problem 29 https://projecteuler.net/problem=29'
    distinct_powers = set()
    for a in range(2, 101):
        for b in range(2, 101):
            distinct_powers.add(a ** b)
    print 'Answer is %d' % len(distinct_powers)


def pe0030():
    print 'Project Euler Problem 30 https://projecteuler.net/problem=30'
    fifth_power = []
    for i in range(2, 6 * 9 ** 5):
        if i == eulerlib.sum_power_of_digits(i, 5):
            fifth_power.append(i)
    print 'Answer is %d' % sum(fifth_power)


def pe0031():
    print 'Project Euler Problem 31 https://projecteuler.net/problem=31'
    coins = [1, 2, 5, 10, 20, 50, 100, 200]
    target = 200
    ways = [0 for i in range(target + 1)]
    ways[0] = 1
    for i in coins:
        for j in range(i, target + 1):
            ways[j] = ways[j] + ways[j - i]
    print 'Answer is %d' % ways[target]


def pe0032():
    print 'Project Euler Problem 32 https://projecteuler.net/problem=32'
    productsum = []
    ranges = [[12, 99, 123, 988], [2, 10, 1234, 9877]]
    for r in ranges:
        for i in range(r[0], r[1]):
            for j in range(r[2], r[3]):
                if '0' in str(i) or '0' in str(j):
                    continue
                p = str(i) + str(j) + str(i * j)
                if len(p) > 9:
                    break
                if len(set(p)) == 9 and '0' not in p:
                    productsum.append(i * j)
    print 'Answer is %d' % sum(set(productsum))


def pe0033():
    print 'Project Euler Problem 33 https://projecteuler.net/problem=33'
    ans = [1, 1]
    num_den = []
    for n in range(0, 10):
        for d in range(1, 10):
            for z in range(1, 10):
                n1 = n * 10 + z
                d1 = z * 10 + d
                if float(n1) / d1 - float(n) / d == 0.0 and n1 != d1:
                    num_den.append([n1, d1])
                    ans = [ans[0] * n, ans[1] * d]
    print 'Answer is %d' % (ans[1] / gcd(ans[0], ans[1]))


def pe0034():
    print 'Project Euler Problem 34 https://projecteuler.net/problem=34'
    curious_number = []
    for i in range(3, 100000):
        if i == sum(math.factorial(int(c)) for c in str(i)):
            curious_number.append(i)
    print curious_number
    print 'Answer is %d' % sum(curious_number)


def pe0035():
    print 'Project Euler Problem 35 https://projecteuler.net/problem=35'
    circular_primes = [2, 3, 5, 7, 11, 13, 17, 31, 37, 71, 73, 79, 97]
    primes = eulerlib.sieve_of_atkin(1000000)
    possible_primes = []
    for i in primes:
        if i > 100 and len(set(['0', '2', '4', '5', '6', '8']).intersection(set(str(i)))) == 0:
            possible_primes.append(i)
    for p in possible_primes:
        r = list(eulerlib.get_num_digits_rotated(p))
        for j in range(0, len(r)):
            if r[j] not in primes:
                break
            if j == len(r) - 1:
                for z in r:
                    circular_primes.append(z)
    print 'Answer is %d' % len(set(circular_primes))


def pe0036():
    print 'Project Euler Problem 36 https://projecteuler.net/problem=36'
    dbp = []
    for i in range(1, 1000000):
        if i == int(str(i)[::-1]):
            j = bin(i)[2:]
            if int(j) == int(j[::-1]):
                dbp.append(i)
    print set(dbp)
    print 'Answer is %d' % sum(set(dbp))


def pe0037():
    print 'Project Euler Problem 37 https://projecteuler.net/problem=37'
    primes = eulerlib.sieve_of_atkin(1000000)
    trunc_primes = []
    possible_primes = []
    for i in primes:
        if i < 100 or len(set(['0', '2', '4', '5', '6', '8']).intersection(set(str(i)))) == 0:
            possible_primes.append(i)
    for p in possible_primes:
        for z in range(1, len(str(p))):
            if int(str(p)[:z]) not in primes or int(str(p)[z:]) not in primes:
                break
            if z == len(str(p)) - 1:
                trunc_primes.append(p)
    print trunc_primes
    print 'Answer is %d' % sum(trunc_primes)


def pe0038():
    print 'Project Euler Problem 38 https://projecteuler.net/problem=38'
    biggest_pan = 0
    for i in range(2, 9876):
        concat_str = ''
        for j in range(1, 8):
            concat_str += str(i * j)
            if int(concat_str) > 987654321:
                break
            elif len(set(concat_str)) == 9 and '0' not in concat_str:
                if int(concat_str) > biggest_pan:
                    biggest_pan = int(concat_str)
    print 'Answer is %d' % biggest_pan


def pe0039():
    print 'Project Euler Problem 39 https://projecteuler.net/problem=39'
    p = {}
    py_triples = set()
    ans, max_val = 0, 0
    for m in range(2, 501):
        for n in range(1, int((500 - (m ** 2)) / m) + 1):
            a = (m ** 2) - (n ** 2)
            b = 2 * m * n
            c = (m ** 2) + (n ** 2)
            if a > 0 and c > b and b > a and a + b + c < 1001:
                py_triples.add(tuple([a, b, c]))
                for i in range(2, 84):
                    if i * (a + b + c) < 1001:
                        py_triples.add(tuple([i * a, i * b, i * c]))
                    else:
                        break
    for triplet in py_triples:
        i = sum(triplet)
        if i in p:
            p[i] += 1
        else:
            p[i] = 1
    for k, v in p.iteritems():
        if v > max_val:
            ans = k
            max_val = v
    print 'Answer is %d' % ans


def pe0040():
    print 'Project Euler Problem 40 https://projecteuler.net/problem=40'
    fullstr = ''
    product = 1
    for i in range(1, 1000001):
        fullstr += str(i)
    for j in range(7):
        z = 10 ** j
        product *= int(fullstr[z - 1])
    print 'Answer is %d' % product


def pe0041():
    print 'Project Euler Problem 41 https://projecteuler.net/problem=41'
    for i in range(7, 0, -1):
        pan_perms = tuple(permutations(range(1, i + 1), i))[::-1]
        for pan in pan_perms:
            p = int(''.join(str(c) for c in pan))
            if eulerlib.is_prime(p):
                print 'Answer is %d' % p
                return


def pe0042():
    print 'Project Euler Problem 42 https://projecteuler.net/problem=42'
    alphabets = '"ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    namesfh = io.open('pe0042.txt', 'r')
    names = namesfh.read().split(',')
    triangle_words = []
    for name in names:
        charsum = 0
        for c in name:
            charsum += alphabets.find(c)
        sqrt_D = (1 + (8 * charsum)) ** 0.5
        if int(sqrt_D) == sqrt_D and int(sqrt_D) % 2 == 1:
            triangle_words.append(name)
    print 'Answer is %d' % len(triangle_words)


def pe0043():
    print 'Project Euler Problem 43 https://projecteuler.net/problem=43'
    pandigits = tuple(permutations(range(10)))
    divisibles = []
    divisors = [1, 2, 3, 5, 7, 11, 13, 17]
    for pan in pandigits:
        for i in range(1, 8):
            temp = int(''.join(str(c) for c in pan[i:i + 3]))
            if temp % divisors[i] != 0:
                break
            if i == 7:
                divisibles.append(int(''.join(str(c) for c in pan)))
    print sum(divisibles)


def pe0044():
    print 'Project Euler Problem 44 https://projecteuler.net/problem=44'
    k = 10
    while True:
        for j in range(k - 1, 0, -1):
            Pj = eulerlib.pentagonal(j)
            Pk = eulerlib.pentagonal(k)
            if eulerlib.is_pentagonal(abs(Pj - Pk)) and eulerlib.is_pentagonal(Pj + Pk):
                print 'Answer is %d' % abs(Pj - Pk)
                return
        k += 1


def pe0045():
    print 'Project Euler Problem 45 https://projecteuler.net/problem=45'
    n = 144
    while True:
        H = n * (2 * n - 1)
        if eulerlib.is_pentagonal(H) and eulerlib.is_triangular(H):
            print 'Answer is %d' % H
            return
        n += 1


def pe0046():
    print 'Project Euler Problem 46 https://projecteuler.net/problem=46'
    primes = eulerlib.sieve_of_atkin(10000)
    i = 33
    keep_trying = True
    while keep_trying:
        j = 0
        keep_trying = False
        while i >= primes[j]:
            if eulerlib.is_twice_square(i - primes[j]):
                keep_trying = True
                break
            j += 1
        i += 2
    print 'Answer is %d' % (i - 2)


def pe0047():
    print 'Project Euler Problem 47 https://projecteuler.net/problem=47'
    n = 647
    while n < 500000:
        count = 0
        for i in range(4):
            if len(eulerlib.prime_factors(n + i)) != 4:
                n += i + 1
            else:
                count += 1
        if count == 4:
            print 'Answer is %d' % n
            return


def pe0048():
    print 'Project Euler Problem 48 https://projecteuler.net/problem=48'
    power_sum = 0
    for i in range(1, 1001):
        power_sum += (i ** i) % 10000000000
        power_sum %= 10000000000
    print 'Answer is %d' % power_sum


def pe0049():
    print 'Project Euler Problem 49 https://projecteuler.net/problem=49'
    primes = [p for p in eulerlib.sieve_of_atkin(9999) if p > 1000]
    for i in range(0, len(primes) - 1):
        count = 1
        prime_perm = [primes[i]]
        for j in range(i + 1, len(primes)):
            if sorted(str(primes[i])) == sorted(str(primes[j])):
                count += 1
                prime_perm.append(primes[j])
        if count == 3 and abs(prime_perm[1] - prime_perm[0]) == abs(prime_perm[2] - prime_perm[1]):
            print 'Answer is ' + ''.join(str(c) for c in prime_perm)
            return


def pe0050():
    print 'Project Euler Problem 50 https://projecteuler.net/problem=50'
    primes = eulerlib.sieve_of_atkin(4000)
    for i in range(len(primes) - 1, 1, -1):
        start = 0
        while True:
            prime_sum = sum(primes[start:start + i])
            if prime_sum > 1000000:
                break
            if eulerlib.is_prime(prime_sum):
                    print 'Answer is %d' % prime_sum
                    return
            start += 1


def pe0051():
    print 'Project Euler Problem 51 https://projecteuler.net/problem=51'

    def get_family_count(mk, n, d):
        count = 1
        num = list(str(n))
        for i in range(d + 1, 10):
            for j in range(0, len(mk)):
                if not mk[j]:
                    num[j] = str(i)
            if eulerlib.is_prime(int(''.join(x for x in num))):
                count += 1
        return count

    def generate_num(mk, n, d):
        num = []
        t = str(n)
        for m in mk:
            if m:
                num.append(t[:1])
                t = t[1:]
            else:
                num.append(str(d))
        return int(''.join(x for x in num))

    mask = [[True, True, False, False, False, True],
            [True, False, True, False, False, True],
            [True, False, False, True, False, True],
            [True, False, False, False, True, True],
            [False, True, True, False, False, True],
            [False, True, False, True, False, True],
            [False, True, False, False, True, True],
            [False, False, True, True, False, True],
            [False, False, True, False, True, True],
            [False, False, False, True, True, True]]

    for i in range(101, 1000, 2):
        if i % 5 == 0:
            continue
        for mk in mask:
            for j in range(3):
                if not mk[0] and j == 0:
                    continue
                num = generate_num(mk, i, j)
                if eulerlib.is_prime(num):
                    fc = get_family_count(mk, num, j)
                    if fc == 8:
                        print 'Answer is %d' % num


def pe0052():
    print 'Project Euler Problem 52 https://projecteuler.net/problem=52'
    i = 125875
    while True:
        for j in range(2, 7):
            if set(str(i)) != set(str(j * i)):
                break
            if j == 6:
                print 'Answer is %d' % i
                return
        i += 1


def pe0053():
    print 'Project Euler Problem 53 https://projecteuler.net/problem=53'
    count = 0
    limit = 1000000
    for n in range(23, 101):
        offset = 1
        mid = int(n / 2)
        if eulerlib.n_C_r(n, mid) > limit:
            count += 1
            while True:
                off_sum = eulerlib.n_C_r(n, mid + offset)
                off_diff = eulerlib.n_C_r(n, mid - offset)
                offset += 1
                if off_sum > limit and off_diff > limit:
                    count += 2
                elif off_sum > limit or off_diff > limit:
                    count += 1
                else:
                    break
    print 'Answer is %d' % count


def pe0054():
    print 'Project Euler Problem 54 https://projecteuler.net/problem=54'
    character_map = {'T': 'B', 'J': 'C', 'Q': 'D', 'K': 'E', 'A': 'F'}

    def get_rank(de):
        deck = '23456789BCDEF'
        suit = set()
        values = set()
        v = []
        for i in range(0, 5):
            suit.add(de[i][1])
            values.add(de[i][0])
            v.append(de[i][0])
        t = sorted(v)
        counter = collections.Counter(t)
        mc = counter.most_common()
        cv = counter.values()
        if len(suit) == 1:
            if values == ['B', 'C', 'D', 'E', 'F']:
                return 10, t[::-1]
            elif deck.index(t[0]) + 4 == deck.index(t[4]):
                return 9, t[::-1]
            elif len(values) == 5:
                return 6, t[::-1]
        else:
            if len(values) == 2:
                if 4 in cv:
                    z = [0, 0]
                    for k, v in counter:
                        if v == 4:
                            z[0] = k
                        elif v == 1:
                            z[1] = k
                    return 8, z
                else:
                    return 7, [mc[0][0], mc[1][0]]
            elif len(values) == 5:
                if deck.index(t[0]) + 4 == deck.index(t[4]):
                    return 5, t[::-1]
                else:
                    return 1, t[::-1]
            elif len(values) == 3:
                if 3 in cv:
                    return 4, [mc[0][0], mc[2][0], mc[1][0]]
                if 2 in collections.Counter(cv).values():
                    return 3, [mc[1][0], mc[0][0], mc[2][0]]
            elif len(values) == 4:
                return 2, [mc[0][0], mc[3][0], mc[2][0], mc[1][0]]

    dealsio = io.open('pe0054.txt', 'r')
    deals = dealsio.readlines()
    ans, l = 0, 0
    for deal in deals:
        l += 1
        d = [str(i) for i in list(deal.rstrip('\n').split(' '))]
        for i in range(0, len(d)):
            if d[i][0] in ['T', 'J', 'K', 'Q', 'A']:
                d[i] = character_map[d[i][0]] + d[i][1]
        p1, p2 = sorted(d[:5], key=lambda x: (x[1], x[0])), sorted(d[5:], key=lambda x: (x[1], x[0]))
        r1, pow1 = get_rank(p1)
        r2, pow2 = get_rank(p2)
        # print 'Deal number %d' % l
        if r1 > r2:
            # print 'Player1 wins!'
            ans += 1
        elif r2 > r1:
            pass
            # print 'Player2 wins!'
        else:
            for i in range(0, min(len(pow1), len(pow2))):
                if pow1[i] > pow2[i]:
                    # print 'Player1  wins!'
                    ans += 1
                    break
                elif pow1[i] < pow2[i]:
                    # print 'Player2 wins!'
                    break
    print 'Player1 won %d times' % ans


def pe0055():
    print 'Project Euler Problem 55 https://projecteuler.net/problem=55'
    ln = 0
    checked = []
    for i in range(195, 10000):
        t = []
        if i not in checked:
            z = i
            t.append(z)
            for j in range(50):
                t.append(int(str(z)[::-1]))
                z += int(str(z)[::-1])
                if z == int(str(z)[::-1]):
                    checked += t
                    break
                if j == 49:
                    ln += 1
    print 'Answer is %d' % ln


def pe0056():
    print 'Project Euler Problem 56 https://projecteuler.net/problem=56'
    ans = 0
    for i in range(100):
        for j in range(100):
            t = eulerlib.sum_digits(i ** j)
            if ans < t:
                ans = t
    print 'Answer is %d' % ans


def pe0057():
    print 'Project Euler Problem 57 https://projecteuler.net/problem=57'
    n, d = 3, 2
    ans = 0
    for i in range(2, 1001):
        t = n
        n += 2 * d
        d += t
        if len(str(n)) > len(str(d)):
            ans += 1
    print 'Answer is %d' % ans


def pe0058():
    print 'Project Euler Problem 58 https://projecteuler.net/problem=58'
    primes = []
    diags = [1]
    t = 1
    i = 2
    while True:
        for j in range(1, 5):
            z = t + (i * j)
            diags.append(z)
            if eulerlib.is_prime(z):
                primes.append(z)
        if float(len(primes)) / len(diags) < 0.1:
            ans = 1 + (len(diags) / 2)
            break
        t = t + 4 * i
        i += 2
    print 'Answer is %d' % ans


def pe0059():
    print 'Project Euler Problem 59 https://projecteuler.net/problem=59'
    key = []
    with open('pe0059.txt', 'rt') as handle:
        # read the cipher
        cipher = handle.read().split(",")
        cipher = [int(x) for x in cipher]

        for n in range(3):
            # initialize counts for cipher values at positions == n (mod 3)
            counts = [0]*256
            for x in range(len(cipher[n::3])):
                counts[cipher[n::3][x]] += 1

            # recover the most frequent cipher character for the n-th key value
            # this is expected to be the space character (ASCII value == 32)
            k, _ = max(enumerate(counts), key=itemgetter(1))

            # xor with 32 to recover n-th key value
            key.append(k ^ 32)

    print 'Answer is %d' % sum([cipher[n] ^ key[n % 3] for n in range(len(cipher))])


def pe0060():
    print 'Project Euler Problem 60 https://projecteuler.net/problem=60'
    # Trial and error to find the uppoerbound as 10000 for sensible execution time
    primes = eulerlib.sieve_of_atkin(10000)
    # removed 2 & 5 from primes since numbers ending with 2 & 5 are not primes
    primes.remove(2)
    primes.remove(5)

    def isPrimePair(x, y):
        if eulerlib.is_prime(int(str(x) + str(y))) and eulerlib.is_prime(int(str(y) + str(x))):
            return True
        else:
            return False

    def find5thpair(primes):
        for i in range(len(primes)):
            for j in range(i + 1, len(primes)):
                pairs = []
                if isPrimePair(primes[i], primes[j]):
                    pairs.append(primes[i])
                    pairs.append(primes[j])
                    for a in range(j + 1, len(primes)):
                        if isPrimePair(pairs[0], primes[a]) and isPrimePair(pairs[1], primes[a]):
                            pairs.append(primes[a])
                            for b in range(a + 1, len(primes)):
                                if isPrimePair(pairs[0], primes[b]) and isPrimePair(pairs[1], primes[b]) and \
                                   isPrimePair(pairs[2], primes[b]):
                                    pairs.append(primes[b])
                                    for c in range(b + 1, len(primes)):
                                        if isPrimePair(pairs[0], primes[c]) and \
                                           isPrimePair(pairs[1], primes[c]) and isPrimePair(pairs[2], primes[c]) and \
                                           isPrimePair(pairs[3], primes[c]):
                                            pairs.append(primes[c])
                                            return pairs

    result = find5thpair(primes)
    print 'Answer is %d' % sum(result)


def pe0061():
    print 'Project Euler Problem 61 https://projecteuler.net/problem=61'
    c_list = []
    global pairs
    pairs = [0] * 6
    indx = 0

    def find_pair(last, l):
        for x in range(6):
            if pairs[x] != 0:
                continue
            for y in range(len(c_list[x])):
                if c_list[x][y] not in pairs and str(c_list[x][y])[:2] == str(pairs[last])[2:]:
                    pairs[x] = c_list[x][y]
                    if l == 5 and str(pairs[5])[:2] == str(pairs[x])[2:]:
                        return True
                    elif find_pair(x, l + 1):
                        return True
            pairs[x] = 0
        return False

    for s in range(3, 9):
        c_list.append([])
        for n in range(150):
            pn = eulerlib.polygonal(s, n)
            if pn > 1000 and pn < 10000 and int(str(pn)[2:]) > 9:
                c_list[indx].append(pn)
            if pn > 9999:
                indx += 1
                break

    for i in range(len(c_list[5])):
        pairs[5] = c_list[5][i]
        if find_pair(5, 1):
            break
    print 'Answer is %d' % sum(pairs)


def pe0062():
    print 'Project Euler Problem 62 https://projecteuler.net/problem=62'
    num = 345
    cubes = {}
    while True:
        k = ''.join(sorted(str(num ** 3)))
        if k not in cubes.keys():
            cubes[k] = []
        cubes[k].append(num)
        if len(cubes[k]) == 5:
            result = sorted(cubes[k])[0] ** 3
            break
        num += 1

    print 'Answer is %d' % result


def pe0063():
    print 'Project Euler Problem 63 https://projecteuler.net/problem=63'
    #
    # x^n is a n-digit perfect nth power
    # If 10^{n-1} <= x^n < 10^n and 1 <= x < 10
    # n-1 <= nlog10(x) therefore: n <= 1 / (1 - log10(x))
    # We are looking for sum of x 1 --> 9 of 1 / 1 - log10(x)
    #
    print 'Answer is %d' % sum([int(1/(1-math.log10(x))) for x in range(1, 10)])


def pe0064():
    print 'Project Euler Problem 64 https://projecteuler.net/problem=64'

    def period(n):
        m, d, a0 = 0, 1, int(n ** 0.5)
        a = a0
        cf = [a0]
        while a != 2 * a0:
            m = d * a - m
            d = (n - m * m) // d
            a = (a0 + m) // d
            cf.append(a)
        return cf

    total = 0
    for n in range(1, 10001):
        if not eulerlib.is_perfect_square(n) and (len(period(n)) - 1) % 2 != 0:
            total += 1

    print 'Answer is %d' % total


def pe0065():
    print 'Project Euler Problem 65 https://projecteuler.net/problem=65'
    n, n1, n2 = 0, 1, 1
    for i in range(1, 101):
        a = (1, (i / 3) * 2)[i % 3 == 0]
        n = (n1 * a) + n2
        n2 = n1
        n1 = n
    print 'Answer is %d' % sum(map(int, str(n)))


def pe0066():
    print 'Project Euler Problem 66 https://projecteuler.net/problem=66'
    max_X = 0
    max_D = 0

    def chakravala(N):
        a, b, m, k = 1, 0, 1, 1
        while k != 1 or b == 0:
            m = k * (m/k+1) - m
            m = m - int((m - math.sqrt(N))/k) * k

            A = (a*m + N*b) / abs(k)
            b = (a + b*m) / abs(k)
            k = (m*m - N) / k

            a = A

        return a

    for N in range(2, 1001):
        if not eulerlib.is_perfect_square(N):
            z = chakravala(N)
            if z > max_X:
                max_X = z
                max_D = N

    print 'Answer is %d' % max_D


def pe0067():
    print 'Project Euler Problem 67 https://projecteuler.net/problem=67'
    pyramid = []
    gridio = io.open('pe0067.txt', 'r')
    grid = gridio.readlines()
    for p in grid:
        t = p.rstrip().split(' ')
        for q in range(0, len(t)):
            t[q] = int(t[q])
        pyramid.append(t)
    print 'Answer is ' + str(eulerlib.sum_of_pyramid(pyramid))


def pe0068():
    print 'Project Euler Problem 68 https://projecteuler.net/problem=68'
    for outer_list in ((10,) + o for o in combinations(range(9, 0, -1), 4)):
        inner_list = tuple(i for i in range(9, 0, -1) if i not in outer_list)
        for outer in (outer_list[-1:] + o for o in permutations(outer_list[:-1])):
            for inner in permutations(inner_list):
                total = outer[4] + inner[4] + inner[0]
                if all(outer[i] + inner[i] + inner[i + 1] == total for i in range(4)):
                    ans = (a for i in range(5) for a in (outer[i], inner[i], inner[(i + 1) % 5]))
                    print 'Answer is ' + ''.join(map(str, ans))
                    return


def pe0069():
    print 'Project Euler Problem 69 https://projecteuler.net/problem=69'
    # To have higher n / phi(n), phi(n) should be smaller
    # phi(n) = n * sum(1 - 1/p), where p is all prime factors of n
    # hence if we have more prime factors for a number then that will have less phi(n)
    limit = 1000001
    primes = eulerlib.sieve_of_atkin(50)
    prod = 1
    pnum = []
    for p in primes:
        prod *= p
        if prod < limit:
            pnum.append(p)
        else:
            break
    print 'Answer is ' + str(reduce(mulop, pnum, 1))


def pe0070():
    print 'Project Euler Problem 70 https://projecteuler.net/problem=70'
    # To have lower n / phi(n), phi(n) should be bigger.
    # Only when n is prime it's bigger with phi(n) = n - 1, but n - 1 will not be permutation of n
    # phi(n) = n * sum(1 - 1/p), where p is all prime factors of n
    # Hence if we have less prime factors for a number then that will have higher phi(n), hence go for 2 prime factors.
    # phi(p1 x p2) = phi(p1) x phi(p2) = (p1 - 1) x (p2 - 1)
    # It's rational to start from sqrt(10^7)
    limit = 10 ** 7
    primes = eulerlib.sieve_of_atkin(2 * int(math.sqrt(limit)))
    ans = 1
    min_num = float(10 ** 7)
    for p1, p2 in combinations(primes, 2):
        if p1 * p2 < limit:
            n = p1 * p2
            phi = (p1 - 1) * (p2 - 1)
            tmp = float(n) / phi
            if sorted(str(n)) == sorted(str(phi)) and tmp < min_num:
                min_num = tmp
                ans = n
    print 'Answer is ' + str(ans)


def pe0071():
    print 'Project Euler Problem 71 https://projecteuler.net/problem=71'
    n, d = 2, 5
    while d <= (10 ** 6) - 7:
        n += 3
        d += 7
    print 'Answer is ' + str(n)


def pe0072():
    print 'Project Euler Problem 72 https://projecteuler.net/problem=72'
    limit = 10 ** 6
    phi = range(limit + 1)
    phi[1] = 0
    for n in range(2, limit + 1):
        if phi[n] == n:
            for i in range(n, limit + 1, n):
                phi[i] -= phi[i] // n
    print 'Answer is ' + str(sum(phi))


def pe0073():
    print 'Project Euler Problem 73 https://projecteuler.net/problem=73'
    limit = 12001
    den = [0] * limit
    count = 0
    for i in range(2, limit):
        fractions = i - (i / 2) - (i / 3) - 1
        count += fractions
        num = fractions - den[i]
        for j in range(2 * i, limit, i):
            count -= num
            den[j] += num
    print 'Answer is ' + str(count)


def pe0074():
    print 'Project Euler Problem 74 https://projecteuler.net/problem=74'
    limit = 1000000
    count = 0
    for i in range(limit, 9, -1):
        flag = True
        tmp = i
        chain = [i]
        while flag:
            next_in = sum(math.factorial(int(c)) for c in str(tmp))
            if next_in in chain:
                break
            else:
                chain.append(next_in)
                tmp = next_in
        if len(chain) == 60:
            count += 1
    print 'Answer is ' + str(count)


def pe0075():
    print 'Project Euler Problem 75 https://projecteuler.net/problem=75'
    # Refer: https://en.wikipedia.org/wiki/Pythagorean_triple#Generating_a_triple
    limit = 1500000
    py_sum = []
    m = 2
    for m in range(2, int(math.sqrt(limit/2)) + 1):
        mm = m ** 2
        for n in range(1, m):
            nn = n ** 2
            a = mm - nn
            b = 2 * m * n
            c = mm + nn
            pysum = a + b + c
            if gcd(a, b) == 1:
                k = 1
                multiple = k * pysum
                while multiple < limit + 1:
                    py_sum.append(multiple)
                    k += 1
                    multiple = k * pysum
    print 'Answer is %d' % collections.Counter(py_sum).values().count(1)


def pe0076():
    print 'Project Euler Problem 76 https://projecteuler.net/problem=76'
    num = 101
    combi = [1] * num
    for i in range(2, num):
        for j in range(i, num):
            combi[j] = combi[j] + combi[j - i]

    print 'Answer is %d' % (combi[num - 1] - 1)


def pe0077():
    print 'Project Euler Problem 77 https://projecteuler.net/problem=77'
    primes = eulerlib.sieve_of_atkin(100)
    print primes
    ways = [0] * 100
    ways[0] = 1
    for p in primes:
        for z in range(p, 100):
            ways[z] = ways[z] + ways[z - p]
    for w in ways:
        if w >= 5000:
            print 'Answer is %d' % ways.index(w)
            break


def pe0078():
    print 'Project Euler Problem 78 https://projecteuler.net/problem=78'
    # Refer: https://en.wikipedia.org/wiki/Partition_(number_theory)#Partition_function_formulas
    part = [1, 1, 2]
    while True:
        val = 0
        n = len(part)
        for dk in [1, -1]:
            k = dk
            while True:
                i = n - (k*(3*k-1))//2
                if i < 0:
                    break
                val += (1 if k % 2 else -1) * part[i]
                k += dk
        if val % 10 ** 6 == 0:
            print 'Answer is %d' % len(part)
            break
        part.append(val)


def pe0079():
    print 'Project Euler Problem 79 https://projecteuler.net/problem=79'
    with open('pe0079.txt') as f:
        num = f.readlines()

    num = list(set([l.rstrip() for l in num]))
    digits = list(set(''.join(map(str, num))))

    def reorder(code, l):
        i = code.index(l[0])
        j = code.index(l[1])
        k = code.index(l[2])
        i, j, k = sorted((i, j, k))
        code[i], code[j], code[k] = l
        return code

    for p in num:
        reorder(digits, p)
    print 'Answer is %s' % ''.join(digits)


def pe0080():
    print 'Project Euler Problem 80 https://projecteuler.net/problem=80'
    getcontext().prec = 102
    total = 0
    for i in range(2, 100):
        if not eulerlib.is_perfect_square(i):
            root = Decimal(i).sqrt()
            rootnum = [int(j) for j in (str(root)[:101]).replace('.', '')]
            total += sum(rootnum)
    print 'Answer is %d' % total


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Program that solves problems in Project Euler')
    parser.add_argument('-p', '--problem', help='Enter the Project Euler program number', required=True)
    args = vars(parser.parse_args())
    allmthd = globals().copy()
    allmthd.update(locals())
    method = allmthd.get('pe' + str(args['problem']).zfill(4))
    if not method:
        raise Exception("Method pe%s is not implemented" % str(args['problem']).zfill(4))
    method()
