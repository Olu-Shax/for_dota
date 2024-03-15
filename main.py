import matplotlib.pyplot as plt
import numpy as np


def func(t):
    return -149.3 + 775.3*np.exp(-0.075*t) # точное решение функции из видео


def find_roots(x,y): # функция нахождения нуля (взял отсюда "https://translated.turbopages.org/proxy_u/en-ru.ru.fd689f06-65f4c7d5-6b4799e9-74722d776562/https/stackoverflow.com/questions/46909373/how-to-find-the-exact-intersection-of-a-curve-as-np-array-with-y-0")
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)


tau = 0.5 #шаг по времени или периодичность нанесения урона
T = 25 # условно общая длительность скила
N = int(T / 0.5 + 1) # количество точек по временной оси
h_max = 626 # максимальное здоровье героя
H = h_max * np.ones(N) # массив значения здоровья (начальное)
k = 0.1 # процент
r = 3.8 # реген
f = h_max*np.ones(N) # массив с точным решением
er = np.zeros(N) # массив отклонения численного от аналитического решения
td = 0 # время смерти

t = np.linspace(0, T, N)
for i in range(1, N):
    H[i] = H[i-1]*(1 - 0.75*k*tau) - 15*tau + r*tau # явная схема приближения
    f[i] = func(tau*i)
    er[i] = abs(H[i] - f[i])

s1 = find_roots(t, H)
s2 = find_roots(t, f)
print(er)

plt.grid()
plt.plot(t, H, 'b')
plt.plot(t, f, 'r')
plt.plot(s1, np.zeros(len(s1)), marker="o", ls="", ms=4)
plt.plot(s2, np.zeros(len(s2)), marker="o", ls="", ms=4)
plt.legend(['Численное решение', 'Точное решение'])
plt.show()
