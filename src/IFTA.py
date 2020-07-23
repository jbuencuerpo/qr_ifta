#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

def GSA_A(I, steps=100,  **kwargs):
    """ IFTA method using binary gratings, original Gerberch Saxton"""
    N = (I.shape[0] - 1) // 2
    #In = I[:]
    In = np.fft.ifftshift(I)
    I0 = In[:]
    I0 = I0 / I0.max()
    I0 = np.array(I0, dtype='complex128')
    if "seed" in kwargs:
        np.random.seed(kwargs['seed'])
    Iimag = kwargs.get('Iimag', np.random.random((In.shape[0], In.shape[1])))
    I0.imag = Iimag
    error = np.zeros(steps)
    for s in range(steps):
        F2 = np.fft.ifft2(I0)
        F2 = np.arctan2(F2.imag, F2.real)
        F2 = np.abs(F2 / F2.max())
        F2 = np.rint(F2)
        F2 = F2 * np.pi 

        I2 =  np.fft.fft2(np.exp(F2*1j))

        error[s] = np.sqrt((In - np.abs(I2) / np.abs(I2).max()) ** 2).sum() / \
                   (In.shape[0] * In.shape[1])  * 100
        I2_phase = np.arctan2(I2.imag, I2.real)
        I0 = In * np.exp(I2_phase*1j)

    I2 = np.fft.fft2(np.exp(F2 * 1j))
    return F2, np.fft.fftshift(I2), error


def WA_A(I, I_GSA=None, M=2, steps=100, **kwargs):
    """ IFTA method using a ZeroPadding frame with high frequencies using as seed the conventiona I_GSA"""

    if I_GSA is None:
        I_GSA = GSA_A(I,steps=steps, **kwargs)[1]

    if I.shape[0] % 2 == 0:
        N = I.shape[0] // 2
        IE = np.zeros(((2 * N) + 2 * N * M, (2 * N) + 2 * N * M))
    else:
        N = (I.shape[0] - 1) // 2
        IE = np.zeros(((2 * N + 1) + 2 * N * M, (2 * N + 1) + 2 * N * M))

    win2 = [M * N, -M * N]
    IE[win2[0]:win2[1], win2[0]:win2[1]] = I[:]

    IE = np.fft.ifftshift(IE)
    In = np.fft.ifftshift(I)

    I0E = np.zeros_like(IE)
    I0E = np.array(I0E, dtype='complex128')

    if "seed" in kwargs:
        if kwargs['seed'] > 0:
            np.random.seed(kwargs['seed'])
        

    if 'random_window' in kwargs:
        print('using random window')
        Iimag = np.random.random((IE.shape[0], IE.shape[1]))
        I0E.imag = Iimag

    I0E[win2[0]:win2[1], win2[0]:win2[1]] = I_GSA
    I0E = np.fft.ifftshift(I0E)

    WLUT = np.zeros_like(I0E.real)
    WLUT[win2[0]:win2[1], win2[0]:win2[1]] = 1
    WLUT = np.fft.ifftshift(WLUT)

    WLUT = WLUT > 0.1  # To create look up table
    error = np.zeros(steps)
    for s in range(steps):
        F2 = np.fft.ifft2(I0E)
        if not np.isfinite(F2).all():
            print(F2[np.isnan(F2)])
        F2 = np.arctan2(F2.imag, F2.real)

        F2 = np.abs(F2 / F2.max())
        F2 = np.rint(F2 + 1e-19)
        F2 = F2 * np.pi 

        I2 = np.fft.fft2(np.exp(F2 * 1j + 1e-19))
        error[s] = np.sqrt(((IE[WLUT] - np.abs(I2[WLUT]) / np.abs(I2[WLUT]).max()) ** 2)).sum() / \
                   (In.shape[0] * In.shape[1])   * 100
        if not np.isfinite(I2).all():
            print('Bad seed conditioning, low dimensional image')
            raise ValueError

        I2_phase = np.arctan2(I2.imag, I2.real)
        # I2 = I2.imag
        I0E_phase = np.exp(I2_phase*1j)
        I0E[WLUT] = IE[WLUT][:]  # * I2[N:3*N+1, N:3*N+1]
        I0E = I0E * I0E_phase 


    I2 = np.fft.fft2(np.exp(F2 * 1j))
    return F2, np.fft.fftshift(I2), error


def main():
    import matplotlib
    matplotlib.use("QT5Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import skimage.io

    N = 23
    I = np.ones((2 * N + 1, 2 * N + 1))
    x, y = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1))

    I[x ** 2 + y ** 2 <= 36] = 0.
    I[x ** 2 + y ** 2 > 17 ** 2] = 0.

    M =  3 # Zero Padding M*N size
    extra = {'seed':23} # Just my Birthday
    
    F_GSA, I_GSA, error_GSA = GSA_A(I, steps=100,  **extra)
    F_WA, I_WA, error_WA = WA_A(I, M=M, steps=100, **extra)

    I_GSA = abs(I_GSA) / abs(I_GSA).max()
    I_WA = abs(I_WA) / abs(I_WA).max()

    N = (I.shape[0] - 1) // 2
    win2 = [M * N, -M * N]
    
    ## Plotting the results
    fg = plt.figure()
    ax1 = fg.add_subplot(221)
    plt.title('Objective')
    im1 = ax1.imshow(I, interpolation='nearest',)#norm=LogNorm())
    fg.colorbar(im1, ax=ax1)

    ax2 = fg.add_subplot(222)
    plt.title("Result with Window")
    im2 = ax2.imshow(I_WA, interpolation='nearest')#,norm=LogNorm())
    fg.colorbar(im2, ax=ax2)

    ax3 = fg.add_subplot(223)
    plt.title("Result without Window")
    im3 = ax3.imshow(abs(I_GSA), interpolation='nearest')#,norm=LogNorm())
    fg.colorbar(im3, ax=ax3)

    ax4 = fg.add_subplot(224)
    plt.title("Region of interest result with Window")
    im4 = ax4.imshow(abs(I_WA[win2[0]:win2[1], win2[0]:win2[1]]), interpolation='nearest')#,norm=LogNorm())
    
    fg.colorbar(im4, ax=ax4)
    fg.tight_layout()

    fg2 = plt.figure()
    ax21 = fg2.add_subplot(121)
    plt.title("DOE without Window")
    im21 = ax21.imshow(F_GSA.real / np.pi, interpolation='nearest')
    fg2.colorbar(im21)

    ax22 = fg2.add_subplot(122)
    plt.title("DOE with Window")
    im22 = ax22.imshow(F_WA.real / np.pi, interpolation='nearest')
    fg2.colorbar(im22)
    
    fg3 = plt.figure()
    ax3 = fg3.add_subplot(111)
    plt.title("Error")
    ax3.plot(error_WA, label='$\delta_\mathrm{ZeroPad}$')
    ax3.plot(error_GSA, label='$\delta_\mathrm{Direct}$')
    ax3.legend(loc=0)
    plt.show()

   
if __name__ == '__main__':
    main()
