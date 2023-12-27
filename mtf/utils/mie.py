import numpy as np


def bhmie(x, refrel, nang):
    # This file is converted from mie.m, see http://atol.ucsd.edu/scatlib/index.htm
    # Bohren and Huffman originally published the code in their book on light scattering

    # Calculation based on Mie scattering theory
    # input:
    #      x      - size parameter = k*radius = 2pi/lambda * radius
    #                   (lambda is the wavelength in the medium around the scatterers)
    #      refrel - refraction index (n in complex form for example:  1.5+0.02*i;
    #      nang   - number of angles for S1 and S2 function in range from 0 to pi/2
    # output:
    #        S1, S2 - funtion which correspond to the (complex) phase functions
    #        Qext   - extinction efficiency
    #        Qsca   - scattering efficiency
    #        Qback  - backscatter efficiency
    #        gsca   - asymmetry parameter

    nmxx = 150000

    s1_1 = np.zeros(nang, dtype=np.complex128)
    s1_2 = np.zeros(nang, dtype=np.complex128)
    s2_1 = np.zeros(nang, dtype=np.complex128)
    s2_2 = np.zeros(nang, dtype=np.complex128)
    pi = np.zeros(nang, dtype=np.complex128)
    tau = np.zeros(nang, dtype=np.complex128)

    if nang > 1000:
        print("error: nang > mxnang=1000 in bhmie")
        return

    # Require NANG>1 in order to calculate scattering intensities
    if nang < 2:
        nang = 2

    pii = 4.0 * np.arctan(1.0)
    dx = x

    drefrl = refrel
    y = x * drefrl
    ymod = np.abs(y)

    #    Series expansion terminated after NSTOP terms
    #    Logarithmic derivatives calculated from NMX on down

    xstop = x + 4.0 * x**0.3333 + 2.0
    # xstop = x + 4.*x**0.3333 + 10.0
    nmx = max(xstop, ymod) + 15.0
    nmx = np.fix(nmx)

    # BTD experiment 91/1/15: add one more term to series and compare resu<s
    #      NMX=AMAX1(XSTOP,YMOD)+16
    # test: compute 7001 wavelen>hs between .0001 and 1000 micron
    # for a=1.0micron SiC grain.  When NMX increased by 1, only a single
    # computed number changed (out of 4*7001) and it only changed by 1/8387
    # conclusion: we are indeed retaining enough terms in series!

    nstop = np.int(xstop)

    if nmx > nmxx:
        print("error: nmx > nmxx=%f for |m|x=%f" % (nmxx, ymod))
        return

    dang = 0.5 * pii / (nang - 1)

    amu = np.arange(0.0, nang, 1)
    # print amu*90/(nang-1)
    amu = np.cos(amu * dang)
    pi0 = np.zeros(nang, dtype=np.complex128)
    pi1 = np.ones(nang, dtype=np.complex128)

    # Logarithmic derivative D(J) calculated by downward recurrence
    # beginning with initial value (0.,0.) at J=NMX

    nn = np.int(nmx) - 1
    d = np.zeros(nn + 1, dtype=np.complex128)
    for n in range(0, nn):
        en = nmx - n
        d[nn - n - 1] = (en / y) - (1.0 / (d[nn - n] + en / y))

    # *** Riccati-Bessel functions with real argument X
    #    calculated by upward recurrence

    psi0 = np.cos(dx)
    psi1 = np.sin(dx)
    chi0 = -np.sin(dx)
    chi1 = np.cos(dx)
    xi1 = psi1 - chi1 * 1j
    qsca = 0.0
    gsca = 0.0
    p = -1

    for n in range(0, nstop):
        en = n + 1.0
        fn = (2.0 * en + 1.0) / (en * (en + 1.0))

        # for given N, PSI  = psi_n        CHI  = chi_n
        #              PSI1 = psi_{n-1}    CHI1 = chi_{n-1}
        #              PSI0 = psi_{n-2}    CHI0 = chi_{n-2}
        # Calculate psi_n and chi_n
        psi = (2.0 * en - 1.0) * psi1 / dx - psi0
        chi = (2.0 * en - 1.0) * chi1 / dx - chi0
        xi = psi - chi * 1j

        # *** Compute AN and BN:
        an = (d[n] / drefrl + en / dx) * psi - psi1
        an = an / ((d[n] / drefrl + en / dx) * xi - xi1)
        bn = (drefrl * d[n] + en / dx) * psi - psi1
        bn = bn / ((drefrl * d[n] + en / dx) * xi - xi1)

        # *** Store previous values of AN and BN for use
        #    in computation of g=<cos(theta)>
        if n > 0:
            an1 = an
            bn1 = bn

        # *** Augment sums for Qsca and g=<cos(theta)>
        qsca += (2.0 * en + 1.0) * (np.abs(an) ** 2 + np.abs(bn) ** 2)
        gsca += ((2.0 * en + 1.0) / (en * (en + 1.0))) * (
            np.real(an) * np.real(bn) + np.imag(an) * np.imag(bn)
        )

        if n > 0:
            gsca += ((en - 1.0) * (en + 1.0) / en) * (
                np.real(an1) * np.real(an)
                + np.imag(an1) * np.imag(an)
                + np.real(bn1) * np.real(bn)
                + np.imag(bn1) * np.imag(bn)
            )

        # *** Now calculate scattering intensity pattern
        #    First do angles from 0 to 90
        pi = 0 + pi1  # 0+pi1 because we want a hard copy of the values
        tau = en * amu * pi - (en + 1.0) * pi0
        s1_1 += fn * (an * pi + bn * tau)
        s2_1 += fn * (an * tau + bn * pi)

        # *** Now do angles greater than 90 using PI and TAU from
        #    angles less than 90.
        #    P=1 for N=1,3,...% P=-1 for N=2,4,...
        #   remember that we have to reverse the order of the elements
        #   of the second part of s1 and s2 after the calculation
        p = -p
        s1_2 += fn * p * (an * pi - bn * tau)
        s2_2 += fn * p * (bn * pi - an * tau)

        psi0 = psi1
        psi1 = psi
        chi0 = chi1
        chi1 = chi
        xi1 = psi1 - chi1 * 1j

        # *** Compute pi_n for next value of n
        #    For each angle J, compute pi_n+1
        #    from PI = pi_n , PI0 = pi_n-1
        pi1 = ((2.0 * en + 1.0) * amu * pi - (en + 1.0) * pi0) / en
        pi0 = 0 + pi  # 0+pi because we want a hard copy of the values

    # *** Have summed sufficient terms.
    #    Now compute QSCA,QEXT,QBACK,and GSCA

    #   we have to reverse the order of the elements of the second part of s1 and s2
    s1 = np.concatenate((s1_1, s1_2[-2::-1]))
    s2 = np.concatenate((s2_1, s2_2[-2::-1]))
    gsca = 2.0 * gsca / qsca
    qsca = (2.0 / (dx * dx)) * qsca
    qext = (4.0 / (dx * dx)) * np.real(s1[0])

    # more common definition of the backscattering efficiency,
    # so that the backscattering cross section really
    # has dimension of length squared
    qback = 4 * (np.abs(s1[2 * nang - 2]) / dx) ** 2
    # qback = ((abs(s1[2*nang-2])/dx)**2 )/pii  #old form

    return s1, s2, qext, qsca, qback, gsca
