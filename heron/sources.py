"""
Massive black-hole binaries of the LISA Data Challenge "Sangria" (LDC2a) data
set used in the paper: source-frame-independent injection parameters exactly as
analysed in arXiv:2602.22074 (Table I). Angles in radians, masses in solar
masses (detector frame), distance in Mpc; (beta, lam) are ecliptic latitude and
longitude in the solar-system-barycentre frame.

Case 1 is the SNR-357 binary of Figs. 2 and 5 (one month of data, merger 24 h
after the end of the segment).
"""

SANGRIA = {
    1: dict(name='Case 1 (SNR 357)',
            src=dict(m1=4956676.2876, m2=4067166.60352, chi1=-0.523732715, chi2=-0.117412144,
                     dist_Mpc=61097.116076, iota=1.420048341, beta=-1.081082148,
                     lam=4.052962883, psi=1.22844350008, phi0=0.6417162631)),
    2: dict(name='Case 2 (SNR 1500)',
            src=dict(m1=915992.64936, m2=701752.71444, chi1=-0.4502524448, chi2=0.39653644863,
                     dist_Mpc=7705.8526404, iota=2.00630505987, beta=-0.878569862,
                     lam=4.5497195573, psi=3.1386816158, phi0=4.291953078)),
    3: dict(name='Case 3 (low SNR)',
            src=dict(m1=1762682.6427672794, m2=1089989.6619681264,
                     chi1=-0.46132290147769756, chi2=-0.16935723186110851,
                     dist_Mpc=41193.570938256955, iota=1.9532771281418118,
                     beta=0.7936086306404683, lam=0.8531144458899738,
                     psi=1.30104949851723, phi0=5.782796448062781)),
}
