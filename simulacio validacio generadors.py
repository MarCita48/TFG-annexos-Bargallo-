import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ==========================
# PARÁMETROS
# ==========================

N = 5000

lambda_exp = 4      # Exponencial
lambda_pois = 4     # Poisson

np.random.seed(42)

# ==========================
# EXPONENCIAL + Método de la inversa
# ==========================

def generar_exponencial(n, lambd):
    U = np.random.uniform(0, 1, n)
    X = -np.log(U) / lambd
    return X

# ==========================
# POISSON
# ==========================

def generar_poisson (n,lambd ,T) : 
    """
    Igual que generar_poisson_correcta, però retorna també
    totes les U~Uniform(0,1) usades internament.
    """
    X    = []
    U_all = []       # totes les uniformes generades
    for _ in range(n):
        i = 0
        t = 0.0
        while True:
            u = np.random.uniform(0, 1)
            U_all.append(u)              # guardem la llavor
            t += -(1 / lambd) * np.log(u)
            if t >= T:
                break
            i += 1
        X.append(i)
    return X, np.array(U_all)

# ==========================
# GENERACIÓN
# ==========================

exponenciales = generar_exponencial(N, lambda_exp)
poisson, U_seeds = generar_poisson(N, lambda_pois, 1)


# ==============================================================
# 1. ESTADÍSTICS BÀSICS
# ==============================================================
 
def estadistics_basics(mostra, nom, dist_teorica):
    """Imprimeix estadístics mostrals vs. teòrics."""
    mostra = np.array(mostra)
    n         = len(mostra)
    mitjana   = np.mean(mostra)
    varianca  = np.var(mostra, ddof=1)
    std       = np.std(mostra, ddof=1)
    cv        = std / mitjana
    asimetria = stats.skew(mostra)
    curtosi   = stats.kurtosis(mostra)   # excés de curtosi (normal = 0)
 
    print(f"\n{'='*57}")
    print(f"  ESTADÍSTICS BÀSICS — {nom}")
    print(f"{'='*57}")
    print(f"  {'Mètrica':<25} {'Mostral':>13} {'Teòric':>13}")
    print(f"  {'-'*52}")
    for etiqueta, mostral, teoric in [
        ("Mitjana (μ)",      mitjana,   dist_teorica["mitjana"]),
        ("Variància (σ²)",   varianca,  dist_teorica["varianca"]),
        ("Desv. típica (σ)", std,       dist_teorica["std"]),
        ("Coef. variació",   cv,        dist_teorica["cv"]),
        ("Asimetria",        asimetria, dist_teorica["asimetria"]),
        ("Curtosi (excés)",  curtosi,   dist_teorica["curtosi"]),
    ]:
        print(f"  {etiqueta:<25} {mostral:>13.4f} {teoric:>13.4f}")
    print(f"  {'N (mostra)':<25} {n:>13}")
 
 
# Valors teòrics Exponencial(λ)
teorica_exp = {
    "mitjana":   1 / lambda_exp,
    "varianca":  1 / lambda_exp**2,
    "std":       1 / lambda_exp,
    "cv":        1.0,          # sempre 1 per a l'exponencial
    "asimetria": 2.0,          # sempre 2
    "curtosi":   6.0,          # excés de curtosi = 6
}
 
# Valors teòrics Poisson(λ·T), aquí T=1
teorica_pois = {
    "mitjana":   lambda_pois,
    "varianca":  lambda_pois,
    "std":       np.sqrt(lambda_pois),
    "cv":        1 / np.sqrt(lambda_pois),
    "asimetria": 1 / np.sqrt(lambda_pois),
    "curtosi":   1 / lambda_pois,
}
 
estadistics_basics(exponenciales,    f"Exponencial(λ={lambda_exp})", teorica_exp)
estadistics_basics(poisson, f"Poisson(λ={lambda_pois})",    teorica_pois)
 
 
# ==============================================================
# 2. TESTS DE BONDAT D'AJUST
# ==============================================================
 
print(f"\n{'='*57}")
print(f"  TESTS DE BONDAT D'AJUST")
print(f"{'='*57}")
 
# --- 2a. Test Kolmogorov-Smirnov per a l'Exponencial ---
# H₀: la mostra segueix Exp(λ), és a dir F(x) = 1 - e^{-λx}
ks_stat, ks_pval = stats.kstest(
    exponenciales, 'expon', args=(0, 1 / lambda_exp)
)
print(f"\n  Exponencial — Test Kolmogorov-Smirnov")
print(f"    Estadístic KS  : {ks_stat:.4f}")
print(f"    p-valor        : {ks_pval:.4f}")
conclusio = "NO rebutgem H₀ ✓" if ks_pval > 0.05 else "REBUTGEM H₀ ✗"
print(f"    Conclusió      : {conclusio}")
 
# --- 2b. Test Chi² per a la Poisson ---
# H₀: la mostra segueix Poisson(λ)
# Construïm les classes agrupant la cua (esperats < 5 → agrupem)
poisson_arr = np.array(poisson)
k_max       = int(np.percentile(poisson_arr, 99))
 
observats = np.bincount(poisson_arr, minlength=k_max + 2)
observats[-1] = np.sum(observats[k_max + 1:])   # agrupem cua dreta
observats = observats[:k_max + 2]
 
probs     = stats.poisson.pmf(np.arange(k_max + 1), mu=lambda_pois)
prob_cua  = 1 - stats.poisson.cdf(k_max, mu=lambda_pois)
probs     = np.append(probs, prob_cua)
esperats  = N * probs
 
# Agrupem classes amb esperats < 5 (requisit del test chi²)
obs_agr, esp_agr = [], []
oa, ea = 0, 0
for o, e in zip(observats, esperats):
    oa += o;  ea += e
    if ea >= 5:
        obs_agr.append(oa);  esp_agr.append(ea)
        oa, ea = 0, 0
if ea > 0 and obs_agr:       # absorb resta si n'hi ha
    obs_agr[-1] += oa;  esp_agr[-1] += ea
 
obs_agr   = np.array(obs_agr)
esp_agr   = np.array(esp_agr)
gl        = len(obs_agr) - 1 - 1   # classes - 1 paràmetre estimat
 
chi2_stat = np.sum((obs_agr - esp_agr)**2 / esp_agr)
chi2_pval = 1 - stats.chi2.cdf(chi2_stat, df=gl)
 
print(f"\n  Poisson — Test Chi²")
print(f"    Classes agrupades  : {len(obs_agr)}")
print(f"    Graus de llibertat : {gl}")
print(f"    Estadístic χ²      : {chi2_stat:.4f}")
print(f"    p-valor            : {chi2_pval:.4f}")
conclusio = "NO rebutgem H₀ ✓" if chi2_pval > 0.05 else "REBUTGEM H₀ ✗"
print(f"    Conclusió          : {conclusio}")
 
 
# ==============================================================
# 3. HISTOGRAMES + CORBES TEÒRIQUES + Q-Q PLOTS
# ==============================================================
# Fila superior: histograma + densitat/massa teòrica.
# Fila inferior: Q-Q plot (quantils mostrals vs. quantils teòrics).
# Fons blanc en tota la figura.

fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.4)

COLOR_BARRA  = "#1f77b4"
COLOR_TEORIA = "#d62728"
COLOR_TEXT   = "#1a1a1a"
COLOR_FONS   = "white"
COLOR_GRID   = "#dddddd"
COLOR_QQLINE = "#2ca02c"

# ---- Histograma esquerra: Exponencial ----
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(COLOR_FONS)

ax1.hist(exponenciales, bins=50, density=True,
         color=COLOR_BARRA, alpha=0.75, edgecolor="white", linewidth=0.4,
         label="Freq. relativa")

x_vals = np.linspace(0, np.percentile(exponenciales, 99.5), 300)
ax1.plot(x_vals, stats.expon.pdf(x_vals, scale=1 / lambda_exp),
         color=COLOR_TEORIA, linewidth=2.5, label=f"Exp(λ={lambda_exp}) teòrica")

ax1.axvline(np.mean(exponenciales), color="#ff9800", linewidth=1.5,
            linestyle="--", label=f"Mitjana mostral = {np.mean(exponenciales):.3f}")

ax1.set_title(f"Histograma — Exponencial (λ={lambda_exp})",
              color=COLOR_TEXT, fontsize=11, pad=10)
ax1.set_xlabel("x", color=COLOR_TEXT)
ax1.set_ylabel("Densitat", color=COLOR_TEXT)
ax1.tick_params(colors=COLOR_TEXT)
ax1.grid(True, color=COLOR_GRID, linewidth=0.6, alpha=0.7)
for spine in ax1.spines.values():
    spine.set_edgecolor("#aaaaaa")
ax1.legend(facecolor=COLOR_FONS, labelcolor=COLOR_TEXT, fontsize=8.5,
           edgecolor="#aaaaaa")

# ---- Histograma dreta: Poisson ----
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(COLOR_FONS)

k_unique, comptes = np.unique(poisson_arr, return_counts=True)
ax2.bar(k_unique, comptes / N, color=COLOR_BARRA, alpha=0.75,
        edgecolor="white", linewidth=0.4, label="Freq. relativa")

k_plot = np.arange(0, k_unique.max() + 1)
ax2.plot(k_plot, stats.poisson.pmf(k_plot, mu=lambda_pois),
         'o-', color=COLOR_TEORIA, linewidth=2, markersize=5,
         label=f"Poisson(λ={lambda_pois}) teòrica")

ax2.set_title(f"Histograma — Poisson (λ={lambda_pois})",
              color=COLOR_TEXT, fontsize=11, pad=10)
ax2.set_xlabel("k", color=COLOR_TEXT)
ax2.set_ylabel("Probabilitat", color=COLOR_TEXT)
ax2.tick_params(colors=COLOR_TEXT)
ax2.grid(True, color=COLOR_GRID, linewidth=0.6, alpha=0.7)
for spine in ax2.spines.values():
    spine.set_edgecolor("#aaaaaa")
ax2.legend(facecolor=COLOR_FONS, labelcolor=COLOR_TEXT, fontsize=9,
           edgecolor="#aaaaaa")

# ---- Q-Q plot esquerra: Exponencial ----
# stats.probplot calcula els quantils teòrics (Exp(scale=1/lambda_exp))
# i els compara amb els quantils mostrals ordenats.
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(COLOR_FONS)

(osm, osr), (slope, intercept, r) = stats.probplot(
    exponenciales, dist=stats.expon, sparams=(0, 1 / lambda_exp), fit=True
)
ax3.scatter(osm, osr, color=COLOR_BARRA, s=12, alpha=0.6, label="Quantils mostrals")
ax3.plot(osm, slope * osm + intercept, color=COLOR_QQLINE, linewidth=2,
         label=f"Recta teòrica (R²={r**2:.4f})")

ax3.set_title("Q-Q plot — Exponencial", color=COLOR_TEXT, fontsize=11, pad=10)
ax3.set_xlabel(f"Quantils teòrics Exp(λ={lambda_exp})", color=COLOR_TEXT)
ax3.set_ylabel("Quantils mostrals", color=COLOR_TEXT)
ax3.tick_params(colors=COLOR_TEXT)
ax3.grid(True, color=COLOR_GRID, linewidth=0.6, alpha=0.7)
for spine in ax3.spines.values():
    spine.set_edgecolor("#aaaaaa")
ax3.legend(facecolor=COLOR_FONS, labelcolor=COLOR_TEXT, fontsize=8.5,
           edgecolor="#aaaaaa")

# ---- Q-Q plot dreta: Poisson ----
# La Poisson és discreta, per la qual cosa stats.probplot no s'aplica
# directament. Construïm el Q-Q plot manualment: per a cada probabilitat
# acumulada empírica p_i de la mostra ordenada, el quantil teòric és la
# inversa de la CDF de la Poisson (ppf) amb el paràmetre lambda_pois.
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(COLOR_FONS)

poisson_sorted   = np.sort(poisson_arr)
n_pois           = len(poisson_sorted)
probs_emp        = (np.arange(1, n_pois + 1) - 0.5) / n_pois
quantils_teorics = stats.poisson.ppf(probs_emp, mu=lambda_pois)

ax4.scatter(quantils_teorics, poisson_sorted, color=COLOR_BARRA, s=10,
            alpha=0.4, label="Quantils mostrals")

lim_min = min(quantils_teorics.min(), poisson_sorted.min())
lim_max = max(quantils_teorics.max(), poisson_sorted.max())
ax4.plot([lim_min, lim_max], [lim_min, lim_max], color=COLOR_QQLINE,
         linewidth=2, label="Recta y = x")

ax4.set_title("Q-Q plot — Poisson", color=COLOR_TEXT, fontsize=11, pad=10)
ax4.set_xlabel(f"Quantils teòrics Poisson(λ={lambda_pois})", color=COLOR_TEXT)
ax4.set_ylabel("Quantils mostrals", color=COLOR_TEXT)
ax4.tick_params(colors=COLOR_TEXT)
ax4.grid(True, color=COLOR_GRID, linewidth=0.6, alpha=0.7)
for spine in ax4.spines.values():
    spine.set_edgecolor("#aaaaaa")
ax4.legend(facecolor=COLOR_FONS, labelcolor=COLOR_TEXT, fontsize=8.5,
           edgecolor="#aaaaaa")

fig.suptitle("Validació dels Generadors de Variables Aleatòries",
             color=COLOR_TEXT, fontsize=14, fontweight="bold", y=0.995)

plt.savefig("validacio generadors.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("\n  Figura guardada: validacio generadors.png")



# ==============================================================
# RUNS TEST (Wald-Wolfowitz) — Test d'independència
# ==============================================================
# Valida que la seqüència generada és aleatòria (sense patrons).
# H₀: els números s'han generat de forma independent i aleatòria.
# ==============================================================

def runs_test(mostra, nom, alpha=0.05):
    """
    Test de runs (Wald-Wolfowitz) per a independència.
    Aproximació normal vàlida per n > 20.
    """
    v = np.array(mostra)
    n = len(v)

    # Pas 1: vector de signes (+1 puja, -1 baixa o igual)
    signes = np.where(v[:-1] < v[1:], 1, -1)

    # Pas 2: comptar canvis de direcció → nombre de runs
    canvis = np.sum(signes[:-1] != signes[1:])
    r = canvis + 1          # nombre total de runs

    # Pas 3: mitjana i variància teòriques sota H₀
    mu_r    = (2 * n - 1) / 3
    sigma_r = (16 * n - 29) / 90

    # Pas 4: estadístic Z
    Z = (r - mu_r) / np.sqrt(sigma_r)

    # Pas 5: valors crítics
    z_critic = stats.norm.ppf(1 - alpha / 2)
    rebutgem = abs(Z) >= z_critic
    conclusio = "REBUTGEM H₀ ✗  (patró detectat)" if rebutgem else "NO rebutgem H₀ ✓  (seqüència independent)"

    print(f"\n{'='*57}")
    print(f"  RUNS TEST — {nom}")
    print(f"{'='*57}")
    print(f"  N (mostra)             : {n}")
    print(f"  Nombre de runs (R)     : {r}")
    print(f"  Mitjana teòrica (μR)   : {mu_r:.4f}")
    print(f"  Variància teòrica (σ²) : {sigma_r:.4f}")
    print(f"  Estadístic Z           : {Z:.4f}")
    print(f"  Valor crític ±z_α/2    : ±{z_critic:.4f}  (α={alpha})")
    print(f"  Conclusió              : {conclusio}")

    return Z, r, rebutgem


Z_exp,  r_exp,  _ = runs_test(exponenciales,    f"Exponencial(λ={lambda_exp})")
Z_pois, r_pois, _ = runs_test(poisson, f"Poisson(λ={lambda_pois})")
Z_unif, r_unif, _ = runs_test(U_seeds, f"Uniform(0,1) seeds internes del generador de Poisson(λ={lambda_pois})")

