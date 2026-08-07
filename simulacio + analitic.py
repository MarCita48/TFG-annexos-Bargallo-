# =====================
# PAQUETS NECESSARIS
# =====================

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# ===============================================
# FUNCIONS DE GENERACIÓ DE VARIABLES ALEATÒRIES
# ===============================================

def generar_exponencial(n, lambd):
    U = np.random.uniform(0, 1, n)
    X = -np.log(U) / lambd
    if n ==1:
        return X[0]
    return X

def generar_poisson (n,lambd ,T) : 
    X    = []
    for _ in range(n):
        i = 0
        t = 0.0
        while True:
            u = np.random.uniform(0, 1)
            t += -(1 / lambd) * np.log(u)
            if t >= T:
                break
            i += 1
        X.append(i)
    return X 


# ========================
# FUNCIONS DE SIMULACIÓ
# ========================

def generar_tau():
    return generar_exponencial(1, 1/mitjana_tau)

def generar_x_prima(c, mu):
    return sum(generar_exponencial(1, mu) for _ in range(c))

def generar_c():
    return min(generar_poisson(1, mitjana_c, 1)[0], C_max)

def generar_S(x):
    temps = 0
    clients = 0
    while temps < x:
        servei = generar_exponencial(1, mu)
        temps += servei
        if temps < x:
            clients += 1
    return clients

def generar_arribada(temps, lambd):
    t, n, W = 0, 0, 0
    while t < temps:
        entrearribades = generar_exponencial(1, lambd)
        if t + entrearribades > temps:
            W += n * (temps - t)
            break
        W += n * entrearribades
        t += entrearribades
        n += 1
    return n, W


# ========================
# ESTIMACIÓ DE GAMMA I X
# ========================

def estimar_gamma_x( mitjana_c, mu, mostres, inici = False, metro = False):
    gammes = []
    xs = []
    for _ in range(mostres):
        if metro == True:
            tau = mitjana_tau
        else:
            tau = generar_tau()                        # τi
        if inici == True:
            c = mitjana_c
        else:
            c = generar_c()                            # ci+1
        x = min(generar_x_prima(c, mu), tau)
        xs.append(x)
        temps = 0
        clients = 0
        while temps < x:
            servei = generar_exponencial(1, mu)
            temps += servei
            if temps < x:
                clients += 1
        gammes.append(min(clients, c))
    return np.mean(gammes), np.mean(xs)


# ===================
# LAMBDA CORRECTE
# ===================

def calcular_lambda_objetiu(rho_obj):
    return rho_obj * E_gamma_global / mitjana_tau


# ========================
# CICLE DE SIMULACIÓ
# ========================

def simular_cicle(X_i, lambda_clients, tau_now, c_now, x_now, inici, metro):
    
    # PAS 1 - Generar τi, ci+1 i xi+1
    tau_next = mitjana_tau if metro else generar_tau()        # τi
    c_next = mitjana_c if inici else generar_c()              # ci+1
    x_next = min(tau_next, generar_x_prima(c_next, mu))       # xi+1 = min(τi,x')

    # PAS 2 - Generar A¹i durant τi − xi
    A1, Wq = generar_arribada(tau_now - x_now, lambda_clients)

    # PAS 3 - Generar A²i durant xi+1
    A2, D = generar_arribada(x_next, lambda_clients)

    # PAS 4
    A = A1 + A2

    # PAS 5
    W_prima = Wq + X_i * (tau_now - x_now + x_next) + D
    W = W_prima + A1 * x_next

    # PAS 6 - Generar Si+1
    S_next = generar_S(x_next)

    # PAS 7 - Xi+1 = clients no atesos
    X_seguent = max(X_i + A - min(S_next, c_next), 0)

    # PAS 8 - zi+1 = clients efectivament atesos
    z = min(X_i + A, c_next, S_next)

    # PAS 9 - Yi = màxim nombre de clients presents just abans del servei
    Y = X_i + A1

    return X_seguent, A, S_next, c_next, z, Y, W, W_prima, tau_next, x_next


# ===================================
# SIMULACIÓ DE N CICLES AMB BURN-IN
# ====================================

def simular_sistema(N, lambda_clients, burn_in, inici, metro):

    X = 0
    X_ultima = 0
    x_ultima = 0

    tau_now = mitjana_tau if metro else generar_tau()
    c_now = mitjana_c if inici else generar_c()
    x_now = min(tau_now, generar_x_prima(c_now, mu))

    for _ in range(burn_in):
        X, A, S, c_next, z, Y, W, Wp, tau_next, x_next = simular_cicle(X, lambda_clients, tau_now, c_now, x_now, inici, metro)
        tau_now, c_now, x_now = tau_next, c_next, x_next
             
    suma_A = suma_W = suma_Wp = suma_Z = 0
    suma_S = suma_c = suma_gamma = 0
    suma_tau = suma_x = 0
    suma_Y = 0
    suma_X = 0
    X_max = 0; X_min = float('inf')
    Y_max = 0; Y_min = float('inf')

    sigma_tau = sigma_c = sigma_z = 0
    sigma_x = sigma_S = sigma_gamma = sigma_X = sigma_Y = 0
    
    for _ in range(N):
        
        X, A, S, c_next, z, Y, W, Wp, tau_next, x_next = simular_cicle(X, lambda_clients, tau_now, c_now, x_now, inici, metro)
        tau_now, c_now, x_now = tau_next, c_next, x_next

        gamma = min(S, c_now)
        
        suma_A += A
        suma_W += W
        suma_Wp += Wp
        suma_Z += z
        suma_S += S
        suma_c += c_now
        suma_gamma += gamma
        suma_Y    += Y
        suma_X    += X          # X és el X_seguent retornat
        X_ultima   = X
        X_max      = max(X_max, X)
        X_min      = min(X_min, X)
        Y_max      = max(Y_max, Y)
        Y_min      = min(Y_min, Y)

        suma_tau += tau_now
        suma_x += x_now
        
        sigma_tau += tau_now**2
        sigma_c += c_now**2
        sigma_z += z**2
        sigma_x   += x_now**2
        sigma_S   += S**2
        sigma_gamma += gamma**2
        sigma_X   += X**2
        sigma_Y   += Y**2

        x_ultima = x_now

        
    return {
        "A_media": suma_A / N,
        "W_total": suma_W,
        "W_prime_total": suma_Wp,
        "Z_media": suma_Z / N,
        "S_media": suma_S / N,
        "c_media": suma_c / N,
        "gamma_media": suma_gamma / N,
        "tau_media": suma_tau / N,
        "x_media": suma_x / N,
        "Y_media": suma_Y / N,
        "X_media": suma_X / N,
        "X_max": X_max,
        "X_min": X_min,
        "Y_max": Y_max,
        "Y_min": Y_min,
        "T_N_xN": suma_tau + x_ultima,
        "sigma_x": sigma_x,
        "sigma_S": sigma_S,
        "sigma_gamma": sigma_gamma,
        "sigma_X": sigma_X,
        "sigma_Y": sigma_Y,
        "suma_tau": suma_tau,
        "sigma_tau": sigma_tau,
        "sigma_c": sigma_c,
        "sigma_z": sigma_z,
        "passatgers_totals": suma_A,
    }
    

# =========================
# MÈTRIQUES ESTADÍSTIQUES
# =========================

def calcular_metriques(res, N):
    
    def std(sigma, mitjana):
        return np.sqrt((N/(N-1)) * (sigma/N - mitjana**2))
    
    tau_bar = res["tau_media"];   s_tau  = std(res["sigma_tau"],  tau_bar)
    x_bar   = res["x_media"];     s_x    = std(res["sigma_x"],    x_bar)
    c_bar   = res["c_media"];     s_c    = std(res["sigma_c"],    c_bar)
    S_bar   = res["S_media"];     s_S    = std(res["sigma_S"],    S_bar)
    g_bar   = res["gamma_media"]; s_g    = std(res["sigma_gamma"],g_bar)
    z_bar   = res["Z_media"];     s_z    = std(res["sigma_z"],    z_bar)
    X_bar   = res["X_media"];     s_X    = std(res["sigma_X"],    X_bar)
    Y_bar   = res["Y_media"];     s_Y    = std(res["sigma_Y"],    Y_bar)

    rho = res["A_media"] / min(g_bar, S_bar)

    T_N = res["suma_tau"]
    W0  = (T_N / (2*N)) * (1 + (N**2 * s_tau**2) / (T_N**2))

    L  = res["W_total"]       / res["T_N_xN"]
    Lp = res["W_prime_total"] / res["T_N_xN"]
    W  = res["W_total"]       / (N * res["A_media"])
    Wp = res["W_prime_total"] / (N * res["A_media"])

    return {
        "0  rho_estimada":  rho,                                                #factor de carga
        "1  tau_bar":       tau_bar,                                            #temps entre arribades
        "2  Ctau":          s_tau / tau_bar,                                    #coeficient d'variació de tau
        "3  W0":            W0,                                                 #temps mitjà d'espera en cua per a N arribades
        "4  x_bar":         x_bar,                                              #temps de servei mitja
        "5  Cx":            s_x / x_bar,                                        #coeficient d'variació de x
        "6  c_bar":         c_bar,                                              #capacitat mitja
        "7  Cc":            s_c / c_bar,                                        #coeficient d'variació de c
        "8  S_bar":         S_bar,                                              # nombre màxim de clients que poden ser atesos durant el temps de servei
        "9  CS":            s_S / S_bar,                                        #coeficient d'variació de S
        "10 mu_bar":        S_bar / x_bar,                                      #taxa de servei mitjana
        "11 A_bar":         res["A_media"],                                     #nombre mitjà d'arribades per cicle
        "12 L":             L,                                                  #nombre mitjà de clients al sistema
        "13 L'":            Lp,                                                 #nombre mitjà de clients a la cua
        "14 X_bar":         X_bar,                                              #nombre mitjà de clients al sistema al final de cada cicle
        "15 X_max":         res["X_max"],                                       #nombre màxim de clients al sistema al final de cada cicle
        "16 X_min":         res["X_min"],                                       #nombre mínim de clients al sistema al final de cada cicle
        "17 CX":            s_X / X_bar if X_bar > 0 else float('nan'),         #coeficient d'variació de X
        "18 Y_bar":         Y_bar,                                              # nombre mitjà de clients presents just abans de l'inici del servei
        "19 Y_max":         res["Y_max"],                                       #nombre màxim de clients al sistema al moment de les arribades
        "20 Y_min":         res["Y_min"],                                       #nombre mínim de clients al sistema al moment de les arribades
        "21 CY":            s_Y / Y_bar if Y_bar > 0 else float('nan'),         #coeficient d'variació de Y
        "22 z_bar":         z_bar,                                              #nombre mitjà de clients que es poden servir per cicle
        "23 Cz":            s_z / z_bar,                                        #coeficient d'variació de z
        "24 W/W0":          W / W0,                                             #factor d'espera respecte a W0
        "25 W'/W0":         Wp / W0,                                            #factor d'espera en cua respecte a W0                                                                                                                                                                                                                                                                                                                                                          
        "26 W":             W,
        "27 W'":            Wp
    }


# ==========================================================
# MÈTODES NUMÈRICS PER A LA RESOLUCIÓ DE L'EQUACIÓ CARACTERÍSTICA
# ==========================================================

def secante(f, x0, x1, tol=1e-6, max_iter=100):
    iteraciones = [x0, x1]

    for i in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)

        if abs(fx1 - fx0) < 1e-12:
            raise ValueError("Denominador massa petit a la secant")

        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        iteraciones.append(x2)

        if abs(x2 - x1) < tol:
            print(f"Convergencia ha trigat {i+1} iteracions")
            return x2

        x0, x1 = x1, x2

    raise ValueError(f"Màxim d'iteracions ({max_iter}) assolit sense convergència")


def biseccion(f, a, b, tol=1e-6, max_iter=100):
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("L'interval no conté cap canvi de signe")

    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c)

        if abs(fc) < tol or abs(b - a) < tol:
            print(f"Convergencia ha trigat {i+1} iteracions")
            return c

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    raise ValueError("No ha convergit")


def encontrar_intervalos(f, a, b, n=1000):
    xs = np.linspace(a, b, n)
    intervalos = []
    for i in range(len(xs) - 1):
        if f(xs[i]) * f(xs[i + 1]) < 0:
            intervalos.append((xs[i], xs[i + 1]))
    return intervalos


def eliminar_duplicados(raices, tol=1e-6):
    raices_unicas = []
    for r in raices:
        if not any(abs(r - ru) < tol for ru in raices_unicas):
            raices_unicas.append(r)
    return raices_unicas


# ======================================================================
# FUNCIÓ ÚNICA: SIMULACIÓ + RESOLUCIÓ ANALÍTICA M/Mᵏ/1 PER A CADA rho
# ======================================================================

def experiment_convergencia_mu(mu, rho_valors, N, mostres_gamma=2000, inici=True, metro=False):
    """
    Realitza una simulació i resolució analítica per a diferents valors de rho en un sistema M/Mᵏ/1.
    """
    resultats = []
    global E_gamma_global, E_x_global

    for rho_actual, N_actual in zip(rho_valors, N):

        burn_in =  N_actual//2
        
        # ---------- 1) SIMULACIÓ ----------
        E_gamma_global, E_x_global = estimar_gamma_x(mitjana_c, mu, mostres_gamma, inici=True, metro=False)
        lambd = calcular_lambda_objetiu(rho_actual)
        res = simular_sistema(N_actual, lambd, burn_in, inici=True, metro=False)
        met = calcular_metriques(res, N_actual)
        print(f"rho = {rho_actual}, rho_estimada = {met['0  rho_estimada']}, L = {met['12 L']}, L' = {met["13 L'"]}, W = {met['26 W']}, W' = {met["27 W'"]}, W/W0 = {met['24 W/W0']}, W'/W0 = {met["25 W'/W0"],}, E[c] = {met['6  c_bar']}, E[γ] = {met['8  S_bar']}")

        L_sim = met["12 L"]
        W_sim = met["26 W"]


        # ---------- 2) RESOLUCIÓ ANALÍTICA M/Mᵏ/1 ----------
        tau_bar = mitjana_tau
        c_bar = mitjana_c
        #c_bar = int(round(res["gamma_media"]))   # capacitat efectiva, no física

        mu_teoric = 1 / tau_bar
        K_teoric = int(round(c_bar))
        print(f"  [INFO] Paràmetres teòrics: μ={mu_teoric:.10f}, K={K_teoric}, λ={lambd:.10f}")

        def f(r, lam=lambd, mu_t=mu_teoric, K=K_teoric):
            # Equació característica: μ·r^(K+1) - (λ+μ)·r + λ = 0
            return mu_t * r ** (K + 1) - (lam + mu_t) * r + lam

        intervals = encontrar_intervalos(f, 0, 1, n=10000)

        r0 = None
        for a, b in intervals:
            try:
                r_cand = secante(f, a, b)
            except ValueError:
                continue
            if 0 < r_cand < 1:
                r0 = r_cand
                residual = f(r0)
                print(f"  [INFO] Arrel r0 trobada: {r0:.10f} per rho = {rho_actual} amb residu f(r0)={residual:.3e}")
                
                from scipy.optimize import brentq
                r0_check = brentq(f, a, b, xtol=1e-14, rtol=1e-14)
                print(f"  [INFO] r0 (secant)={r0:.10f}  r0 (brentq)={r0_check:.10f}")
                break
        
        if r0 is None:
            print(f"  [AVÍS] No s'ha trobat l'arrel r0 ∈ (0,1) per rho = {rho_actual}. S'omet aquest cas.")
            continue

        L_an = r0 / (1 - r0)
        Lq_an = L_an - (lambd / mu_teoric)
        W_an = L_an / lambd
        Wq_an = W_an - (1 / mu_teoric)

        # ---------- 3) ERROR RELATIU (%) ----------
        def error_rel(sim, an):
            return abs(sim - an) / abs(an) * 100 if an != 0 else float('nan')

        resultats.append({
            "rho_obj": rho_actual,
            "rho_sim": met["0  rho_estimada"],
            "lambda": lambd,
            "tau_bar": tau_bar,
            "c_bar": c_bar,
            "mu_teoric": mu_teoric,
            "K_teoric": K_teoric,
            "r0": r0,

            "L_sim": L_sim, "W_sim": W_sim,
            "L_an":  L_an, "W_an":  W_an,

            "err_L":  error_rel(L_sim,  L_an),
            "err_W":  error_rel(W_sim,  W_an),
        })

    return resultats


# ======================================================================
# EXPORTACIÓ DE LA TAULA COMPARATIVA A IMATGE
# ======================================================================

def exportar_taula_convergencia(resultats, filename="taula_convergencia.png"):
    """
    Genera una taula amb una fila per cada rho simulat, i tres columnes:
    Simulació | Resolució analítica (M/Mᵏ/1) | Error relatiu (%).
    Cada cel·la mostra les 4 mètriques L, Lq, W i Wq.
    """
    # --- 1. CONFIGURAR FUENTE CALIBRI GLOBALMENTE ---
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Calibri']

    n_files = len(resultats)
    if n_files == 0:
        print("[AVÍS] No hi ha resultats per exportar.")
        return

    headers = ["rho", "Simulació", "Resolució analítica (M/Mᵏ/1)", "Error relatiu (%)"]

    cell_text = []
    for r in resultats:
        col_sim = (
            f"L  = {r['L_sim']:.4f}\n"
            f"W  = {r['W_sim']:.4f}\n"
        )
        col_an = (
            f"L  = {r['L_an']:.4f}\n"
            f"W  = {r['W_an']:.4f}\n"
        )
        col_err = (
            f"L:  {r['err_L']:.2f} %\n"
            f"W:  {r['err_W']:.2f} %\n"
        )
        cell_text.append([f"{r['rho_obj']}", col_sim, col_an, col_err])

    fig, ax = plt.subplots(figsize=(11, 1.0 * n_files + 1.2), dpi=300)
    ax.axis('off')

    fig.text(
        0.5, 1,
        f"Convergència Simulació vs. M/Mᵏ/1 ( E[τ]={mitjana_tau}, E[c]={mitjana_c}, C={C_max}, μ={mu} )",
        ha='center', va='top', fontsize=11, weight='bold', fontname='Calibri'
    )

    tabla = ax.table(cellText=cell_text, colLabels=headers, loc='center', cellLoc='center')

    tabla.auto_set_font_size(False)
    tabla.set_fontsize(12)
    tabla.scale(1.0, 3)

    for (row, col), cell in tabla.get_celld().items():
        cell.set_text_props(ha='center', va='center', fontname='Calibri')
        if row == 0:
            cell.set_text_props(weight='bold', fontname='Calibri')
            cell.set_height(cell.get_height() * 0.6)
        if col == 0:
            cell.set_width(0.10)
        else:
            cell.set_width(0.30)

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Taula de convergència exportada correctament a: '{filename}'")


# =========================================================
# PARÀMETRES DEL SISTEMA EXPERIMENT:
# CONVERGENCIA ENTRE SIMULACIÓ I RESOLUCIÓ ANALÍTICA M/Mᵏ/1
# =========================================================

mitjana_tau = 12       # E[τ]: temps mitjà entre arribades consecutives de servidors
mitjana_c = 50         # E[c]: capacitat física mitjana dels servidors
C_max = 50             # C: capacitat física màxima admissible
mu = 275               # taxa de servei individual dels clients (es reassigna a cada μ de l'experiment)

np.random.seed(42)    # llavor per garantir la reproductibilitat dels resultats

# Execució de l'experiment de convergència per a diferents valors de rho
if __name__ == "__main__":

    rho_valors = [0.2, 0.4, 0.6, 0.8, 0.95, 0.98]  # Valors de rho a simular
    N = [3000, 3000, 3000, 3000, 3000, 3000]  # Nombre de cicles a simular per a cada rho

    resultats_convergencia = experiment_convergencia_mu(mu, rho_valors, N, inici = True, metro = False)

    for r in resultats_convergencia:
        print(
            f"rho_obj={r['rho_obj']:>2}  |  rho_sim={r['rho_sim']:>2}  |  L_sim={r['L_sim']:.2f}  L_an={r['L_an']:.2f} Error={r['err_L']:.2f}% |  W_sim={r['W_sim']:.2f}  W_an={r['W_an']:.2f} Error={r['err_W']:.2f}%  \n"
        )

    exportar_taula_convergencia(resultats_convergencia, filename="simulacio-analitic.png")




