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
    negatius = 0
    if temps <= 0:
        negatius = negatius + 1
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

def estimar_gamma_x( mitjana_c, mu, mostres=20000, inici = False, metro = False):
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
        "26 W":             W,                                                  #temps mitjà d'espera en cua per a cada client
        "27 W'":            Wp,                                                 #temps mitjà d'espera en cua per a cada client que arriba al sistema    
    }


# ====================================
# EXPORTAR TAULA RESULTATS A IMATGE
# ====================================
def exportar_taula_a_imatge(taula_dades, N, mitjana_tau, mitjana_c, C_max, mu, filename="resultats_taula.png"):
    
    # --- CONFIGURAR CALIBRI ---
    import matplotlib.pyplot as plt # (Asegúrate de que está importado)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Calibri', 'DejaVu Sans'] # DejaVU actúa como respaldo
    # -------------------------------------------

    rhos_keys = [0.2, 0.4, 0.6, 0.8, 0.95, 0.98]
    
    # Definició exacta de les files i els seus símbols matemàtics corresponents
    estructura_filas = [
        ("0", r"$\tilde{\rho}$"),
        ("1", r"$\bar{\tau}$"),
        ("2", r"$C_{\tau}$"),
        ("3", r"$W_0$"),
        ("4", r"$\bar{x}$"),
        ("5", r"$C_x$"),
        ("6", r"$\bar{c} (= \gamma_c)$"),
        ("7", r"$C_c$"),
        ("8", r"$\bar{S}$"),
        ("9", r"$C_S$"),
        ("10", r"$\bar{\mu}$"),
        ("11", r"$\bar{A}$"),
        ("12", r"$L$"),
        ("13", r"$L'$"),
        ("14", r"$\bar{X}$"),
        ("15", r"$\hat{X}$"),  # X_max el seu símbol matemàtic
        ("16", r"$X_{min}$"),
        ("17", r"$C_X$"),
        ("18", r"$\bar{Y}$"),
        ("19", r"$\hat{Y}$"),  # Y_max el seu símbol matemàtic
        ("20", r"$Y_{min}$"),
        ("21", r"$C_Y$"),
        ("22", r"$\bar{z}$"),
        ("23", r"$C_z$"),
        ("24", r"$W/W_0$"),
        ("25", r"$W'/W_0$")
    ]
    
    # Mapeig de les claus de la taula a les etiquetes corresponents
    map_claus = {
        "0": "0  rho_estimada", "1": "1  tau_bar", "2": "2  Ctau", "3": "3  W0",
        "4": "4  x_bar", "5": "5  Cx", "6": "6  c_bar", "7": "7  Cc",
        "8": "8  S_bar", "9": "9  CS", "10": "10 mu_bar", "11": "11 A_bar",
        "12": "12 L", "13": "13 L'", "14": "14 X_bar", "15": "15 X_max",
        "16": "16 X_min", "17": "17 CX", "18": "18 Y_bar", "19": "19 Y_max",
        "20": "20 Y_min", "21": "21 CY", "22": "22 z_bar", "23": "23 Cz",
        "24": "24 W/W0", "25": "25 W'/W0"
    }

    # Construir la matriu de dades per a la taula
    headers = ["", ""] + [f"$\\rho = {r}$" for r in rhos_keys]
    cell_text = []
    
    for idx_str, label in estructura_filas:
        fila = [idx_str, label]
        clau_metrica = map_claus[idx_str]
        
        for r in rhos_keys:
            valor = taula_dades[r][clau_metrica]
            if hasattr(valor, "__len__") and not isinstance(valor, (str, dict)):
                valor = valor[0]
            
            # Formatejar els valors numèrics amb dos decimals
            fila.append(f"{valor:.2f}" if isinstance(valor, float) else str(valor))
            
        cell_text.append(fila)

    # Configurar dimensions de la figura de Matplotlib
    fig, ax = plt.subplots(figsize=(10, 11), dpi=300)
    ax.axis('off')


    # ====================================
    # ADJUNTAR INFORMACIÓ DE LA SIMULACIÓ
    # ====================================

    text_esquerra = (
        f"Paràmetres d'entrada:\n\n"
        f"E[τ] = {mitjana_tau}\n"
        f"E[c] = {mitjana_c}\n"
        f"C = {C_max}\n"
        f"μ = {mu}"
    )

    passatgers_totals = taula_dades[0.98]["passatgers_totals"]

    text_dreta = (
        f"Grandàries mostrals:\n\n"
        f"Passatgers totals = {passatgers_totals:.0f}\n"
        f"Busos simulats = {N}"
    )

    fig.text(
        0.24,          # marge esquerre
        0.97,
        text_esquerra,
        ha='left',
        va='top',
        fontsize=10
    )

    fig.text(
        0.615,          # marge dret
        0.97,
        text_dreta,
        ha='left',
        va='top',
        fontsize=10
    )


    # Crear la taula
    tabla = ax.table(cellText=cell_text, colLabels=headers, loc='center', cellLoc='center')
    
    # Estilitzar la taula
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(12)
    tabla.scale(1.2, 1.4)  # Amplada y alçada de les cel·les
    
    # Fer que la primera fila y columnas resaltin
    for (row, col), cell in tabla.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
        # Amplada de la columna d'etiquetes
        if col == 0: cell.set_width(0.05)
        elif col == 1: cell.set_width(0.18)
        else: cell.set_width(0.13)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Taula exportada correctament com imatge a: '{filename}'")


# ========================
# INTERVALS DE CONFIANÇA
# ========================

def experiment_intervals(N, rho_obj, repeticions=15, inici = False, metro = False):
    
    burn_in_local = N // 2
    rho_vals = []; L_vals = []; Y_vals = []; A_vals = []; WW0_vals = []
    
    for i in range(repeticions):
        np.random.seed(i)
        
        lambd = calcular_lambda_objetiu(rho_obj)
        res = simular_sistema(N, lambd, burn_in_local, inici, metro)
        met = calcular_metriques(res, N)

        # Plot de les X amb el pas del temps
        plt.plot(res["XS"], label=f"Simulació {i+1}")

        rho_vals.append(met["0  rho_estimada"])
        L_vals.append(met["12 L"])
        Y_vals.append(met["18 Y_bar"])
        A_vals.append(met["11 A_bar"])
        WW0_vals.append(met["24 W/W0"])
    
    def ic(vals, nom):
        m   = np.mean(vals)
        s   = np.std(vals, ddof=1)
        n   = len(vals)
        t   = stats.t.ppf(0.975, df=n-1)   # t-Student amb n-1 graus de llibertat
        mg  = t * s / np.sqrt(n)
        print(f"  {nom}: mitjana={m:.4f} / IC=({m-mg:.4f}, {m+mg:.4f}) / amplada={2*mg:.4f}")
        
    print(f"\nIntervals de confiança (N={N}, ρ_obj={rho_obj}, C_max={C_max}):")
    ic(rho_vals, "rho")
    ic(L_vals,   "L  ")
    ic(Y_vals,   "Y  ")
    ic(A_vals,   "A  ")
    ic(WW0_vals, "W/W0")
    

# =================================================
# TAULA DE RESULTATS PER DIFERENTS VALORS DE RHO 
# ==================================================
def generar_taula_i_exportar(N, nomfitxer, inici, metro):
    rhos = [0.2, 0.4, 0.6, 0.8, 0.95, 0.98]
    dades_completes = {}
    
    for rho_obj in rhos:
        lambd = calcular_lambda_objetiu(rho_obj)
        res = simular_sistema(N, lambd, burn_in, inici, metro)
        met = calcular_metriques(res, N)
        met["passatgers_totals"] = res["passatgers_totals"]
        print(f"rho = {rho_obj}, rho_estimada = {met['0  rho_estimada']}, L = {met['12 L']}, L' = {met["13 L'"]}, W = {met['26 W']}, W' = {met["27 W'"]}, W/W0 = {met['24 W/W0']}, W'/W0 = {met["25 W'/W0"]}, A_bar = {met['11 A_bar']}, Y_bar = {met['18 Y_bar']}, S_bar = {met['8  S_bar']}, gamma_bar = {met['22 z_bar']}, c_bar = {met['6  c_bar']}, W0 = {met['3  W0']}, tau_bar = {met['1  tau_bar']}, x_bar = {met['4  x_bar']}")
        dades_completes[rho_obj] = met
        
    # Llamamos a la función de renderizado
    exportar_taula_a_imatge(dades_completes, N, mitjana_tau, mitjana_c, C_max, mu, filename= nomfitxer)


# ===========================
# PPARÀMETRES DE SIMULACIÓ
# ===========================

N = 3000
burn_in =  N//2
np.random.seed(42)    # llavor per garantir la reproductibilitat dels resultats

# ===========================
# EXECUCIÓ DELS EXPERIMENTS
# ===========================

# PARADA DE BUS A PRINCIPI DE LINEA
# busos arriben cada 12 mins, capacitat mitjana = 50, capacitat màxima = 50 (venen buits)
# passatgers triguen 5 segons en pujar (1/5 clients per segon, o 12 clients per minut)
mitjana_tau = 12      
mitjana_c = 50
C_max = 50
mu = 12
E_gamma_global, E_x_global = estimar_gamma_x( mitjana_c, mu, inici = True, metro = False)
print("\nExperiment 1: (busos arriben cada 12 mins, capacitat mitjana = 50, capacitat màxima = 50)")
print(f"Estimació global de gamma: {E_gamma_global}, Estimació global de x: {E_x_global}")
generar_taula_i_exportar(N, nomfitxer="simulacio-50-50-12.png", inici = True, metro = False)

# PARADA DE BUS A PRINCIPI DE LINEA 
# busos arriben cada 5 mins, capacitat mitjana = 50, capacitat màxima = 50 (venen buits)
# passatgers triguen 5 segons en pujar (1/5 clients per segon, o 12 clients per minut)
mitjana_tau = 5
mitjana_c = 50
C_max = 50
mu = 12
E_gamma_global, E_x_global = estimar_gamma_x( mitjana_c, mu, inici = True, metro = False)
print("\nExperiment 2: (busos arriben cada 5 mins, capacitat mitjana = 50, capacitat màxima = 50)")   
print(f"Estimació global de gamma: {E_gamma_global}, Estimació global de x: {E_x_global}")
generar_taula_i_exportar(N, nomfitxer="simulacio-50-50-5.png", inici = True, metro = False)

# PARADA DE BUS A MEITAT DE LINEA
# busos arriben cada 12 mins, capacitat mitjana = 30, capacitat màxima = 50
# passatgers triguen 5 segons en pujar (1/5 clients per segon, o 12 clients per minut)
mitjana_tau = 12      
mitjana_c = 30
C_max = 50
mu = 12
E_gamma_global, E_x_global = estimar_gamma_x( mitjana_c, mu, inici = False, metro = False)
print("\nExperiment 3: (busos arriben cada 12 mins, capacitat mitjana = 30, capacitat màxima = 50)")
print(f"Estimació global de gamma: {E_gamma_global}, Estimació global de x: {E_x_global}")
generar_taula_i_exportar(N, nomfitxer="simulacio-50-30-12.png", inici = False, metro = False)

# PARADA DE BUS A MEITAT DE LINEA 
# busos arriben cada 5 mins, capacitat mitjana = 30, capacitat màxima = 50
# passatgers triguen 5 segons en pujar (1/5 clients per segon, o 12 clients per minut)
mitjana_tau = 5
mitjana_c = 30
C_max = 50
mu = 12
E_gamma_global, E_x_global = estimar_gamma_x( mitjana_c, mu, inici = False, metro = False)
print("\nExperiment 4: (busos arriben cada 5 mins, capacitat mitjana = 30, capacitat màxima = 50)")
print(f"Estimació global de gamma: {E_gamma_global}, Estimació global de x: {E_x_global}")
generar_taula_i_exportar(N, nomfitxer="simulacio-50-30-5.png", inici = False, metro = False)

# PARADA DE BUS A MEITAT DE LINEA
# busos arriben cada 12 mins, capacitat mitjana = 50, capacitat màxima = 75
# passatgers triguen 5 segons en pujar (1/5 clients per segon, o 12 clients per minut)
mitjana_tau = 12 
mitjana_c = 50
C_max = 75
mu = 12
E_gamma_global, E_x_global = estimar_gamma_x( mitjana_c, mu, inici = False, metro = False)
print("\nExperiment 5: (busos arriben cada 12 mins, capacitat mitjana = 50, capacitat màxima = 75)")
print(f"Estimació global de gamma: {E_gamma_global}, Estimació global de x: {E_x_global}")
generar_taula_i_exportar(N, nomfitxer="simulacio-75-50-12.png", inici = False, metro = False)

# PARADA DE BUS A MEITAT DE LINEA
# busos arriben cada 5 mins, capacitat mitjana = 50, capacitat màxima = 75
# passatgers triguen 5 segons en pujar (1/5 clients per segon, o 12 clients per minut)
mitjana_tau = 5
mitjana_c = 50
C_max = 75
mu = 12
E_gamma_global, E_x_global = estimar_gamma_x( mitjana_c, mu, inici = False, metro = False)
print("\nExperiment 6: (busos arriben cada 5 mins, capacitat mitjana = 50, capacitat màxima = 75)")
print(f"Estimació global de gamma: {E_gamma_global}, Estimació global de x: {E_x_global}")
generar_taula_i_exportar(N, nomfitxer="simulacio-75-50-5.png", inici = False, metro = False)

# PARADA DE METRO A MEITAT DE LINEA
# metros arriben cada 4 mins de manera constant, capacitat mitjana = 50, capacitat màxima = 75
# passatgers triguen 5 segons en pujar però hi ha 8 portes (8/5 clients per segon, o 96 clients per minut)
mitjana_tau = 4
mitjana_c = 300
C_max = 800
mu = 96
E_gamma_global, E_x_global = estimar_gamma_x( mitjana_c, mu, inici = False, metro = True)
print("\nExperiment 7: (metros arriben cada 4 mins, capacitat mitjana = 300, capacitat màxima = 800)")
print(f"Estimació global de gamma: {E_gamma_global}, Estimació global de x: {E_x_global}")      
generar_taula_i_exportar(N, nomfitxer="simulacio-800-300-4.png", inici = False, metro = True)


# ===========================
# EXPERIMENTS D'INTERVALS DE CONFIANÇA
# =========================

mitjana_tau = 12 
mitjana_c = 50
C_max = 75
mu = 12

E_gamma_global, E_x_global = estimar_gamma_x( mitjana_c, mu, inici = False, metro = False)

experiment_intervals(N = 100, rho_obj = 0.6, inici = False, metro = False)
experiment_intervals(N = 1000, rho_obj = 0.6, inici = False, metro = False)
experiment_intervals(N = 3000, rho_obj = 0.6, inici = False, metro = False)

experiment_intervals(N = 100, rho_obj = 0.98, inici = False, metro = False)
experiment_intervals(N = 1000, rho_obj = 0.98, inici = False, metro = False)
experiment_intervals(N = 3000, rho_obj = 0.98, inici = False, metro = False)
