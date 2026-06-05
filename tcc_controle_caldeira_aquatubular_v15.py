"""
=============================================================================
TCC - Aplicação de Controle Fuzzy na Estabilização do Nível em
Caldeiras Aquatubulares Frente aos Efeitos Shrink e Swell

Autor  : Lucas Laporti Pinto
Curso  : Engenharia de Controle e Automação - Ifes
Ref.   : Bianco, A. F.; Ferreira, V. R. A.; Mattioli, L. R.
         "Modelo Matemático no Espaço de Estados de uma Caldeira de Vapor
          Aquatubular". XXXV CNMAC, Natal-RN, 2014.

=============================================================================
ORGANIZAÇÃO DO CÓDIGO
=============================================================================

  BLOCO 1  - Parâmetros físicos e de construção da caldeira (Araxá-MG)
  BLOCO 2  - Propriedades termodinâmicas em função da pressão de operação
  BLOCO 3  - Ponto de operação: cálculo das condições iniciais (Eq. 18)
  BLOCO 4  - Modelo não linear no espaço de estados (Eq. 2–8 do artigo)
  BLOCO 5  - Matrizes do sistema linearizado A, B, C (Eq. 19–21 do artigo)
  BLOCO 6  - Verificação de controlabilidade
  BLOCO 7  - Configuração da simulação e representação de shrink e swell
  BLOCO 8  - Controlador PID (com anti-windup)
  BLOCO 9  - Controlador fuzzy Mamdani (duas entradas e 15 regras)
  BLOCO 10 - Simulação comparativa PID e Fuzzy
  BLOCO 11 - Métricas de desempenho (IAE, ISE, ITAE)
  BLOCO 12 - Geração e exportação das figuras

=============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')

PASTA_SAIDA = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
NOME_BASE = os.path.splitext(os.path.basename(__file__))[0] if '__file__' in globals() else 'tcc_controle_caldeira_aquatubular_v15'

def salvar_figura(fig, sufixo):
    caminho = os.path.join(PASTA_SAIDA, f'{NOME_BASE}_{sufixo}.png')
    fig.savefig(caminho, dpi=300, bbox_inches='tight')
    print(f'  Figura salva: {caminho}')


def fmt_num(valor, casas=2, remover_zeros=False):
    s = f'{valor:.{casas}f}'
    if remover_zeros:
        s = s.rstrip('0').rstrip('.')
    return s.replace('.', ',')


def fmt_tick(valor, pos=None):
    if abs(valor) < 1e-12:
        valor = 0.0
    if np.isclose(valor, round(valor), atol=1e-9):
        return f'{int(round(valor))}'
    for casas in (1, 2, 3):
        if np.isclose(valor, round(valor, casas), atol=10**(-(casas + 2))):
            return f'{valor:.{casas}f}'.replace('.', ',')
    return f'{valor:.2f}'.replace('.', ',')


def aplicar_formatacao_eixos(ax, formatar_x=True, formatar_y=True):
    formatter = FuncFormatter(fmt_tick)
    if formatar_x:
        ax.xaxis.set_major_formatter(formatter)
    if formatar_y:
        ax.yaxis.set_major_formatter(formatter)


# =============================================================================
# BLOCO 1 - PARÂMETROS FÍSICOS E DE CONSTRUÇÃO DA CALDEIRA (Araxá-MG)
# =============================================================================
# Fonte: parâmetros apresentados na Seção 3.2 do artigo-base para a planta
# industrial localizada em Araxá, Minas Gerais.
# =============================================================================

Vt   = 61.49      # Volume total do sistema [m³]
Vdc  = 9.00       # Volume dos downcomers (tubos de circulação de água) [m³]
Vr   = 18.59      # Volume dos risers (tubos de subida água/vapor) [m³]
Ad   = 14.08      # Área da superfície líquida no tubulão [m²]
Adc  = 0.234      # Área da seção transversal dos downcomers [m²]
Cp   = 550.0      # Calor específico do metal [J/(kg·°C)]
mr   = 61000.0    # Massa dos risers [kg]
md   = 29100.0    # Massa do tubulão (drum) [kg]
mt   = 147100.0   # Massa total do sistema [kg]
g    = 9.81       # Aceleração gravitacional [m/s²]
k    = 25.0       # Coeficiente de fricção do loop downcomer-riser [-]
beta = 0.3        # Parâmetro empírico (ajustado por testes na planta) [-]
Td   = 10.0       # Tempo de residência do vapor no tubulão [s]

# Dados operacionais e condições de contorno (Seção 3.2)
qs       = 27.78        # Vazão mássica de saída de vapor [kg/s] (~100 ton/h)
Vsd0     = 10.88        # Volume de vapor no tubulão na situação sem condensação [m³]
hf       = 450700.0     # Entalpia da água de alimentação [J/kg]  (450,7 kJ/kg)
p_op     = 4.74e6       # Pressão de operação [Pa] = 4,74 MPa


# =============================================================================
# BLOCO 2 - PROPRIEDADES TERMODINÂMICAS EM FUNÇÃO DA PRESSÃO
# =============================================================================
# As propriedades do vapor e da água saturada são aproximadas por polinômios
# ajustados em função da pressão p [Pa]. Fonte: Seção 3.2 do artigo.
#
# Notação:
#   rho_w = densidade da água saturada [kg/m³]
#   rho_s = densidade do vapor saturado [kg/m³]
#   h_w   = entalpia da água saturada [J/kg]
#   h_s   = entalpia do vapor saturado [J/kg]
#   ts    = temperatura de saturação [°C]
# =============================================================================

def ts_sat(p):
    """Temperatura de saturação [°C]   t(p) = 1,31e-5·p + 198,59"""
    return 1.31e-5 * p + 198.59

def rho_w(p):
    """Densidade da água saturada [kg/m³]   ρ_w(p) = -2,05e-5·p + 880,23"""
    return -2.05e-5 * p + 880.23

def rho_s(p):
    """Densidade do vapor saturado [kg/m³]   ρ_s(p) = 5,31e-6·p - 1,19"""
    return 5.31e-6 * p - 1.19

def h_w(p):
    """Entalpia da água saturada [J/kg]   h_w(p) = 305,84·ln(p) - 3563,10"""
    return 305.84 * np.log(p) - 3563.10

def h_s(p):
    """Entalpia do vapor saturado [J/kg]   h_s(p) = -1,58e-12·p² + 7,34e-6·p + 2796,6"""
    return -1.58e-12 * p**2 + 7.34e-6 * p + 2796.6

# Derivadas das propriedades em relação à pressão (necessárias para os coeficientes e_ij)
def drho_w_dp(p):
    """dρ_w/dp"""
    return -2.05e-5

def drho_s_dp(p):
    """dρ_s/dp"""
    return 5.31e-6

def dh_w_dp(p):
    """dh_w/dp"""
    return 305.84 / p

def dh_s_dp(p):
    """dh_s/dp"""
    return -2 * 1.58e-12 * p + 7.34e-6

def dts_dp(p):
    """dt_s/dp"""
    return 1.31e-5


# =============================================================================
# BLOCO 3 - PONTO DE OPERAÇÃO: CONDIÇÕES INICIAIS (Equação 18 do artigo)
# =============================================================================
# Com a caldeira em operação à pressão p = 4,74 MPa, as condições iniciais
# são obtidas anulando as derivadas das equações de estado (regime permanente).
#
# Sistema de equações (Eq. 18):
#   qf  = qs                          → vazão de entrada = saída (equilíbrio de massa)
#   Q   = qs·hs - qf·hf              → balanço de energia (calor necessário)
#   Q   = qdc·alpha_r·hc             → define o fluxo nos downcomers
#   Vsd = Vsd0 - Td·(hw-hf)/(ρs·hc)·qf  → volume de vapor no tubulão
#
# Os valores numéricos abaixo são os calculados no artigo (Seção 3.2).
# =============================================================================

p0     = p_op                        # Pressão de operação [Pa]

# Propriedades termodinâmicas no ponto de operação
ts0    = ts_sat(p0)                  # 260,68 °C
rho_w0 = rho_w(p0)                   # 983,06 kg/m³
rho_s0 = rho_s(p0)                   # 23,98 kg/m³
hw0    = h_w(p0)                     # 1.138,13 kJ/kg → convertido em J/kg abaixo
hs0    = h_s(p0)                     # 2.795,89 kJ/kg → em J/kg abaixo

# As funções h_w e h_s retornam valores em kJ/kg, conforme os polinômios
# apresentados no artigo-base. A conversão para J/kg é realizada multiplicando
# os valores por 1000.
hw0  = hw0 * 1000.0    # → J/kg    (1.138.130 J/kg)
hs0  = hs0 * 1000.0    # → J/kg    (2.795.890 J/kg)
hf_J = hf              # já em J/kg (450.700 J/kg)
hc0  = hs0 - hw0       # Entalpia de vaporização [J/kg]  hc = hs - hw

# Condições iniciais de equilíbrio (regime permanente)
qf0  = qs                            # 27,78 kg/s
Q0   = qs * hs0 - qf0 * hf_J        # Fluxo de calor [W] = 65,15 MW (≈ artigo)
Vwt0 = Vt - Vr                       # Volume total de água [m³] = 42,90 m³ (≈ artigo: 31,21 m³ - ver nota)
# Nota: O artigo cita Vwt = 31,21 m³. Essa diferença existe porque o artigo
# resolve um sistema não linear completo (Eq. 7 e 8) para encontrar alpha_r e qdc,
# o que exige um procedimento iterativo. Neste código, Vt - Vr é adotado como
# aproximação inicial compatível com a estrutura do modelo.

# Qualidade do vapor e volume no tubulão (calculados via Eq. 7 e 8)
# alpha_r = 0,036 conforme artigo
alpha_r0 = 0.036

# Fração volumétrica média de vapor nos risers (Eq. 7)
def alpha_v_bar(p, alpha_r):
    """
    Fração volumétrica média de vapor nos risers.
    Eq. (7): ᾱ_v = (ρ_w/(ρ_w-ρ_s)) · (1 - ρ_s/((ρ_w-ρ_s)·α_r) · ln(1 + (ρ_w-ρ_s)/ρ_s · α_r))
    """
    rw = rho_w(p)
    rs = rho_s(p)
    if (rw - rs) < 1e-9 or alpha_r < 1e-9:
        return 0.0
    term = (rw - rs) / rs * alpha_r
    if term <= -1:
        return 0.0
    av = (rw / (rw - rs)) * (1.0 - (rs / ((rw - rs) * alpha_r)) * np.log(1.0 + term))
    return np.clip(av, 0.0, 1.0)

# Fluxo nos downcomers (Eq. 8)
def Q_dc(p, alpha_r, alpha_v_b):
    """
    Fluxo mássico nos downcomers [kg/s].
    Eq. (8): Q_dc = α_r · h_c · √(2·ρ_w·A_dc·(ρ_w-ρ_s)·g·ᾱ_v·V_r / k)
    """
    rw = rho_w(p)
    rs = rho_s(p)
    hc = h_s(p)*1000 - h_w(p)*1000
    inner = 2.0 * rw * Adc * (rw - rs) * g * alpha_v_b * Vr / k
    if inner < 0:
        return 0.0
    return alpha_r * hc * np.sqrt(inner)

# Volume de água no tubulão descontando vapor nos risers (Eq. 6)
def V_wd(Vwt, p, alpha_r):
    """
    Volume de água no tubulão (downside).
    Eq. (6): V_wd = V_wt - V_dc - (1 - ᾱ_v)·V_r
    """
    av_b = alpha_v_bar(p, alpha_r)
    return Vwt - Vdc - (1.0 - av_b) * Vr

# Volume de vapor no tubulão no ponto de operação
av_b0  = alpha_v_bar(p0, alpha_r0)
Vsd0_calc = Vsd0 - Td * (hw0 - hf_J) / (rho_s0 * hc0) * qf0
# Valor adotado conforme apresentado no artigo-base:
Vsd_op = 6.08     # [m³] - valor explícito do artigo (Seção 3.2)

# Vetor de estado inicial x0 = [Vwt, p, alpha_r, Vsd]ᵀ
x0 = np.array([Vwt0, p0, alpha_r0, Vsd_op])

# Vetor de entrada de equilíbrio u0 = [qf, qs, Q]ᵀ
u0 = np.array([qf0, qs, Q0])

print("=" * 65)
print("  PONTO DE OPERAÇÃO: Caldeira de Araxá-MG")
print("=" * 65)
print(f"  Pressão              p   = {p0/1e6:.2f} MPa")
print(f"  Temperatura sat.     ts  = {ts0:.2f} °C")
print(f"  Densidade água       ρ_w = {rho_w0:.2f} kg/m³")
print(f"  Densidade vapor      ρ_s = {rho_s0:.2f} kg/m³")
print(f"  Entalpia água        h_w = {hw0/1000:.2f} kJ/kg")
print(f"  Entalpia vapor       h_s = {hs0/1000:.2f} kJ/kg")
print(f"  Calor latente        h_c = {hc0/1000:.2f} kJ/kg")
print(f"  Vazão entrada        qf  = {qf0:.2f} kg/s")
print(f"  Fluxo de calor       Q   = {Q0/1e6:.3f} MW")
print(f"  Qualidade do vapor   α_r = {alpha_r0:.3f}")
print(f"  Volume vapor tubulão   Vsd = {Vsd_op:.2f} m³")
print(f"  Fração vol. média    ᾱ_v = {av_b0:.4f}")


# =============================================================================
# BLOCO 4 - MODELO NÃO LINEAR NO ESPAÇO DE ESTADOS (Equações 2–8 do artigo)
# =============================================================================
# O modelo completo da caldeira é descrito pelo sistema:
#
#   E(x(t)) · ẋ(t) = f(x(t), u(t))          [Eq. 4]
#   y(x(t)) = l(x(t))                        [Eq. 5]
#
# onde x = [Vwt, p, α_r, Vsd]ᵀ  e  u = [qf, qs, Q]ᵀ
#
# As quatro equações diferenciais (Eq. 2) são:
#   e11·dVwt/dt + e12·dp/dt          = qf - qs
#   e21·dVwt/dt + e22·dp/dt          = Q + qf·hf - qs·hs
#   e32·dp/dt   + e33·dα_r/dt        = Q - α_r·hc·qdc
#   e42·dp/dt   + e43·dα_r/dt + e44·dVsd/dt = ρs·(Vsd0-Vsd)/Td + (hf-hw)/hc·qf
#
# A saída y é o nível da água no tubulão [m], calculado por l(x) - Eq. 15.
# =============================================================================

def calcular_coeficientes_E(x, p):
    """
    Calcula os coeficientes e_ij da matriz E(x) - Equação (3) do artigo.
    Estes coeficientes dependem do estado atual do sistema.
    """
    Vwt, p_, alpha_r, Vsd = x
    rw  = rho_w(p_)
    rs  = rho_s(p_)
    hw_ = h_w(p_) * 1000
    hs_ = h_s(p_) * 1000
    hc_ = hs_ - hw_
    ts_ = ts_sat(p_)

    drw = drho_w_dp(p_)
    drs = drho_s_dp(p_)
    dhw = dh_w_dp(p_) * 1000
    dhs = dh_s_dp(p_) * 1000
    dts = dts_dp(p_)

    av_b = alpha_v_bar(p_, alpha_r)
    Vwd_ = V_wd(Vwt, p_, alpha_r)

    # Derivada da fração volumétrica média em relação à pressão (numérica)
    dp_num = p_ * 1e-5
    dav_dp = (alpha_v_bar(p_ + dp_num, alpha_r) - alpha_v_bar(p_ - dp_num, alpha_r)) / (2 * dp_num)

    # Derivada da fração volumétrica média em relação a alpha_r (numérica)
    da_num = max(alpha_r * 1e-4, 1e-8)
    dav_dar = (alpha_v_bar(p_, alpha_r + da_num) - alpha_v_bar(p_, alpha_r - da_num)) / (2 * da_num)

    # Coeficientes e_ij - Equação (3) do artigo
    e11 = rw - rs
    e12 = Vwt * drw + Vr * drs          # Simplificado: Vst = Vwt, Vsd ignorado no e12

    e21 = rw * hw_ + rs * hs_
    e22 = (Vwt * (hw_ * drw + rw * dhw) +
           (hs_ * drs + rs * dhs) * Vr -
           Vt + mt * Cp * dts)

    e32 = ((rw * dhw / dts - alpha_r * hc_ * drw / dts) * (1 - av_b) * Vr
           + ((1 - alpha_r) * hc_ * drs + rs * dhs) * av_b * Vr
           + (rs + (rw - rs) * alpha_r) * hc_ * Vr * dav_dp
           - Vr + mr * Cp * dts / dts)

    e33 = ((1 - av_b) * rs + alpha_r * rw) * hc_ * Vr * dav_dar

    e42 = (Vsd * drs
           + (1 / hc_) * (rs * Vsd * dhs + rw * Vwd_ * dhw - Vsd - Vwd_ + md * Cp * dts)
           + alpha_r * (1 + beta) * Vr)

    e43 = alpha_r * (1 + beta) * (rs - rw) * Vr * dav_dar

    e44 = rs

    return e11, e12, e21, e22, e32, e33, e42, e43, e44


def f_nao_linear(x, u, p):
    """
    Lado direito do sistema não linear: E(x)·ẋ = f(x,u) - Equações (2) e (5).

    Retorna o vetor f(x,u) = [f1, f2, f3, f4]ᵀ
    """
    Vwt, p_, alpha_r, Vsd = x
    qf, qs_, Q_ = u

    hs_ = h_s(p_) * 1000
    hw_ = h_w(p_) * 1000
    rs  = rho_s(p_)
    hc_ = hs_ - hw_

    av_b = alpha_v_bar(p_, alpha_r)
    qdc  = Q_dc(p_, alpha_r, av_b)

    f1 = qf - qs_
    f2 = Q_ + qf * hf_J - qs_ * hs_
    f3 = Q_ - alpha_r * hc_ * qdc
    f4 = rs * (Vsd0 - Vsd) / Td + (hf_J - hw_) / hc_ * qf

    return np.array([f1, f2, f3, f4])


def montar_E(x):
    """
    Monta a matriz E(x) (4×4) a partir dos coeficientes e_ij.
    Eq. (5) do artigo: E(x)·ẋ = f(x,u)
    """
    e11, e12, e21, e22, e32, e33, e42, e43, e44 = calcular_coeficientes_E(x, x[1])

    E = np.array([
        [e11,  e12,   0.0,   0.0],
        [e21,  e22,   0.0,   0.0],
        [0.0,  e32,   e33,   0.0],
        [0.0,  e42,   e43,   e44],
    ])
    return E


def nivel_balao(x):
    """
    Saída do sistema: nível da água no tubulão [m].
    l(x) = (V_wd + V_sd) / A_d   - Equação (15) do artigo.
    """
    Vwt, p_, alpha_r, Vsd = x
    Vwd_ = V_wd(Vwt, p_, alpha_r)
    return (Vwd_ + Vsd) / Ad


# =============================================================================
# BLOCO 5 - MATRIZES DO SISTEMA LINEARIZADO A, B, C (Eqs. 19–21 do artigo)
# =============================================================================
# O artigo lineariza o sistema em torno do ponto de operação calculando as
# matrizes Jacobianas J1 (∂f/∂x), J2 (∂f/∂u) e J3 (∂l/∂x).
#
# O sistema linearizado é:
#   ẋ̃ = A·x̃ + B·ũ       com  A = E0⁻¹·J1  e  B = E0⁻¹·J2   [Eq. 12]
#   ỹ  = C·x̃             com  C = J3                           [Eq. 16]
#
# Os valores abaixo são transcritos do artigo-base (Equações 19, 20 e 21).
# =============================================================================

# Matriz A = E0⁻¹·J1  [4×4]  - Equação (19) do artigo
A_mat = np.array([
    [0.0,   0.0,   998.008e-18,   0.0  ],
    [0.0,   0.0,  -1.046e-9,      0.0  ],
    [0.0,   0.0,  -224.252e-3,    0.0  ],
    [0.0,   0.0,   146.568,      -0.1  ],
])

# Matriz B = E0⁻¹·J2  [4×3]  - Equação (20) do artigo
B_mat = np.array([
    [ 783.94e-6,   -1.723e-3,    400.516e-12],
    [-518.015,     -1.363e3,     801.905e-6 ],
    [  19.501e-6,   51.297e-6,    62.745e-12],
    [  -9.302e-3,   21.018e-3,     6.929e-9 ],
])

# Matriz C = J3  [1×4] transposta → linha  - Equação (21) do artigo
C_mat = np.array([[71.023e-3,  0.0,  7.879,  71.023e-3]])

print("\n" + "=" * 65)
print("  MATRIZES DO ESPAÇO DE ESTADOS - Sistema Linearizado")
print("=" * 65)
print("\nMatriz A (4×4):")
for linha in A_mat:
    print("  [" + "  ".join(f"{v: .4e}" for v in linha) + "]")
print("\nMatriz B (4×3):")
for linha in B_mat:
    print("  [" + "  ".join(f"{v: .4e}" for v in linha) + "]")
print("\nMatriz C (1×4):")
print("  [" + "  ".join(f"{v: .4e}" for v in C_mat[0]) + "]")


# =============================================================================
# BLOCO 6 - VERIFICAÇÃO DE CONTROLABILIDADE
# =============================================================================
# Um sistema é controlável se, a partir de qualquer estado inicial, é possível
# levá-lo a qualquer estado final em tempo finito com uma entrada adequada.
#
# A matriz de controlabilidade é definida como:
#   C = [B  A·B  A²·B  A³·B]   (para n=4 estados)
#
# O sistema é controlável se e somente se rank(C) = n = 4.
# O artigo confirma isso na Seção 3 (após Eq. 23).
# =============================================================================

def matriz_controlabilidade(A, B, n):
    """Calcula a matriz de controlabilidade C = [B AB A²B ... Aⁿ⁻¹B]"""
    colunas = [B]
    Ak = A.copy()
    for _ in range(n - 1):
        colunas.append(Ak @ B)
        Ak = Ak @ A
    return np.hstack(colunas)

Mc   = matriz_controlabilidade(A_mat, B_mat, 4)
rank = np.linalg.matrix_rank(Mc)

print("\n" + "=" * 65)
print("  VERIFICAÇÃO DE CONTROLABILIDADE")
print("=" * 65)
print(f"  Ordem do sistema (n)       : 4")
print(f"  Rank da matriz C [4×12]    : {rank}")
print(f"  Sistema controlável        : {'SIM ✓' if rank == 4 else 'NÃO ✗'}")
print("  (Conforme verificado no artigo via Matlab®, Seção 3)")


# =============================================================================
# BLOCO 7 - CONFIGURAÇÃO DA SIMULAÇÃO E EFEITOS SHRINK & SWELL
# =============================================================================
# A simulação usa o MODELO LINEARIZADO (matrizes A, B, C do artigo),
# integrado pelo método de Euler de passo fixo sobre as variáveis
# perturbadas em torno do ponto de operação:
#
#   ẋ̃ = A·x̃ + B·ũ    →  ẋ̃[k+1] = x̃[k] + dt·(A·x̃[k] + B·ũ[k])
#   ỹ  = C·x̃
#
# onde x̃ = x - x0  e  ũ = u - u0  são as perturbações em relação ao
# ponto de operação. A saída ỹ é o desvio de nível em metros.
#
# A representação linearizada corresponde ao modelo utilizado nas análises
# comparativas e no desenvolvimento dos controladores.
#
# EFEITOS SHRINK & SWELL (resposta inversa):
#   Quando a demanda de vapor aumenta → pressão interna cai → bolhas
#   nos risers se expandem → nível SOBE momentaneamente (SWELL).
#   Quando a demanda cai → pressão sobe → bolhas se comprimem → nível
#   CAI momentaneamente (SHRINK).
#   Modelagem: y_medido = y_real + α_sw · d(ũ_qs)/dt
# =============================================================================

# --- Configuração da simulação ---
dt   = 1.0          # Passo de integração [s]
T    = 1800.0       # Tempo total [s] = 30 minutos
N    = int(T / dt)
t    = np.linspace(0, T, N)

# Nível no ponto de operação [m]  (l(x0) = (Vwd + Vsd) / Ad)
y0       = (V_wd(x0[0], x0[1], x0[2]) + Vsd_op) / Ad   # ≈ 1,97 m
setpoint = 0.0       # No modelo linearizado, setpoint = 0 (manter no ponto de op.)

# Intensidade do efeito shrink/swell [m·s/kg]
# Valor empírico calibrado para produzir desvio de ~5-8% do nível de operação
alpha_sw = 0.035

# --- Perfil de PERTURBAÇÃO na demanda de vapor ũ_qs(t) [kg/s] ---
# O perfil de demanda de vapor inclui dois aumentos de carga e dois retornos
# ao valor nominal. Os intervalos foram definidos para permitir a análise da
# resposta transitória e da recuperação do sistema.
#
# Os degraus são definidos como níveis-alvo percentuais em torno do valor
# nominal de qs e depois suavizados por uma dinâmica de 1ª ordem, evitando
# descontinuidades abruptas que gerariam derivações infinitas no termo de
# shrink/swell.
DEGRAUS_QS = [
    (0,    0.00, 'regime nominal'),
    (300,  0.12, 'degrau +12%'),
    (720,  0.00, 'retorno ao nominal'),
    (1140, 0.08, 'degrau +8%'),
    (1560, 0.00, 'retorno ao nominal'),
]

def nivel_alvo_qs(tk):
    """Nível-alvo da perturbação de demanda de vapor (fração de qs nominal)."""
    frac = 0.0
    for t_ini, frac_alvo, _ in DEGRAUS_QS:
        if tk >= t_ini:
            frac = frac_alvo
        else:
            break
    return frac * qs

# Vetor-alvo em degraus e vetor efetivamente aplicado (suavizado)
u_qs_alvo = np.array([nivel_alvo_qs(tk) for tk in t])
u_qs_vec = np.zeros(N)
tau_qs = 50.0  # [s] suaviza as mudanças de carga sem esconder os degraus
for k in range(1, N):
    u_qs_vec[k] = u_qs_vec[k-1] + (dt / tau_qs) * (u_qs_alvo[k] - u_qs_vec[k-1])
qs_vec = qs + u_qs_vec  # demanda total (para plotar)


# =============================================================================
# BLOCO 8 - CONTROLADOR PID (com anti-windup)
# =============================================================================
# u(t) = Kp·e(t) + Ki·∫e(t)dt + Kd·de(t)/dt
#
# O PID ajusta a vazão de entrada qf em torno do valor de equilíbrio qf0.
# Anti-windup: suspende a integração quando a saída satura, evitando que
# o termo integral acumule indefinidamente fora dos limites físicos da válvula.
# =============================================================================

# Limites físicos da válvula de entrada [kg/s]
qf_min = 0.0
qf_max = 60.0     # Limite superior adotado para a vazão de alimentação [kg/s]

# Ganhos do PID ajustados para a dinâmica da planta de Araxá-MG:
#   - Nível em metros (~2 m no ponto de op.)
#   - Erro típico na ordem de 0,05–0,20 m
#   - Atuação em kg/s (qf ~ 27,78 kg/s nominal)
#   - Constante de tempo dominante do sistema: ~100–300 s
Kp_pid = 120.0    # Proporcional
Ki_pid = 0.15     # Integral
Kd_pid = 300.0    # Derivativo

def passo_pid(erro, erro_ant, integral, Kp, Ki, Kd, dt, u_eq=0.0,
              u_min=qf_min, u_max=qf_max):
    """
    Executa um passo do controlador PID discreto.

    A saída é calculada em torno do ponto de equilíbrio:
        u = u_eq + Kp·e + Ki·∫e·dt + Kd·de/dt

    Parâmetros:
        erro, erro_ant : erro atual e anterior
        integral       : acumulador da integral
        u_eq           : ponto de operação (feed-forward)
        u_min, u_max   : limites físicos do atuador

    Retorna:
        u        : ação de controle saturada
        integral : integral atualizada (com anti-windup)
    """
    integral += erro * dt
    derivada  = (erro - erro_ant) / dt
    u = u_eq + Kp * erro + Ki * integral + Kd * derivada

    # Anti-windup: desfaz a integração se a saída saturou
    if u > u_max or u < u_min:
        integral -= erro * dt

    return np.clip(u, u_min, u_max), integral


# =============================================================================
# BLOCO 9 - CONTROLADOR FUZZY COM INFERÊNCIA MAMDANI
# =============================================================================
# Estrutura adotada para o controlador:
#   - duas entradas: erro e derivada do erro;
#   - cinco conjuntos para o erro: NG, NP, ZE, PP, PG;
#   - três conjuntos para a variação do erro: NE, ZE, PO;
#   - cinco conjuntos para a saída: FE, FR, ND, AM, AE;
#     FE = fechamento elevado; FR = fechamento reduzido;
#     ND = não deslocar; AM = abertura moderada; AE = abertura elevada;
#   - inferência Mamdani com AND = mínimo e agregação = máximo;
#   - defuzzificação por centroide de área.
#
# Os universos de discurso são definidos em unidades físicas: o erro de nível
# é expresso em metros, sua derivada em metros por segundo e a saída corresponde
# à variação da vazão mássica de alimentação em kg/s. A distribuição assimétrica
# das ações de saída representa o conhecimento heurístico incorporado à estratégia
# de controle para lidar com os transitórios associados a shrink e swell.
# =============================================================================

E_MAX   = 0.22      # Erro máximo considerado [m]
DE_MAX  = 0.0025    # Variação máxima do erro considerada [m/s]
DQ_MIN = -18.0      # Delta mínimo de vazão de entrada [kg/s]
DQ_MAX =  18.0      # Delta máximo de vazão de entrada [kg/s]


def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


def trapezoidal(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1.0
    else:
        return (d - x) / (d - c)


def triangular_vetor(x, a, b, c):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    subida = (a < x) & (x <= b)
    descida = (b < x) & (x < c)
    if abs(b - a) > 1e-12:
        y[subida] = (x[subida] - a) / (b - a)
    if abs(c - b) > 1e-12:
        y[descida] = (c - x[descida]) / (c - b)
    return y


def trapezoidal_vetor(x, a, b, c, d):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    subida = (a < x) & (x < b)
    topo = (b <= x) & (x <= c)
    descida = (c < x) & (x < d)
    if abs(b - a) > 1e-12:
        y[subida] = (x[subida] - a) / (b - a)
    y[topo] = 1.0
    if abs(d - c) > 1e-12:
        y[descida] = (d - x[descida]) / (d - c)
    return y


def fuzzificar_erro(e):
    e = np.clip(e, -E_MAX, E_MAX)
    return {
        'NG': trapezoidal(e, -E_MAX, -E_MAX, -0.095, -0.025),
        'NP': triangular(e,  -0.070, -0.025,  0.000),
        'ZE': triangular(e,  -0.018,  0.000,  0.018),
        'PP': triangular(e,   0.000,  0.025,  0.070),
        'PG': trapezoidal(e,  0.025,  0.095, E_MAX, E_MAX),
    }


def fuzzificar_delta_erro(de):
    de = np.clip(de, -DE_MAX, DE_MAX)
    return {
        'NE': trapezoidal(de, -DE_MAX, -DE_MAX, -7.0e-04, -1.5e-04),
        'ZE': triangular(de,  -1.4e-03, 0.0, 1.4e-03),
        'PO': trapezoidal(de,  1.5e-04, 7.0e-04, DE_MAX, DE_MAX),
    }


def pertinencia_saida_delta_qf_vetor(universo):
    # A saída é assimétrica: a faixa de abertura é mais ampla para reduzir
    # quedas prolongadas do nível após perturbações na demanda de vapor, enquanto
    # o fechamento permanece mais contido durante o retorno à condição nominal.
    # Rótulos padronizados: FE, FR, ND, AM e AE.
    return {
        'FE': trapezoidal_vetor(universo, DQ_MIN, DQ_MIN, -6.0, -2.0),
        'FR': triangular_vetor(universo,  -4.2,   -1.8,   0.0),
        'ND': triangular_vetor(universo,  -1.4,    0.0,   1.4),
        'AM': triangular_vetor(universo,   0.0,   12.0,  18.0),
        'AE': trapezoidal_vetor(universo, 14.0,   16.5, DQ_MAX, DQ_MAX),
    }


def defuzzificar_centroide_mamdani(ativacoes, pontos=501):
    universo = np.linspace(DQ_MIN, DQ_MAX, pontos)
    mu_saida = pertinencia_saida_delta_qf_vetor(universo)
    agregado = np.zeros_like(universo)
    for rotulo, ativacao in ativacoes.items():
        agregado = np.maximum(agregado, np.minimum(ativacao, mu_saida[rotulo]))
    area = np.trapezoid(agregado, universo)
    if area < 1e-10:
        return 0.0
    return np.trapezoid(universo * agregado, universo) / area


BASE_DE_REGRAS = [
    ('NG', 'NE', 'FE'), ('NG', 'ZE', 'FE'), ('NG', 'PO', 'FR'),
    ('NP', 'NE', 'FE'), ('NP', 'ZE', 'FR'), ('NP', 'PO', 'ND'),
    ('ZE', 'NE', 'FR'), ('ZE', 'ZE', 'ND'), ('ZE', 'PO', 'AM'),
    ('PP', 'NE', 'AM'), ('PP', 'ZE', 'AM'), ('PP', 'PO', 'AE'),
    ('PG', 'NE', 'AE'), ('PG', 'ZE', 'AE'), ('PG', 'PO', 'AE'),
]


def controlador_fuzzy(erro, delta_erro):
    mu_e  = fuzzificar_erro(erro)
    mu_de = fuzzificar_delta_erro(delta_erro)
    ativacoes = {k: 0.0 for k in ['FE', 'FR', 'ND', 'AM', 'AE']}
    for (ce, cde, saida) in BASE_DE_REGRAS:
        grau = min(mu_e[ce], mu_de[cde])
        ativacoes[saida] = max(ativacoes[saida], grau)
    delta_qf = defuzzificar_centroide_mamdani(ativacoes)
    return np.clip(delta_qf, -qf0, qf_max - qf0)


# =============================================================================
# BLOCO 10 - SIMULAÇÃO COMPARATIVA ENTRE OS CONTROLADORES PID E FUZZY
# =============================================================================
# Os controladores são simulados sobre o mesmo modelo linearizado em espaço
# de estados e submetidos ao mesmo perfil de perturbações na demanda de vapor.
# Essa configuração permite comparar as estratégias sob condições equivalentes.
#
# A distorção associada a shrink e swell é incorporada ao nível medido, de modo
# que os controladores atuem a partir do sinal disponibilizado pelo sensor.
# =============================================================================

def simular(controlador='pid'):
    """
    Simula o sistema linearizado com o controlador escolhido.

    Usa as matrizes A, B, C do artigo (Eqs. 19–21).
    O estado x̃ e a saída ỹ representam DESVIOS em relação ao ponto de operação.

    Parâmetros:
        controlador : 'pid' ou 'fuzzy'

    Retorna:
        y_real   : desvio de nível real [m]  (positivo = acima do setpoint)
        y_medido : desvio medido com distorção shrink/swell [m]
        u_hist   : perturbação de controle ũ_qf(t) [kg/s]
        e_hist   : histórico do erro real [m]
    """
    # --- Estado inicial: x̃ = 0 (começa no ponto de operação) ---
    x_til    = np.zeros(4)   # perturbação dos estados [Vwt, p, α_r, Vsd]
    y_real   = np.zeros(N)
    y_medido = np.zeros(N)
    u_hist   = np.zeros(N)
    e_hist   = np.zeros(N)

    integral = 0.0
    e_ant    = 0.0

    # -------------------------------------------------------------------------
    # Dinâmica física da válvula de alimentação
    # -------------------------------------------------------------------------
    # O controlador calcula um comando de vazão (u_cmd), mas a planta recebe a
    # vazão efetivamente entregue pela válvula (u_qf). Em sistemas reais, a
    # válvula não muda instantaneamente; ela possui inércia mecânica, atrito e
    # tempo de resposta. Essa dinâmica é representada por um sistema de 1ª ordem:
    #
    #      tau_valvula * du_qf/dt + u_qf = u_cmd
    #
    # Quanto maior tau_valvula, mais lenta e suave é a resposta da válvula.
    # Essa representação limita microvariações de alta frequência não associadas
    # à dinâmica física do atuador, sem alterar a estrutura do controlador fuzzy.
    tau_valvula = 18.0  # [s] constante de tempo aproximada da válvula
    u_qf        = 0.0   # vazão real aplicada à planta (perturbação em torno de qf0)
    u_cmd       = 0.0   # comando calculado pelo controlador

    # Filtro exponencial para suavizar a saída do Fuzzy antes da dinâmica da válvula.
    alpha_filtro_fuzzy = 0.0350
    u_fuzzy_filtrado = 0.0

    for k in range(N - 1):
        # Saída do sistema linearizado: ỹ = C·x̃  [desvio de nível em metros]
        y_real[k] = float(np.dot(C_mat[0], x_til))

        # Efeito shrink/swell: perturbação no nível medido proporcional
        # à taxa de variação da demanda de vapor
        du_qs_dt = (u_qs_vec[k] - u_qs_vec[k-1]) / dt if k > 0 else 0.0
        y_medido[k] = y_real[k] + alpha_sw * du_qs_dt

        # Erro que o controlador vê (baseado no nível MEDIDO)
        e_k  = setpoint - y_medido[k]
        # Zona morta: erros menores que 0,5% do nível de op. são tratados como zero.
        # Essa condição reduz comutações sucessivas causadas por variações
        # microscópicas do erro próximo ao ponto de operação.
        zona_morta = 0.005 * y0
        e_k  = e_k if abs(e_k) > zona_morta else 0.0
        de_k = (e_k - e_ant) / dt

        # --- Comando de controle (perturbação em torno de qf0) ---
        if controlador == 'pid':
            # passo_pid retorna o comando de vazão em torno do equilíbrio
            u_abs, integral = passo_pid(e_k, e_ant, integral,
                                        Kp_pid, Ki_pid, Kd_pid, dt,
                                        u_eq=0.0,           # u_eq=0 pois ũ é perturbação
                                        u_min=float(-qf0),  # não pode fechar mais que qf0
                                        u_max=float(qf_max-qf0))
            u_cmd = float(u_abs)
        else:  # fuzzy
            delta_qf = controlador_fuzzy(e_k, de_k)
            u_fuzzy_filtrado = (alpha_filtro_fuzzy * delta_qf +
                                (1.0 - alpha_filtro_fuzzy) * u_fuzzy_filtrado)
            u_cmd = float(np.clip(u_fuzzy_filtrado, -qf0, qf_max - qf0))

        # Dinâmica da válvula: o comando não entra instantaneamente na planta.
        # A planta recebe u_qf, que segue u_cmd suavemente.
        u_qf = u_qf + (dt / tau_valvula) * (u_cmd - u_qf)
        u_qf = float(np.clip(u_qf, -qf0, qf_max - qf0))

        u_hist[k] = u_qf
        e_hist[k] = setpoint - y_real[k]
        e_ant      = e_k

        # --- Vetor de perturbação de entrada ũ = [ũ_qf, ũ_qs, 0] ---
        # Q é mantido constante (sem perturbação de calor)
        u_til = np.array([u_qf, u_qs_vec[k], 0.0])

        # --- Integração de Euler: x̃[k+1] = x̃[k] + dt·(A·x̃[k] + B·ũ[k]) ---
        dx_til     = A_mat @ x_til + B_mat @ u_til
        x_til      = x_til + dt * dx_til

    # Preenche último ponto
    y_real[-1]   = float(np.dot(C_mat[0], x_til))
    y_medido[-1] = y_real[-1]
    u_hist[-1]   = u_hist[-2]
    e_hist[-1]   = setpoint - y_real[-1]

    return y_real, y_medido, u_hist, e_hist


print("\n[1/2] Simulando PID...")
y_pid, ym_pid, u_pid, e_pid = simular('pid')

print("[2/2] Simulando Fuzzy...")
y_fuz, ym_fuz, u_fuz, e_fuz = simular('fuzzy')
print("Simulações concluídas.\n")


# =============================================================================
# BLOCO 11 - MÉTRICAS DE DESEMPENHO
# =============================================================================

def calcular_metricas(y, setpoint, dt):
    """
    Calcula IAE, ISE e ITAE a partir do histórico de nível.

    IAE  = ∫|e(t)|dt           → total de desvio acumulado
    ISE  = ∫e²(t)dt            → penaliza desvios grandes
    ITAE = ∫t·|e(t)|dt         → penaliza desvios persistentes

    Também calcula o máximo desvio percentual e o último instante fora da faixa de ±2%
    para faixa de ±2% do setpoint.
    """
    e    = setpoint - y
    IAE  = np.trapezoid(np.abs(e),          dx=dt)
    ISE  = np.trapezoid(e**2,               dx=dt)
    ITAE = np.trapezoid(t * np.abs(e),      dx=dt)

    # Máximo desvio percentual: maior afastamento absoluto em relação ao ponto nominal.
    # Esta métrica não corresponde ao overshoot clássico de uma resposta a degrau.
    pico = np.max(np.abs(y))
    max_desvio_pct = (pico / y0) * 100 if y0 > 0 else 0.0

    # Último instante fora da faixa de ±2%
    faixa = 0.02 * y0   # ±2% do nível de operação
    fora  = np.where(np.abs(e) > faixa)[0]
    t_ass = t[fora[-1]] if len(fora) > 0 else 0.0

    return IAE, ISE, ITAE, max_desvio_pct, t_ass

IAE_p, ISE_p, ITAE_p, md_p, ts_p = calcular_metricas(y_pid, setpoint, dt)
IAE_f, ISE_f, ITAE_f, md_f, ts_f = calcular_metricas(y_fuz, setpoint, dt)

print("=" * 65)
print("  MÉTRICAS DE DESEMPENHO")
print("=" * 65)
print(f"  {'Métrica':<20} {'PID':>12} {'Fuzzy':>12}")
print(f"  {'-'*44}")
print(f"  {'IAE [m·s]':<20} {IAE_p:>12.4f} {IAE_f:>12.4f}")
print(f"  {'ISE [m²·s]':<20} {ISE_p:>12.6f} {ISE_f:>12.6f}")
print(f"  {'ITAE [m·s²]':<20} {ITAE_p:>12.2f} {ITAE_f:>12.2f}")
print(f"  {'Máx. desvio [%]':<20} {md_p:>12.2f} {md_f:>12.2f}")
print(f"  {'Últ. fora ±2% [s]':<20} {ts_p:>12.0f} {ts_f:>12.0f}")


# =============================================================================
# BLOCO 12 - GERAÇÃO E EXPORTAÇÃO DAS FIGURAS
# =============================================================================
# Figuras principais:
#   Figura 1 - Propriedades termodinâmicas no ponto de operação
#   Figura 2 - Resposta da planta linearizada em malha aberta
#   Figura 3 - Resposta com PID sob influência de shrink e swell
#   Figura 4 - Comparação entre PID e fuzzy
#   Figura 5 - Métricas de desempenho
# =============================================================================

COR_PID    = '#1f77b4'   # Azul
COR_FUZZY  = '#ff7f0e'   # Laranja
COR_SET    = '#d62728'   # Vermelho
COR_MED    = '#9467bd'   # Roxo
COR_VAPOR  = '#8c564b'   # Marrom
COR_FAIXA  = '#2ca02c'   # Verde
COR_FEN_SWELL  = '#2b8a3e'   # Verde-escuro para anotações de swell
COR_FEN_SHRINK = '#a61e4d'   # Magenta/rosa escuro para anotações de shrink


# ── Figura 1: propriedades termodinâmicas em função da pressão ─────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(12, 8))

p_range = np.linspace(1e6, 8e6, 300)

ax = axes[0, 0]
ax.plot(p_range / 1e6, rho_w(p_range), color=COR_PID,   lw=2, label=r'$\rho_w$ (água)')
ax.plot(p_range / 1e6, rho_s(p_range), color=COR_FUZZY, lw=2, label=r'$\rho_s$ (vapor)')
ax.axvline(p_op / 1e6, color=COR_SET, ls='--', lw=1.2, label=rf'$p_{{op}}$ = {fmt_num(p_op/1e6, 2)} MPa')
ax.set_xlabel('Pressão [MPa]'); ax.set_ylabel('Densidade [kg/m³]')
ax.set_title('Densidade do fluido'); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

ax = axes[0, 1]
ax.plot(p_range / 1e6, h_w(p_range), color=COR_PID,   lw=2, label=r'$h_w$ (água)')
ax.plot(p_range / 1e6, h_s(p_range), color=COR_FUZZY, lw=2, label=r'$h_s$ (vapor)')
ax.axvline(p_op / 1e6, color=COR_SET, ls='--', lw=1.2, label=rf'$p_{{op}}$ = {fmt_num(p_op/1e6, 2)} MPa')
ax.set_xlabel('Pressão [MPa]'); ax.set_ylabel('Entalpia [kJ/kg]')
ax.set_title('Entalpias de saturação'); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

ax = axes[1, 0]
ax.plot(p_range / 1e6, ts_sat(p_range), color='#e377c2', lw=2)
ax.axvline(p_op / 1e6, color=COR_SET, ls='--', lw=1.2, label=rf'$t_s$ = {fmt_num(ts0, 1)} °C')
ax.set_xlabel('Pressão [MPa]'); ax.set_ylabel('Temperatura [°C]')
ax.set_title('Temperatura de saturação'); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

ax = axes[1, 1]
ar_range = np.linspace(0.001, 0.15, 200)
av_range = [alpha_v_bar(p_op, ar) for ar in ar_range]
ax.plot(ar_range, av_range, color='#17becf', lw=2)
ax.axvline(alpha_r0, color=COR_SET, ls='--', lw=1.2,
           label=rf'$\alpha_{{r,0}}$ = {fmt_num(alpha_r0, 3)}')
ax.axhline(av_b0, color=COR_SET, ls=':', lw=1.2,
           label=rf'$\bar{{\alpha}}_{{v,0}}$ = {fmt_num(av_b0, 4)}')
ax.set_xlabel(r'Qualidade do vapor $\alpha_r$')
ax.set_ylabel(r'Fração vol. média $\bar{\alpha}_v$')
ax.set_title('Fração volumétrica de vapor nos risers (Eq. 7)')
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

for ax in axes.flat:
    aplicar_formatacao_eixos(ax)

plt.tight_layout()


# ── Figura 2: Análise em malha aberta da planta linearizada ───────────────────
# Esta figura apresenta o comportamento da planta linearizada em malha aberta.
# A entrada de água qf é mantida
# fixa no ponto de operação, enquanto a demanda de vapor qs sofre perturbações.
# Como o modelo é analisado em torno do ponto de operação, o nível é apresentado
# como DESVIO relativo a y0, e não como nível físico absoluto.

# A análise em malha aberta utiliza a mesma representação linearizada adotada
# na comparação entre controladores. O estado perturbado começa em zero, isto é,
# exatamente no ponto de operação. Como qf permanece fixa, a única entrada
# perturbada é a demanda de vapor u_qs(t).
x_ma_til = np.zeros(4)
y_ma_dev = np.zeros(N)          # desvio em relação ao ponto de operação [m]
qf_ma = np.ones(N) * qf0        # entrada de água mantida constante [kg/s]
qs_ma = qs_vec.copy()           # demanda de vapor com perturbações [kg/s]
balanco_ma = qf_ma - qs_ma      # balanço mássico simplificado [kg/s]

for k in range(N - 1):
    y_ma_dev[k] = float(np.dot(C_mat[0], x_ma_til))

    # Malha aberta: a perturbação de qf é nula e não compensa a variação de qs.
    u_til_ma = np.array([0.0, u_qs_vec[k], 0.0])
    dx_til_ma = A_mat @ x_ma_til + B_mat @ u_til_ma
    x_ma_til = x_ma_til + dt * dx_til_ma

y_ma_dev[-1] = float(np.dot(C_mat[0], x_ma_til))

fig2, axes2 = plt.subplots(3, 1, figsize=(12.5, 10.0), sharex=True,
                           gridspec_kw={'height_ratios': [1.25, 1.0, 1.0]})

# Instantes em que ocorrem mudanças de demanda de vapor.
eventos_seg = [item[0] for item in DEGRAUS_QS[1:]]
eventos_min = [tx / 60 for tx in eventos_seg]
eventos_lbl = [item[2] for item in DEGRAUS_QS[1:]]

# Gráfico 1 - desvio de nível em relação ao ponto de operação
ax = axes2[0]
desvio_pct = y_ma_dev * 100 / (y0 if y0 > 0 else 1)
ax.plot(t / 60, desvio_pct, color=COR_PID, lw=2,
        label='Desvio estimado do nível em relação ao ponto de operação')
ax.axhline(0, color=COR_SET, ls='--', lw=1.4, label='Ponto de operação / setpoint')
ax.axhspan(-2, 2, alpha=0.08, color=COR_FAIXA, label='Faixa operacional ±2%')
for i, tx in enumerate(eventos_min):
    ax.axvline(tx, color='gray', ls=':', lw=1.1,
               label='mudanças de demanda' if i == 0 else None)
ax.set_ylabel('Desvio do nível\n[% de y0]')
# Anotações dos fenômenos associadas às mudanças de demanda.
# Posicionadas em regiões livres para não ficarem sobre as curvas.
posicoes_anotacoes_fig2 = [
    {'xytext': (4.35, 1.35),  'ha': 'right', 'va': 'bottom', 'rad': -0.18, 'cor': COR_FEN_SWELL,  'label': 'SWELL\n(+12% vapor)'},
    {'xytext': (13.25, 1.75), 'ha': 'left', 'va': 'bottom', 'rad': 0.12, 'cor': COR_FEN_SHRINK, 'label': 'SHRINK\n(-12% vapor)'},
    {'xytext': (18.10, -4.90),'ha': 'right', 'va': 'bottom', 'rad': -0.16, 'cor': COR_FEN_SWELL,  'label': 'SWELL\n(+8% vapor)'},
    {'xytext': (25.20, -5.90),'ha': 'right', 'va': 'bottom', 'rad': -0.16, 'cor': COR_FEN_SHRINK, 'label': 'SHRINK\n(-8% vapor)'},
]
for i, evento in enumerate(DEGRAUS_QS[1:]):
    idx = min(int(evento[0] / dt), N - 1)
    tx = t[idx] / 60
    y_evt = desvio_pct[idx]
    pos = posicoes_anotacoes_fig2[i]
    ax.annotate(
        pos['label'],
        xy=(tx, y_evt),
        xytext=pos['xytext'],
        fontsize=8,
        color=pos['cor'],
        ha=pos['ha'],
        va=pos['va'],
        bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.86),
        arrowprops=dict(
            arrowstyle='->',
            color=pos['cor'],
            lw=1.0,
            shrinkA=2,
            shrinkB=2,
            connectionstyle=f"arc3,rad={pos['rad']}"
        )
    )
ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
ax.grid(alpha=0.3, axis='y')

# Gráfico 2 - vazões de entrada e demanda de vapor
ax = axes2[1]
ax.plot(t / 60, qf_ma, color=COR_PID, lw=2,
        label=rf'Entrada de água $q_f(t)$ fixa = {fmt_num(qf0, 2)} kg/s')
ax.plot(t / 60, qs_ma, color=COR_VAPOR, lw=2, ls='--', label=r'Demanda de vapor $q_s(t)$')
ax.axhline(qs, color='gray', ls=':', lw=1, label=f'Valor nominal = {fmt_num(qs, 2)} kg/s')
for tx in eventos_min:
    ax.axvline(tx, color='gray', ls=':', lw=1.1)
ax.set_ylabel('Vazões\n[kg/s]')
ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
ax.grid(alpha=0.3, axis='y')

# Gráfico 3 - balanço mássico simplificado
ax = axes2[2]
ax.plot(t / 60, balanco_ma, color=COR_SET, lw=2, label=r'Balanço simplificado $q_f - q_s$')
ax.axhline(0, color='black', lw=1.2, ls='--', label=r'Equilíbrio nominal: $q_f = q_s$')
ax.fill_between(t / 60, 0, balanco_ma, where=balanco_ma >= 0,
                alpha=0.12, color='green', label='Entrada ≥ demanda')
ax.fill_between(t / 60, 0, balanco_ma, where=balanco_ma < 0,
                alpha=0.12, color='red', label='Demanda > entrada')
for tx in eventos_min:
    ax.axvline(tx, color='gray', ls=':', lw=1.1)
ax.set_ylabel('$q_f - q_s$\n[kg/s]')
ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
ax.grid(alpha=0.3, axis='y')

axes2[-1].set_xlabel('Tempo [min]')
for ax in axes2:
    ax.set_xlim(0, t[-1] / 60)
for ax in axes2:
    aplicar_formatacao_eixos(ax)

plt.tight_layout(rect=[0, 0, 0.83, 1])


# ── Figura 3: resposta do PID sob influência de shrink e swell ────────────────────────────────────
fig3, axes3 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

# Linhas verticais pontilhadas para destacar os instantes em que ocorrem
# os fenômenos shrink e swell. As mesmas linhas são traçadas nos três gráficos
# para evidenciar o alinhamento temporal entre nível, sinal de controle e demanda.
eventos_fenomenos_min = [item[0] / 60 for item in DEGRAUS_QS[1:]]
for ax_aux in axes3:
    for tx in eventos_fenomenos_min:
        ax_aux.axvline(tx, color='gray', ls=':', lw=1.0, alpha=0.75, zorder=0)

ax = axes3[0]
ax.plot(t / 60, 100 + y_pid  * 100 / y0, color=COR_PID,   lw=2, label='Nível real')
ax.plot(t / 60, 100 + ym_pid * 100 / y0, color=COR_MED,   lw=1.5, ls='--', label='Nível medido (com distorção)')
ax.axhline(100, color=COR_SET, ls=':', lw=1.5, label='Setpoint')
ax.axhspan(98, 102, alpha=0.07, color=COR_FAIXA, label='Faixa ±2%')
ax.set_ylabel('Nível [% do setpoint]')
ax.set_ylim(94.6, 103.1)
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
ax.set_title('Nível real e nível medido sob influência dos fenômenos shrink e swell')

# Anotações reposicionadas em regiões mais livres do gráfico e com cores
# distintas das curvas, para destacar os fenômenos sem confundir a leitura.
posicoes_anotacoes = [
    {'xytext': (4.15, 99.55), 'ha': 'right', 'va': 'top',    'rad': -0.20, 'cor': COR_FEN_SWELL,  'label': 'SWELL\n(+12% vapor)'},
    {'xytext': (11.15, 99.35), 'ha': 'left',  'va': 'top',    'rad':  0.18, 'cor': COR_FEN_SHRINK, 'label': 'SHRINK\n(-12% vapor)'},
    {'xytext': (18.25, 100.45),'ha': 'right', 'va': 'bottom', 'rad': -0.16, 'cor': COR_FEN_SWELL,  'label': 'SWELL\n(+8% vapor)'},
    {'xytext': (25.20, 97.45), 'ha': 'right', 'va': 'top',    'rad': -0.16, 'cor': COR_FEN_SHRINK, 'label': 'SHRINK\n(-8% vapor)'},
]

for i in range(1, len(DEGRAUS_QS)):
    idx = min(int(DEGRAUS_QS[i][0] / dt), N - 1)
    tx = t[idx] / 60
    frac_anterior = DEGRAUS_QS[i-1][1]
    frac_atual = DEGRAUS_QS[i][1]
    delta_frac = frac_atual - frac_anterior
    y_evento = 100 + ym_pid[idx] * 100 / y0

    pos = posicoes_anotacoes[i - 1]
    ax.annotate(
        pos['label'],
        xy=(tx, y_evento),
        xytext=pos['xytext'],
        fontsize=8,
        color=pos['cor'],
        ha=pos['ha'],
        va=pos['va'],
        bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.86),
        arrowprops=dict(
            arrowstyle='->',
            color=pos['cor'],
            lw=1.0,
            shrinkA=2,
            shrinkB=2,
            connectionstyle=f"arc3,rad={pos['rad']}"
        )
    )

ax = axes3[1]
ax.plot(t / 60, qf0 + u_pid, color=COR_PID, lw=2, label=r'$q_f$ efetiva - PID')
ax.axhline(qf0, color='gray', ls='--', lw=1, label=rf'$q_{{f0}}$ = {fmt_num(qf0, 2)} kg/s')
ax.set_ylabel(r'Vazão de entrada $q_f$ [kg/s]')
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
ax.set_title('Sinal efetivo de controle PID: comando após dinâmica da válvula')

ax = axes3[2]
ax.plot(t / 60, qs_vec, color=COR_VAPOR, lw=2, label=r'Demanda de vapor $q_s(t)$')
ax.axhline(qs, color='gray', ls='--', lw=1)
ax.set_ylabel('Demanda de vapor [kg/s]')
ax.set_xlabel('Tempo [min]')
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
ax.set_title('Variação da demanda de vapor associada aos fenômenos shrink e swell')
for ax in axes3:
    ax.set_xlim(0, t[-1] / 60)

for ax in axes3:
    aplicar_formatacao_eixos(ax)

plt.tight_layout()

# ── Figura 4: Comparação entre PID e fuzzy — nível e sinal de controle ─────
fig4 = plt.figure(figsize=(14, 8.5))

gs4 = gridspec.GridSpec(2, 1, figure=fig4, hspace=0.40)

# Linhas verticais pontilhadas para evidenciar os instantes dos fenômenos
# nos dois gráficos da comparação final.
eventos_fenomenos_min_fig4 = [item[0] / 60 for item in DEGRAUS_QS[1:]]

# Gráfico 1 - Nível
ax_niv = fig4.add_subplot(gs4[0, 0])
ax_niv.plot(t / 60, 100 + y_pid * 100 / y0, color=COR_PID,   lw=2,         label='Nível - PID')
ax_niv.plot(t / 60, 100 + y_fuz * 100 / y0, color=COR_FUZZY, lw=2, ls='--', label='Nível - Fuzzy')
ax_niv.axhline(100, color=COR_SET, ls=':', lw=1.5, label='Setpoint')
ax_niv.axhspan(98, 102, alpha=0.07, color=COR_FAIXA, label='Faixa ±2%')
for tx in eventos_fenomenos_min_fig4:
    ax_niv.axvline(tx, color='gray', ls=':', lw=1.0, alpha=0.55, zorder=0)
ax_niv.set_ylabel('Nível [% do setpoint]', fontsize=10)
ax_niv.set_xlabel('Tempo [min]', fontsize=10)
ax_niv.set_ylim(94.8, 103.05)
ax_niv.legend(loc='upper right', fontsize=9)
ax_niv.grid(alpha=0.3, axis='y')
ax_niv.set_title('Comparação do nível da água: PID e fuzzy sob influência de shrink e swell', fontsize=10)

# Pequenas anotações em regiões mais limpas do gráfico, com cores distintas
# das curvas, para indicar os fenômenos sem poluir a leitura.
posicoes_anotacoes_fig4 = [
    {'xytext': (4.25, 99.70), 'ha': 'right', 'va': 'top',    'rad': -0.18, 'cor': COR_FEN_SWELL,  'label': 'SWELL'},
    {'xytext': (12.35, 97.75),'ha': 'left',  'va': 'top',    'rad':  0.16, 'cor': COR_FEN_SHRINK, 'label': 'SHRINK'},
    {'xytext': (18.45, 100.35),'ha': 'right','va': 'bottom', 'rad': -0.16, 'cor': COR_FEN_SWELL,  'label': 'SWELL'},
    {'xytext': (25.20, 97.45),'ha': 'right', 'va': 'top',    'rad': -0.14, 'cor': COR_FEN_SHRINK, 'label': 'SHRINK'},
]

for i, evento in enumerate(DEGRAUS_QS[1:]):
    idx = min(int(evento[0] / dt), N - 1)
    tx = t[idx] / 60
    y_pid_evt = 100 + y_pid[idx] * 100 / y0
    y_fuz_evt = 100 + y_fuz[idx] * 100 / y0
    frac_anterior = DEGRAUS_QS[i][1]
    frac_atual = evento[1]
    delta_frac = frac_atual - frac_anterior

    if delta_frac > 0:
        y_evento = max(y_pid_evt, y_fuz_evt)
    else:
        y_evento = min(y_pid_evt, y_fuz_evt)

    pos = posicoes_anotacoes_fig4[i]
    ax_niv.annotate(
        pos['label'],
        xy=(tx, y_evento),
        xytext=pos['xytext'],
        fontsize=8,
        color=pos['cor'],
        ha=pos['ha'],
        va=pos['va'],
        bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.86),
        arrowprops=dict(
            arrowstyle='->',
            color=pos['cor'],
            lw=0.95,
            shrinkA=2,
            shrinkB=2,
            connectionstyle=f"arc3,rad={pos['rad']}"
        )
    )

# Gráfico 2 - Sinais de controle
ax_u = fig4.add_subplot(gs4[1, 0])
ax_u.plot(t / 60, qf0 + u_pid, color=COR_PID,   lw=2,         label=r'$q_f$ efetiva - PID')
ax_u.plot(t / 60, qf0 + u_fuz, color=COR_FUZZY, lw=2, ls='--', label=r'$q_f$ efetiva - Fuzzy')
ax_u.axhline(qf0, color='gray', ls='--', lw=1, label=rf'$q_{{f0}}$ = {fmt_num(qf0, 2)} kg/s (equilíbrio)')
for tx in eventos_fenomenos_min_fig4:
    ax_u.axvline(tx, color='gray', ls=':', lw=1.0, alpha=0.55, zorder=0)
ax_u.set_ylabel(r'Vazão de entrada $q_f$ [kg/s]', fontsize=10)
ax_u.set_xlabel('Tempo [min]', fontsize=10)
ax_u.legend(loc='upper right', fontsize=9)
ax_u.grid(alpha=0.3, axis='y')
u_all  = qf0 + np.concatenate([u_pid, u_fuz])
margem = (u_all.max() - u_all.min()) * 0.20 + 0.5
ax_u.set_ylim(u_all.min() - margem, u_all.max() + margem)
ax_u.set_title('Sinais efetivos de controle: resposta dinâmica da válvula', fontsize=10)
ax_niv.set_xlim(0, t[-1] / 60)
ax_u.set_xlim(0, t[-1] / 60)

aplicar_formatacao_eixos(ax_niv)
aplicar_formatacao_eixos(ax_u)

plt.tight_layout(rect=[0, 0, 1, 1])

# ── Figura 5: Comparação entre PID e fuzzy — métricas e tabela ─────────────
fig5 = plt.figure(figsize=(13.5, 5.8))

gs5 = gridspec.GridSpec(1, 2, figure=fig5, wspace=0.35)

# Gráfico de barras de métricas
ax_bar = fig5.add_subplot(gs5[0, 0])
metricas_nomes = ['IAE\n[m·s]', 'ISE×10³\n[m²·s]', 'ITAE×10⁻²\n[m·s²]']
vals_pid  = [IAE_p, ISE_p * 1e3, ITAE_p * 1e-2]
vals_fuz  = [IAE_f, ISE_f * 1e3, ITAE_f * 1e-2]
x_pos = np.arange(len(metricas_nomes))
w = 0.35
ax_bar.bar(x_pos - w/2, vals_pid, w, color=COR_PID,   label='PID',   alpha=0.85)
ax_bar.bar(x_pos + w/2, vals_fuz, w, color=COR_FUZZY, label='Fuzzy', alpha=0.85)
ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(metricas_nomes, fontsize=9)
ax_bar.set_title('Métricas de erro (menor = melhor)', fontsize=10)
ax_bar.legend(fontsize=9)
ax_bar.grid(axis='y', alpha=0.3)

# Tabela resumo
ax_tab = fig5.add_subplot(gs5[0, 1])
ax_tab.axis('off')

def fmt_melhora(v_pid, v_fuz):
    if abs(v_pid) < 1e-12:
        return '—'
    m = (v_pid - v_fuz) / v_pid * 100
    return f'{m:+.1f}%'.replace('.', ',')

dados = [
    ['Métrica',          'PID',                  'Fuzzy',               'Melhora Fuzzy'],
    ['IAE [m·s]',        fmt_num(IAE_p, 4),      fmt_num(IAE_f, 4),      fmt_melhora(IAE_p, IAE_f)],
    ['ISE [m²·s]',       fmt_num(ISE_p, 5),      fmt_num(ISE_f, 5),      fmt_melhora(ISE_p, ISE_f)],
    ['ITAE [m·s²]',      fmt_num(ITAE_p, 1),     fmt_num(ITAE_f, 1),     fmt_melhora(ITAE_p, ITAE_f)],
    ['Máx. desvio [%]',  fmt_num(md_p, 2) + '%', fmt_num(md_f, 2) + '%', fmt_melhora(md_p, md_f)],
    ['Últ. fora ±2% [s]', fmt_num(ts_p, 0),      fmt_num(ts_f, 0),      fmt_melhora(ts_p, ts_f)],
]

tabela = ax_tab.table(cellText=dados[1:], colLabels=dados[0],
                      cellLoc='center', loc='center',
                      colWidths=[0.25, 0.22, 0.22, 0.26])
tabela.auto_set_font_size(False)
tabela.set_fontsize(8.5)
tabela.scale(1, 1.8)
for j in range(4):
    tabela[0, j].set_facecolor('#2c3e50')
    tabela[0, j].set_text_props(color='white', fontweight='bold')
ax_tab.set_title('Tabela resumo de desempenho', fontsize=10, pad=10)

aplicar_formatacao_eixos(ax_bar, formatar_x=False, formatar_y=True)

plt.tight_layout(rect=[0, 0, 1, 1])


# ── Figuras adicionais: funções de pertinência do controlador Fuzzy ──────────
# Estas figuras são geradas a partir das mesmas funções utilizadas na simulação.
# Portanto, a representação gráfica corresponde aos parâmetros adotados no
# controlador fuzzy.

def plotar_funcoes_pertinencia(ax, universo, mfs, xlabel):
    for nome, valores in mfs.items():
        ax.plot(universo, valores, linewidth=2, label=nome)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Grau de pertinência')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    aplicar_formatacao_eixos(ax)


def gerar_figuras_funcoes_pertinencia():
    # 1) Erro de nível
    universo_e = np.linspace(-E_MAX, E_MAX, 1000)
    mfs_e = {
        'NG': trapezoidal_vetor(universo_e, -E_MAX, -E_MAX, -0.095, -0.025),
        'NP': triangular_vetor(universo_e,  -0.070, -0.025,  0.000),
        'ZE': triangular_vetor(universo_e,  -0.018,  0.000,  0.018),
        'PP': triangular_vetor(universo_e,   0.000,  0.025,  0.070),
        'PG': trapezoidal_vetor(universo_e,  0.025,  0.095, E_MAX, E_MAX),
    }
    fig_mf_e, ax_mf_e = plt.subplots(figsize=(10, 4.8))
    plotar_funcoes_pertinencia(ax_mf_e, universo_e, mfs_e, r'Erro de nível $e$ [m]')
    plt.tight_layout()

    # 2) Derivada do erro
    universo_de = np.linspace(-DE_MAX, DE_MAX, 1000)
    mfs_de = {
        'NE': trapezoidal_vetor(universo_de, -DE_MAX, -DE_MAX, -7.0e-04, -1.5e-04),
        'ZE': triangular_vetor(universo_de,  -1.4e-03, 0.0, 1.4e-03),
        'PO': trapezoidal_vetor(universo_de,  1.5e-04, 7.0e-04, DE_MAX, DE_MAX),
    }
    fig_mf_de, ax_mf_de = plt.subplots(figsize=(10, 4.8))
    plotar_funcoes_pertinencia(ax_mf_de, universo_de, mfs_de, r'Derivada do erro $de/dt$ [m/s]')
    plt.tight_layout()

    # 3) Saída do controlador fuzzy: utiliza a função empregada na simulação
    universo_dqf = np.linspace(DQ_MIN, DQ_MAX, 1000)
    mfs_dqf = pertinencia_saida_delta_qf_vetor(universo_dqf)
    fig_mf_dqf, ax_mf_dqf = plt.subplots(figsize=(10, 4.8))
    plotar_funcoes_pertinencia(ax_mf_dqf, universo_dqf, mfs_dqf,
                              r'Variação da vazão de alimentação $\Delta q_f$ [kg/s]')
    plt.tight_layout()

    # 4) Figura consolidada
    fig_mf_all, axes_mf = plt.subplots(3, 1, figsize=(11, 12))
    plotar_funcoes_pertinencia(axes_mf[0], universo_e, mfs_e, r'Erro de nível $e$ [m]')
    plotar_funcoes_pertinencia(axes_mf[1], universo_de, mfs_de, r'Derivada do erro $de/dt$ [m/s]')
    plotar_funcoes_pertinencia(axes_mf[2], universo_dqf, mfs_dqf,
                              r'Variação da vazão de alimentação $\Delta q_f$ [kg/s]')
    plt.tight_layout()

    return fig_mf_e, fig_mf_de, fig_mf_dqf, fig_mf_all


fig_mf_e, fig_mf_de, fig_mf_dqf, fig_mf_all = gerar_figuras_funcoes_pertinencia()

# ── Salvamento automático das figuras ────────────────────────────────────────
salvar_figura(fig1, 'etapa1_propriedades_termodinamicas')
salvar_figura(fig2, 'etapa2_malha_aberta')
salvar_figura(fig3, 'etapa3_pid_shrink_swell')
salvar_figura(fig4, 'etapa4_nivel_e_sinal_controle')
salvar_figura(fig5, 'etapa5_metricas_e_tabela')
salvar_figura(fig_mf_e, 'funcao_pertinencia_erro')
salvar_figura(fig_mf_de, 'funcao_pertinencia_derivada_erro')
salvar_figura(fig_mf_dqf, 'funcao_pertinencia_saida')
salvar_figura(fig_mf_all, 'funcoes_pertinencia_consolidado')

plt.show()
print("\nPrograma finalizado com sucesso.")
