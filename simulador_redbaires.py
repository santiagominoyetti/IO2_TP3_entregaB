"""Simulador de eventos discretos de RedBaires (TP3, parte B).

Implementa el modelo conceptual de la parte A:
  E1 Aparición de una pieza, E2 Reenvío, E3 Ingreso a la fila, E4 Fin de revisión,
  con las lógicas L1 (Centinela), L2 (fricción), L3 (degradación) y L4 (resultado de la revisión).

Cada pieza es una sola entidad con memoria fija: guarda su potencial pendiente por generación
y por fase del retardo y tiene a lo sumo un evento E2 programado. Las variables de estado
(QM, BM, UM, PA, R_i) se actualizan únicamente en los eventos que las modifican.

Unidad de tiempo interna: minutos. La réplica empieza un lunes a las 0 h.
"""
import heapq
import math
import random
import time
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, asdict, field
from itertools import accumulate

import numpy as np
import pandas as pd
import scipy.stats as st

NIVELES = ['nano', 'micro', 'macro', 'mega']
MEGA = 3
E1, E2, E3, E4 = 1, 2, 3, 4


# =====================================================================================
# Parámetros de una corrida
# =====================================================================================
@dataclass
class Configuracion:
    """Variables de decisión (Tabla 2 de la parte A)."""
    M: int = 1                  # moderadores por turno (1 a 12)
    C: int = 0                  # configuración de Centinela: 0 medida, 1 declarada por Finanzas, 2 máximo del rango
    U: int = 5                  # umbral de fricción: reenvíos acumulados de la pieza (1 a 20); sin efecto si F = 0
    F: float = 0.0              # intensidad de la fricción (0 a 0,9)
    D: str = 'ninguna'          # alcance de la degradación: 'ninguna', 'mega', 'c2', 'mega_c2'
    delta: float = 0.0          # intensidad de la degradación (0 a 0,9)

    def __post_init__(self):
        assert 1 <= self.M <= 12 and self.C in (0, 1, 2) and 1 <= self.U <= 20, 'M, C o U fuera de rango'
        assert 0 <= self.F <= 0.9 and 0 <= self.delta <= 0.9, 'F o delta fuera de rango'
        assert self.D in ('ninguna', 'mega', 'c2', 'mega_c2'), 'D fuera de rango'


@dataclass
class Escenario:
    """Escenarios de la parte A (2.5): factor sobre todas las medias de hijos y episodio del 14 de junio."""
    nombre: str = 'conservador'
    m_objetivo: float = None     # None: la media de la muestra
    episodio: bool = False


ESCENARIOS = {'conservador': Escenario('conservador', None, False),
              'intermedio': Escenario('intermedio', 0.90, False),
              'pesimista': Escenario('pesimista', 0.97, True)}


@dataclass
class Supuestos:
    """Cantidades sin datos que la parte A fija como supuestos (Tabla 8) y parámetros de la réplica.
    None significa: tomar el valor estimado en los insumos."""
    prem_nociva: float = 0.80
    prem_legitima: float = 0.05
    indulgencia: float = 0.5           # multiplica PREM si la cuenta semilla está verificada
    trev_media_min: float = None       # None: costo por caso / costo por hora (19,6 s)
    derivacion_min: float = None       # None: piso de la latencia observada (90 min)
    escala: float = None               # None: ESC calibrada a priori (320)
    ndeg: int = None                   # None: cuentas mega distintas de la muestra (231)
    horizonte_h: float = 720.0
    calentamiento_h: float = 24.0
    generaciones: int = 40
    llegadas_uniformes: bool = False   # tasa de siembra constante (verificación de la fila contra M/M/c)
    retardo: str = 'mezcla'            # 'h2': hiperexponencial de la parte A; 'h2_espera': fase diferida con espera previa;
                                       # 'mezcla': mezcla de exponenciales con espera ajustada a los retardos observados


# =====================================================================================
# Entidades
# =====================================================================================
class Pieza:
    __slots__ = ('id', 'rng', 't0', 'nociva', 'c2', 'nivel0', 'ver0', 'vm', 'lin', 'deg_c2', 'R',
                 'raw', 'tref', 'tot', 'pend', 'ptot', 'estado', 'tiene_caso', 't_marca', 'version',
                 'prox_fase', 'eventos', 'prof', 't_ult', 'imp', 'u_rem', 'e_rev')

    def __init__(self, pid, rng, t0, nf):
        self.id, self.rng, self.t0 = pid, rng, t0
        self.R = 0                       # reenvíos acumulados (VE R_i)
        self.raw = tuple({} for _ in range(nf))      # potencial activo por fase y generación (relativo a tref)
        self.tref = [t0] * nf
        self.tot = [0.0] * nf                        # suma de raw por fase
        self.pend = tuple(deque() for _ in range(nf))  # potencial en espera: [instante de activación, {gen: masa}, masa total]
        self.ptot = [0.0] * nf                       # masa total en espera por fase
        self.estado = 'activa'
        self.tiene_caso = False
        self.t_marca = None
        self.version = 0
        self.prox_fase = None
        self.eventos = 1                 # semilla incluida
        self.prof = 0
        self.t_ult = t0
        self.imp = 0.0


class Caso:
    __slots__ = ('pieza', 't_marca', 't_fila')

    def __init__(self, pieza, t_marca):
        self.pieza, self.t_marca, self.t_fila = pieza, t_marca, None


# =====================================================================================
# Informe de una réplica
# =====================================================================================
@dataclass
class Informe:
    configuracion: dict
    escenario: str
    semilla: int
    # variables de referencia (totales de la ventana de medición)
    IN: float
    IL: float
    CFP: float
    # variables de estado (promedios temporales en la ventana de medición)
    QM_medio: float
    QM_max: int
    BM_medio: float
    UM_medio_pct: float
    PA_medio: float
    PA_max: int
    # indicadores complementarios
    piezas_sembradas: int
    reenvios: int
    reenvios_bloqueados: int
    casos_creados: int
    casos_nocivos: int
    casos_revisados: int
    remociones_nocivas: int
    remociones_legitimas: int
    espera_media_fila_min: float
    demora_media_marcado_remocion_min: float
    latencia_media_min: float            # desde el primer marcado hasta el fin de la revisión
    casos_por_min: float
    tamano_medio_cascada: float          # piezas sembradas y terminadas dentro de la ventana de medición
    duracion_media_cascada_h: float
    # costos (USD por mes; lo medido en la ventana se lleva a un mes de horas_operativas_mes)
    CF_moderadores: float
    CF_clasificador: float
    CF_degradacion: float
    CF: float
    CFP_mes: float
    CD: float                            # CF + CFP_mes
    valor_impresiones_legitimas: float
    costo_social_impresiones_nocivas: float
    # cómputo
    eventos_procesados: int
    segundos: float

    def resumen(self):
        filas = [('IN (impresiones nocivas)', self.IN), ('IL (impresiones legítimas)', self.IL), ('CFP (USD)', self.CFP),
                 ('UM medio (%)', self.UM_medio_pct), ('QM medio', self.QM_medio), ('PA medio', self.PA_medio),
                 ('Reenvíos', self.reenvios), ('Casos revisados', self.casos_revisados),
                 ('Remociones nocivas', self.remociones_nocivas), ('Remociones legítimas', self.remociones_legitimas),
                 ('CF (USD/mes)', self.CF), ('CD = CF + CFP mensual (USD/mes)', self.CD)]
        return pd.DataFrame(filas, columns=['indicador', 'valor']).set_index('indicador')


# =====================================================================================
# Simulador
# =====================================================================================
class SimuladorRedBaires:
    def __init__(self, insumos):
        self.I = insumos
        c = insumos.coef
        self.b_niv = list(c['nivel'])
        self.b_sup = list(c['superficie'])
        self.b_form = [c['formato'][f] for f in insumos.formatos]
        self.cum_hijo = list(accumulate(insumos.p_nivel_hijo))
        self.cum_sup = list(accumulate(insumos.p_superficie))
        self.cum_form = list(accumulate(insumos.p_formato))
        self.cum_sem = {k: list(accumulate(v)) for k, v in insumos.nivel_semilla.items()}
        self.p_ver = list(insumos.p_verificada)
        self.seg = [list(map(float, insumos.seguidores[n])) for n in NIVELES]
        self.r = insumos.r_nb

    # ------------------------------------------------------------------ utilidades
    @staticmethod
    def _elegir(rng, cum):
        return min(bisect_right(cum, rng.random() * cum[-1]), len(cum) - 1)

    def parametros_fila(self, sup):
        """(TREV, DER) en minutos con los supuestos dados. No depende de corridas anteriores."""
        fz = self.I.finanzas
        trev = sup.trev_media_min if sup.trev_media_min else fz['costo_revision_por_item_flag']['base'] / fz['costo_moderador_por_hora']['base'] * 60
        der = sup.derivacion_min if sup.derivacion_min is not None else self.I.piso_latencia
        return trev, der

    def _preparar(self, cfg, esc, sup):
        """Fija los parámetros derivados de una corrida."""
        I = self.I
        fz = I.finanzas
        self.cfg, self.esc, self.sup = cfg, esc, sup
        self.GM = sup.generaciones
        self.ln_factor = math.log(esc.m_objetivo / I.m_reenvio) if esc.m_objetivo else 0.0
        # L1: probabilidades de marcado según C
        pares = {0: (I.precision_ev, I.recall_ev),
                 1: (fz['precision_clasificador']['base'], fz['recall_clasificador']['base']),
                 2: (fz['precision_clasificador']['hi'], fz['recall_clasificador']['hi'])}
        prec, rec = pares[cfg.C]
        pi = I.pi_ev
        self.p_marca = (rec * pi * (1 - prec) / (prec * (1 - pi)), rec)        # (legítima, nociva)
        # retardo padre-hijo: lista de fases (peso, tasa de decaimiento, espera previa)
        if sup.retardo == 'h2':                       # parte A: hiperexponencial de dos fases
            fases = [(I.w_rapida, I.media_rapida, 0.0), (1 - I.w_rapida, I.media_diferida, 0.0)]
        elif sup.retardo == 'h2_espera':              # la fase diferida empieza después del corte
            fases = [(I.w_rapida, I.media_rapida, 0.0), (1 - I.w_rapida, I.media_exceso_diferida, I.corte_retardo)]
        elif sup.retardo == 'mezcla':                 # mezcla de exponenciales con espera, ajustada por EM
            fases = I.fases_retardo
        else:
            raise ValueError(sup.retardo)
        self.fases = [(w, 1 / m, e) for w, m, e in fases]
        self.con_espera = [f for f, x in enumerate(self.fases) if x[2] > 0]
        # degradación
        self.deg_mega = cfg.D in ('mega', 'mega_c2')
        self.deg_c2 = cfg.D in ('c2', 'mega_c2')
        self.fdeg = 1.0 - cfg.delta
        self.ln_fdeg = math.log(self.fdeg)
        # moderación
        self.trev, self.der = self.parametros_fila(sup)
        self.escala = sup.escala if sup.escala else I.escala
        self.ndeg = sup.ndeg if sup.ndeg is not None else I.ndeg

    # ------------------------------------------------------------------ propagación de una pieza
    def _crear_pieza(self, pid, t, rng, forzar=None):
        I, c = self.I, self.I.coef
        p = Pieza(pid, rng, t, len(self.fases))
        f = forzar or {}
        p.nociva = f.get('nociva', int(rng.random() < I.p_nociva))
        p.c2 = f.get('c2', int(rng.random() < I.p_c2))
        p.nivel0 = f.get('nivel', self._elegir(rng, self.cum_sem[p.c2]))
        p.ver0 = int(rng.random() < self.p_ver[p.nivel0])
        form = self._elegir(rng, self.cum_form)
        p.vm = I.vm_mediana * math.exp(I.vm_sigma * rng.gauss(0.0, 1.0))
        p.lin = c['intercepto'] + self.b_form[form] + c['c2'] * p.c2 + c['nociva'] * p.nociva + self.ln_factor
        p.deg_c2 = self.deg_c2 and p.c2 == 1
        p.u_rem = rng.random()                   # uniforme de la decisión de remoción (L4)
        p.e_rev = rng.expovariate(1.0)           # duración de la revisión, en unidades de TREV
        return p

    def _impresiones(self, rng, nivel, degradada):
        s = self.seg[nivel]
        imp = rng.weibullvariate(self.I.frac_escala, self.I.frac_c) * s[int(rng.random() * len(s))]
        return imp * self.fdeg if degradada else imp

    def _agregar_potencial(self, p, t, gen, mu):
        """Suma el potencial de un reenvío: J ~ gamma(r, mu / r), repartido entre las fases del retardo
        según sus pesos. Una fase con espera recibe su parte recién después de la espera."""
        J = p.rng.gammavariate(self.r, mu / self.r)
        if J <= 0:
            return
        g = min(gen, self.GM - 1)
        for f, (w, k, espera) in enumerate(self.fases):
            if espera > 0:
                ta = t + espera
                pend = p.pend[f]
                x = w * J
                if pend and pend[-1][0] >= ta - 1.0:        # cubetas de 1 minuto: memoria acotada a espera + 1
                    u = pend[-1]; d = u[1]; d[g] = d.get(g, 0.0) + x; u[2] += x
                else:
                    pend.append([ta, {g: x}, x])
                p.ptot[f] += x
            else:
                self._activar(p, f, t, {g: w * J})

    def _activar(self, p, f, t, masas):
        k = self.fases[f][1]
        if k * (t - p.tref[f]) > 30:
            self._rebase(p, f, t)
        e = math.exp(k * (t - p.tref[f]))
        rf = p.raw[f]
        for g, m in masas.items():
            x = m * e; rf[g] = rf.get(g, 0.0) + x; p.tot[f] += x

    def _rebase(self, p, f, t):
        e = math.exp(-self.fases[f][1] * (t - p.tref[f]))
        rf = p.raw[f]
        for g in rf:
            rf[g] *= e
        p.tot[f] *= e
        p.tref[f] = t

    def _masa(self, p, f, t):
        return p.tot[f] * math.exp(-self.fases[f][1] * (t - p.tref[f]))

    def _activar_vencidas(self, p, t):
        for f in self.con_espera:
            pend = p.pend[f]
            while pend and pend[0][0] <= t:
                ta, masas, x = pend.popleft()
                p.ptot[f] -= x
                self._activar(p, f, ta, masas)

    def _proximo(self, p, t):
        """Instante del próximo reenvío de la pieza (None si no hay: fin natural).
        En cada fase la intensidad decae exponencialmente; se muestrea por inversión el primer arribo
        de cada fase y se toma el mínimo. En una fase con espera, la integral de la intensidad se
        acumula por tramos entre las activaciones pendientes."""
        rng = p.rng
        mejor, fase = math.inf, None
        for f, (w, k, espera) in enumerate(self.fases):
            A = self._masa(p, f, t)
            E = rng.expovariate(1.0)
            s = math.inf
            if espera == 0:
                if E < A:
                    s = t - math.log1p(-E / A) / k
            elif E < A + p.ptot[f]:                          # si no, la fase no produce más reenvíos
                tau = t
                for ta, _, x in p.pend[f]:
                    lam = A * (1 - math.exp(-k * (ta - tau)))
                    if E < lam:
                        s = tau - math.log1p(-E / A) / k
                        break
                    E -= lam
                    A = A * math.exp(-k * (ta - tau)) + x
                    tau = ta
                else:
                    if E < A:
                        s = tau - math.log1p(-E / A) / k
            if s < mejor:
                mejor, fase = s, f
        if fase is None:
            return None
        p.prox_fase = fase
        return mejor

    def _elegir_generacion(self, p, f):
        rf = p.raw[f]
        u = p.rng.random() * sum(rf.values())
        acum = 0.0
        for g, x in rf.items():
            acum += x
            if acum >= u:
                return g
        return max(rf)

    def _semilla(self, p, t):
        """Lógica de la semilla dentro de E1. Devuelve (impresiones, marcada)."""
        rng, c = p.rng, self.I.coef
        sup = self._elegir(rng, self.cum_sup)
        degr = (self.deg_mega and p.nivel0 == MEGA) or p.deg_c2
        imp = self._impresiones(rng, p.nivel0, degr)
        marcada = rng.random() < self.p_marca[p.nociva]                           # L1
        mu = math.exp(p.lin + self.b_niv[p.nivel0] + c['semilla'] + self.b_sup[sup] + c['verificada'] * p.ver0
                      + (self.ln_fdeg if degr else 0.0))                          # L3 sobre el potencial
        self._agregar_potencial(p, t, 0, mu)
        p.imp += imp
        return imp, marcada

    def _reenvio(self, p, t):
        """Lógica de E2 sobre la pieza. Devuelve None si la fricción bloquea el reenvío, o (impresiones, marcada)."""
        rng, c, cfg = p.rng, self.I.coef, self.cfg
        if self.con_espera:
            self._activar_vencidas(p, t)
        gp = self._elegir_generacion(p, p.prox_fase)
        u = rng.random()                         # se sortea siempre, para conservar los números aleatorios comunes
        if cfg.F > 0 and p.R >= cfg.U and u < cfg.F:                              # L2
            return None
        d = gp + 1
        p.R += 1
        p.eventos += 1
        p.prof = max(p.prof, d)
        p.t_ult = t
        nivel = self._elegir(rng, self.cum_hijo)
        sup = self._elegir(rng, self.cum_sup)
        ver = int(rng.random() < self.p_ver[nivel])
        degr = (self.deg_mega and nivel == MEGA) or p.deg_c2                      # L3
        imp = self._impresiones(rng, nivel, degr)
        marcada = rng.random() < self.p_marca[p.nociva]                           # L1
        mu = math.exp(p.lin + self.b_niv[nivel] + c['gen'] * max(d - 1, 0) + c['edad_rel'] * (t - p.t0) / p.vm
                      + self.b_sup[sup] + c['verificada'] * ver + (self.ln_fdeg if degr else 0.0))
        self._agregar_potencial(p, t, d, mu)
        p.imp += imp
        return imp, marcada

    # ------------------------------------------------------------------ cascadas aisladas (validación)
    def simular_cascadas(self, n, semilla=1, tope=None, cfg=None, esc=None, sup=None):
        """Propaga n piezas independientes, sin moderación, con las mismas funciones que usa E1 y E2.
        Devuelve un DataFrame con tamaño (eventos, semilla incluida), profundidad, duración (h), marcada e impresiones."""
        self._preparar(cfg or Configuracion(), esc or ESCENARIOS['conservador'], sup or Supuestos())
        filas = []
        for i in range(n):
            rng = random.Random(f'{semilla}:c{i}')
            p = self._crear_pieza(i, 0.0, rng)
            _, marc = self._semilla(p, 0.0)
            t = self._proximo(p, 0.0)
            while t is not None and (tope is None or p.eventos < tope):
                res = self._reenvio(p, t)
                if res is not None:
                    marc = marc or res[1]
                t = self._proximo(p, t)
            filas.append((p.eventos, p.prof, p.t_ult / 60.0, int(marc), p.nociva, p.imp))
        return pd.DataFrame(filas, columns=['tamano', 'profundidad', 'duracion_h', 'marcada', 'nociva', 'impresiones'])

    # ------------------------------------------------------------------ corrida completa (eventos E1 a E4)
    def correr(self, cfg=None, escenario='conservador', sup=None, semilla=1, series=False):
        """Una réplica terminante: calentamiento + horizonte de medición, fila vacía al inicio.
        Devuelve (Informe, series). Con series=True, series contiene las variables de estado registradas
        en cada evento que las modifica (QM, BM, UM, PA) y la trayectoria de R_i de cada pieza."""
        cfg = cfg or Configuracion()
        esc = ESCENARIOS[escenario] if isinstance(escenario, str) else escenario
        sup = sup or Supuestos()
        self._preparar(cfg, esc, sup)
        I, fz = self.I, self.I.finanzas
        reloj0 = time.perf_counter()
        t_ini = sup.calentamiento_h * 60.0
        t_fin = t_ini + sup.horizonte_h * 60.0
        rng_lleg = random.Random(f'{semilla}:llegadas')
        tasa = I.tasa_dia_hora * self.escala / 60.0                                # piezas por minuto
        if sup.llegadas_uniformes:                                                 # solo para verificar la fila
            tasa = np.full_like(tasa, tasa.mean())
        tasa = tasa.tolist()
        lam_max = max(max(f) for f in tasa)

        fel, seq = [], 0                                                            # lista de eventos futuros
        def programar(tt, tipo, obj, ver=0):
            nonlocal seq
            seq += 1
            heapq.heappush(fel, (tt, seq, tipo, obj, ver))

        # variables de estado y acumuladores
        QM = BM = PA = 0
        fila = deque()
        area = {'QM': 0.0, 'BM': 0.0, 'PA': 0.0}
        t_ult = 0.0
        QM_max = PA_max = 0
        serie, serie_R = ([], []) if series else (None, None)
        def acumular(tt):
            nonlocal t_ult
            a, b = max(t_ult, t_ini), min(tt, t_fin)
            if b > a:
                area['QM'] += QM * (b - a); area['BM'] += BM * (b - a); area['PA'] += PA * (b - a)
            t_ult = tt
        def anotar(tt):
            if series and t_ini <= tt <= t_fin:
                serie.append((tt, QM, BM, PA))

        IN = IL = CFP = 0.0
        cont = dict(piezas=0, reenvios=0, bloqueados=0, casos=0, casos_noc=0, revisados=0, rem_noc=0, rem_leg=0, eventos=0)
        esperas, demoras, latencias, cascadas = [], [], [], []
        prem = (sup.prem_legitima, sup.prem_nociva)
        pid = 0

        def p_rev(caso):
            return caso.pieza.e_rev * self.trev

        def proxima_llegada(tt):
            while True:
                tt += rng_lleg.expovariate(lam_max)
                dia, hora = int(tt // 1440) % 7, int(tt % 1440 // 60)
                if rng_lleg.random() * lam_max <= tasa[dia][hora]:
                    return tt

        def nuevo_caso(p, tt):
            if not p.tiene_caso:                                                 # a lo sumo un caso por pieza
                p.tiene_caso, p.t_marca = True, tt
                if t_ini <= tt <= t_fin:
                    cont['casos'] += 1
                    cont['casos_noc'] += p.nociva
                programar(tt + self.der, E3, Caso(p, tt))

        def cerrar_cascada(p, tt):
            if t_ini <= p.t0 and tt <= t_fin:                                    # sembrada y terminada en la ventana
                cascadas.append((p.eventos, (p.t_ult - p.t0) / 60.0))

        def programar_reenvio(p, tt):
            nonlocal PA
            s = self._proximo(p, tt)
            if s is None:                                                        # fin natural (condición)
                acumular(tt); PA -= 1; anotar(tt)
                p.estado = 'terminada'
                cerrar_cascada(p, tt)
            else:
                programar(s, E2, p, p.version)

        programar(proxima_llegada(0.0), E1, None)
        if esc.episodio:                                                         # 14 de junio: sábado al mediodía
            t_ep = 5 * 1440 + 12 * 60
            while t_ep < t_ini:
                t_ep += 7 * 1440
            programar(t_ep, E1, 'episodio')

        while fel:
            tt, _, tipo, obj, ver = heapq.heappop(fel)
            if tt > t_fin:
                break
            cont['eventos'] += 1
            medir = tt >= t_ini
            if tipo == E1:                                                       # ---- aparición
                pid += 1
                rng_p = random.Random(f'{semilla}:p{pid}')
                forzar = {'nociva': 1, 'c2': 1, 'nivel': MEGA} if obj == 'episodio' else None
                p = self._crear_pieza(pid, tt, rng_p, forzar)
                acumular(tt); PA += 1; anotar(tt)
                if medir: PA_max = max(PA_max, PA)
                if medir:
                    cont['piezas'] += 1
                imp, marc = self._semilla(p, tt)
                if medir:
                    if p.nociva: IN += imp
                    else: IL += imp
                if marc:
                    nuevo_caso(p, tt)
                programar_reenvio(p, tt)
                if obj is None:
                    programar(proxima_llegada(tt), E1, None)
            elif tipo == E2:                                                     # ---- reenvío
                p = obj
                if p.estado != 'activa' or ver != p.version:
                    continue
                res = self._reenvio(p, tt)
                if res is None:
                    if medir: cont['bloqueados'] += 1
                else:
                    imp, marc = res
                    if series and medir:
                        serie_R.append((tt, p.id, p.R))
                    if medir:
                        cont['reenvios'] += 1
                        if p.nociva: IN += imp
                        else: IL += imp
                    if marc:
                        nuevo_caso(p, tt)
                programar_reenvio(p, tt)
            elif tipo == E3:                                                     # ---- ingreso a la fila
                caso = obj
                caso.t_fila = tt
                acumular(tt)
                if BM < cfg.M:
                    BM += 1
                    if medir: esperas.append(0.0)
                    programar(tt + p_rev(caso), E4, caso)
                else:
                    QM += 1; fila.append(caso)
                    if medir: QM_max = max(QM_max, QM)
                anotar(tt)
            else:                                                                # ---- fin de revisión (L4)
                caso = obj
                p = caso.pieza
                if medir:
                    cont['revisados'] += 1
                    latencias.append(tt - caso.t_marca)
                pr = prem[p.nociva] * (sup.indulgencia if p.ver0 else 1.0)
                if p.u_rem < pr:
                    if medir:
                        if p.nociva: cont['rem_noc'] += 1
                        else:
                            cont['rem_leg'] += 1
                            CFP += fz['costo_reputacional_por_falso_positivo']['base']
                        demoras.append(tt - caso.t_marca)
                    if p.estado == 'activa':
                        acumular(tt); PA -= 1
                        p.estado = 'removida'; p.version += 1
                        cerrar_cascada(p, tt)
                acumular(tt)
                if fila:
                    sig = fila.popleft(); QM -= 1
                    if tt >= t_ini: esperas.append(tt - sig.t_fila)
                    programar(tt + p_rev(sig), E4, sig)
                else:
                    BM -= 1
                anotar(tt)
        acumular(t_fin)
        T = t_fin - t_ini
        # costos de configuración (mensuales)
        cf_mod = cfg.M * fz['horas_operativas_mes']['base'] * fz['costo_moderador_por_hora']['base']
        capex = {0: 0.0, 1: fz['capex_clasificador_ml']['base'], 2: fz['capex_clasificador_ml']['hi']}[cfg.C]
        cf_cla = (capex / (fz['vida_util_clasificador_years']['base'] * 12) + fz['opex_anual_clasificador']['base'] / 12) if cfg.C > 0 else 0.0
        dias_mes = fz['horas_operativas_mes']['base'] / 24                       # 30 días
        cf_deg = (fz['costo_degradar_influencer_por_dia']['base'] * self.ndeg * dias_mes) if (self.deg_mega and cfg.delta > 0) else 0.0
        CF = cf_mod + cf_cla + cf_deg
        a_mes = fz['horas_operativas_mes']['base'] / sup.horizonte_h               # 1 con el horizonte de 30 días
        return Informe(
            configuracion=asdict(cfg), escenario=esc.nombre, semilla=semilla,
            IN=IN, IL=IL, CFP=CFP,
            QM_medio=area['QM'] / T, QM_max=QM_max, BM_medio=area['BM'] / T, UM_medio_pct=100 * area['BM'] / T / cfg.M,
            PA_medio=area['PA'] / T, PA_max=PA_max,
            piezas_sembradas=cont['piezas'], reenvios=cont['reenvios'], reenvios_bloqueados=cont['bloqueados'],
            casos_creados=cont['casos'], casos_nocivos=cont['casos_noc'], casos_revisados=cont['revisados'],
            remociones_nocivas=cont['rem_noc'], remociones_legitimas=cont['rem_leg'],
            espera_media_fila_min=float(np.mean(esperas)) if esperas else float('nan'),
            demora_media_marcado_remocion_min=float(np.mean(demoras)) if demoras else float('nan'),
            latencia_media_min=float(np.mean(latencias)) if latencias else float('nan'),
            casos_por_min=cont['casos'] / (t_fin - t_ini),
            tamano_medio_cascada=float(np.mean([c[0] for c in cascadas])) if cascadas else float('nan'),
            duracion_media_cascada_h=float(np.mean([c[1] for c in cascadas])) if cascadas else float('nan'),
            CF_moderadores=cf_mod, CF_clasificador=cf_cla, CF_degradacion=cf_deg, CF=CF, CFP_mes=CFP * a_mes, CD=CF + CFP * a_mes,
            valor_impresiones_legitimas=IL * a_mes * fz['valor_engagement_por_impresion']['base'],
            costo_social_impresiones_nocivas=IN * a_mes * fz['costo_social_por_impresion_nociva']['base'],
            eventos_procesados=cont['eventos'], segundos=time.perf_counter() - reloj0), (
            {'estado': pd.DataFrame(serie, columns=['t_min', 'QM', 'BM', 'PA']).assign(UM=lambda v: 100 * v.BM / cfg.M),
             'R': pd.DataFrame(serie_R, columns=['t_min', 'pieza', 'R'])} if series else None)


# =====================================================================================
# Recalibración de la escala
# =====================================================================================
def calibrar_escala(sim, n=100_000, semilla=7, retardo='mezcla'):
    """ESC que hace que la tasa simulada de casos coincida con la tasa λ calibrada en la parte A.
    La parte A fija ESC = λ / (piezas marcadas por día en la muestra). El simulador marca una
    proporción de piezas distinta de la observada (supuesto S11), así que la misma tasa de casos
    requiere ESC = λ / (semillas por día × proporción simulada de piezas marcadas).
    La proporción se estima con n cascadas sin tope, sin moderación."""
    I = sim.I
    d = sim.simular_cascadas(n, semilla=semilla, tope=None, sup=Supuestos(retardo=retardo))
    p_sim = float(d.marcada.mean())
    esc = I.observado['lambda_casos_min'] * 1440 / (I.semillas_por_dia * p_sim)
    return esc, p_sim


# =====================================================================================
# Réplicas e intervalos de confianza
# =====================================================================================
_SIM = None                      # copia del simulador en cada proceso de trabajo


def _iniciar(sim):
    global _SIM
    _SIM = sim


def _llamar(args):
    funcion, tarea = args
    return funcion(_SIM, tarea)


def en_paralelo(sim, funcion, tareas, procesos=1):
    """Devuelve [funcion(sim, tarea) for tarea in tareas], repartido en procesos si procesos > 1.
    Cada proceso recibe una copia del simulador al iniciar. Usa el método de inicio del sistema
    (spawn en Windows), así que funcion tiene que estar definida en un módulo y no en el cuaderno."""
    if procesos <= 1:
        return [funcion(sim, t) for t in tareas]
    import multiprocessing as mp
    with mp.Pool(procesos, initializer=_iniciar, initargs=(sim,)) as pool:
        return pool.map(_llamar, [(funcion, t) for t in tareas])


def correr_tarea(sim, tarea):
    """Una réplica de la tarea (cfg, escenario, sup, semilla). Devuelve el informe."""
    cfg, escenario, sup, semilla = tarea
    return sim.correr(cfg, escenario, sup, semilla)[0]


def replicar(sim, cfg=None, escenario='conservador', sup=None, n=10, semilla_base=1, procesos=1):
    """Corre n réplicas independientes (semillas semilla_base, ..., semilla_base + n - 1).
    Con la misma semilla_base, dos configuraciones usan números aleatorios comunes."""
    tareas = [(cfg or Configuracion(), escenario, sup or Supuestos(), semilla_base + i) for i in range(n)]
    return pd.DataFrame([asdict(inf) for inf in en_paralelo(sim, correr_tarea, tareas, procesos)])


def intervalo(x, nivel=0.95):
    """Media, desvío, semiamplitud t de Student y semiamplitud relativa."""
    x = np.asarray(x, float)
    n = len(x)
    m, s = x.mean(), x.std(ddof=1)
    h = st.t.ppf((1 + nivel) / 2, n - 1) * s / math.sqrt(n)
    return dict(media=m, desvio=s, semiamplitud=h, inf=m - h, sup=m + h, relativa=h / abs(m) if m else float('nan'), n=n)


def informe_replicas(df, variables=('IN', 'IL', 'CFP', 'UM_medio_pct', 'QM_medio', 'PA_medio', 'CD')):
    return pd.DataFrame({v: intervalo(df[v]) for v in variables}).T
