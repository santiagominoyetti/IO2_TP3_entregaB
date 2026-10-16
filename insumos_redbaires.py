"""Insumos del simulador RedBaires.

Lee TP3_datos.zip, aplica la limpieza del modelo de datos (parte A, sección 3.2 y Anexo C)
y estima cada parámetro de la Tabla 8 de la parte A. El resultado es un objeto Insumos que
el simulador usa tal cual. Los únicos valores fijados a mano son la dispersión de la binomial
negativa (α = 0,80, elegida en la parte A por verosimilitud perfilada, Tabla 9) y los supuestos
sin datos que la parte A fija explícitamente (PREM, indulgencia y horizonte), que se declaran en
la clase Supuestos del módulo simulador_redbaires.
"""
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf

NIVELES = ['nano', 'micro', 'macro', 'mega']
CORTES = [0, 1_000, 10_000, 100_000, np.inf]
ALFA_NB = 0.80          # dispersión elegida por verosimilitud perfilada en la parte A (Tabla 9)


def leer_datos(ruta_zip):
    """Devuelve (eventos crudos, parámetros de Finanzas) desde el zip entregado."""
    with zipfile.ZipFile(ruta_zip) as z:
        interno = zipfile.ZipFile(io.BytesIO(z.read('muestra_cascadas_REDBAIRES.zip')))
        raw = pd.read_csv(io.BytesIO(interno.read('muestra_cascadas_REDBAIRES.csv')))
        par = pd.read_csv(interno.open('parametros_y_costos.csv'))
    return raw, par


def limpiar(raw):
    """Limpieza del Anexo C de la parte A (solo los pasos que afectan a los insumos)."""
    df = raw.drop_duplicates().copy()
    for c in ['content_type_code', 'platform_surface_code', 'language_code']:
        df[c] = df[c].str.upper()
    df['t'] = pd.to_datetime(df.forward_datetime)
    centinela = df.t.dt.year == 2099
    por_id = df.set_index('event_id')
    t_padre = df.parent_event_id.map(por_id.t)
    rep = centinela & df.parent_event_id.notna()
    df.loc[rep, 't'] = t_padre[rep] + pd.to_timedelta(df.loc[rep, 'inter_event_time_min'], unit='m')
    for i in df.index[centinela & df.parent_event_id.isna()]:
        hijos = df[(df.parent_event_id == df.at[i, 'event_id']) & (df.t.dt.year < 2099)]
        df.at[i, 't'] = ((hijos.t - pd.to_timedelta(hijos.inter_event_time_min, unit='m')).median()
                         if len(hijos) else pd.NaT)
    por_id = df.set_index('event_id')
    dt_real = (df.t - df.parent_event_id.map(por_id.t)).dt.total_seconds() / 60
    neg = df.inter_event_time_min < 0
    df.loc[neg, 'inter_event_time_min'] = dt_real[neg]
    df.loc[df.actor_follower_count == 0, 'actor_follower_count'] = np.nan
    df['nivel'] = pd.cut(df.actor_follower_count, CORTES, right=False, labels=NIVELES)
    df['nociva'] = df.topic_code.str.startswith('DISARM').astype(int)
    df['c2'] = df.hashtag_cluster_ref.str.endswith('C2').astype(int)
    t0 = df[df.seed_flag == 1].set_index('cascade_id').t
    df['edad_min'] = (df.t - df.cascade_id.map(t0)).dt.total_seconds() / 60
    df['edad_rel'] = df.edad_min / df.content_half_life_min
    df['hijos'] = df.reshare_count_from_actor
    return df


@dataclass
class Insumos:
    # siembra
    tasa_dia_hora: np.ndarray            # 7 x 24, piezas por hora en la muestra (lunes = 0)
    dias_muestra: float
    semillas_por_dia: float
    # atributos de la pieza
    p_nociva: float
    p_c2: float
    nivel_semilla: dict                  # c2 -> probabilidades por nivel
    formatos: list
    p_formato: np.ndarray
    vm_sigma: float
    vm_mediana: float
    # reenvíos
    p_nivel_hijo: np.ndarray
    superficies: list
    p_superficie: np.ndarray
    p_verificada: np.ndarray             # por nivel
    coef: dict                           # coeficientes de la regresión binomial negativa
    r_nb: float
    m_reenvio: float                     # media de hijos por reenvío en la muestra
    m_semilla: float
    # retardo padre-hijo (hiperexponencial de la parte A)
    corte_retardo: float
    w_rapida: float
    media_rapida: float
    media_diferida: float
    media_exceso_diferida: float         # media de (retardo - corte) en la componente diferida
    # alcance
    seguidores: dict                     # nivel -> arreglo empírico de seguidores
    frac_c: float
    frac_escala: float
    # Centinela y moderación
    precision_ev: float
    recall_ev: float
    pi_ev: float
    piso_latencia: float
    escala: float                        # ESC calibrada a priori
    ndeg: int
    finanzas: dict
    # para la validación
    observado: dict = field(default_factory=dict)
    # retardo como mezcla de exponenciales con espera: [(peso, media, espera)]
    fases_retardo: list = field(default_factory=list)


def ajustar_mezcla_retardo(x, corte, iniciales=((0.30, 2.0, 0), (0.25, 20.0, 0), (0.25, 30.0, 1), (0.20, 200.0, 1)),
                           iteraciones=3000):
    """Mezcla de exponenciales por máxima verosimilitud (algoritmo EM). Las componentes marcadas con 1 en
    la tercera posición están desplazadas en el corte: su densidad es nula antes del corte.
    Devuelve [(peso, media, espera)] y la log-verosimilitud."""
    w = np.array([c[0] for c in iniciales], float)
    m = np.array([c[1] for c in iniciales], float)
    s = np.array([corte * c[2] for c in iniciales], float)
    for _ in range(iteraciones):
        y = x[:, None] - s[None, :]
        dens = np.where(y >= 0, np.exp(-np.maximum(y, 0) / m) / m, 0.0) * w
        tot = dens.sum(1)
        r = dens / tot[:, None]
        w = r.mean(0)
        m = (r * np.maximum(y, 0)).sum(0) / r.sum(0)
    return [(float(a), float(b), float(c)) for a, b, c in zip(w, m, s)], float(np.log(tot).sum())


def estimar_insumos(ruta_zip='TP3_datos.zip'):
    raw, par = leer_datos(ruta_zip)
    df = limpiar(raw)
    tam = df.groupby('cascade_id').size()
    tope = int(tam.max())
    completas = tam[tam < tope].index
    dfc = df[df.cascade_id.isin(completas)].copy()
    pieza = df[df.seed_flag == 1].set_index('cascade_id')
    pieza['tamano'] = tam

    # --- siembra: Poisson no estacionario por día de la semana y hora (parte A, 3.3)
    sem = df[df.seed_flag == 1].dropna(subset=['t']).sort_values('t').t
    inicio, fin = sem.min().floor('h'), sem.max().ceil('h')
    horas = pd.date_range(inicio, fin, freq='h', inclusive='left')
    cnt = pd.Series(1, index=sem.values).resample('h').sum().reindex(horas, fill_value=0)
    tasa = cnt.groupby([cnt.index.dayofweek, cnt.index.hour]).mean().unstack().reindex(index=range(7), columns=range(24)).values
    dias = (fin - inicio).total_seconds() / 86400

    # --- atributos de la pieza
    semillas = pieza.dropna(subset=['nivel'])
    niv_sem = pd.crosstab(semillas.c2, semillas.nivel, normalize='index').reindex(columns=NIVELES)
    form = pieza.content_type_code.value_counts(normalize=True)
    s_vm, _, esc_vm = st.lognorm.fit(pieza.content_half_life_min.dropna().values, floc=0)

    # --- reenvíos
    padre = dfc.set_index('event_id').nivel
    hij = dfc[dfc.seed_flag == 0].assign(np_=lambda v: v.parent_event_id.map(padre)).dropna(subset=['nivel', 'np_'])
    p_hijo = hij.nivel.value_counts(normalize=True).reindex(NIVELES).values
    surf = dfc[dfc.seed_flag == 0].platform_surface_code.value_counts(normalize=True)
    p_ver = df.groupby('nivel', observed=True).verified_flag.mean().reindex(NIVELES).values

    base = dfc.dropna(subset=['edad_rel', 'nivel']).copy()
    base['gen'] = np.where(base.seed_flag == 1, 0, base.cascade_depth - 1)
    formula = ('hijos ~ C(nivel, Treatment("nano")) + seed_flag + edad_rel + C(content_type_code, Treatment("IMG-MACRO"))'
               ' + C(platform_surface_code, Treatment("DM")) + c2 + verified_flag + nociva + gen')
    glm = smf.glm(formula, data=base, family=sm.families.NegativeBinomial(alpha=ALFA_NB)).fit()
    b = glm.params

    def cat(prefijo, valor):
        k = [i for i in b.index if i.startswith(prefijo) and i.endswith(f'[T.{valor}]')]
        return float(b[k[0]]) if k else 0.0
    coef = {'intercepto': float(b['Intercept']), 'semilla': float(b['seed_flag']), 'edad_rel': float(b['edad_rel']),
            'c2': float(b['c2']), 'verificada': float(b['verified_flag']), 'nociva': float(b['nociva']), 'gen': float(b['gen']),
            'nivel': np.array([cat('C(nivel', n) for n in NIVELES]),
            'formato': {f: cat('C(content_type', f) for f in form.index},
            'superficie': np.array([cat('C(platform_surface', s) for s in surf.index])}

    # --- retardo padre-hijo: hiperexponencial con corte en el salto de la densidad
    x = dfc.loc[dfc.seed_flag == 0, 'inter_event_time_min'].values
    x = x[x > 0]
    cnt_min = np.bincount(np.floor(x[x < 120]).astype(int), minlength=120)
    razon = cnt_min[5:] / np.array([cnt_min[i - 5:i].mean() for i in range(5, 120)])
    corte = float(5 + np.argmax(razon))
    rap, dif = x[x < corte], x[x >= corte]

    # --- alcance
    seg = {n: df.loc[df.nivel == n, 'actor_follower_count'].dropna().values for n in NIVELES}
    frac = (df.impressions / df.actor_follower_count).dropna()
    frac = frac[(frac > 0) & (frac < 1)].values
    c_fr, _, esc_fr = st.weibull_min.fit(frac, floc=0)

    # --- Centinela, latencia y escala (evaluación a priori, parte A 3.6)
    prec = df.loc[df.moderation_flag == 1, 'nociva'].mean()
    rec = df.loc[df.nociva == 1, 'moderation_flag'].mean()
    lat = df.loc[df.moderation_flag == 1, 'moderation_latency_min'].values
    piso = float(lat.min())
    finz = {r.param: dict(base=float(r.valor_base), lo=float(r.range_min), hi=float(r.range_max), unidad=r.unidad) for r in par.itertuples()}
    ts_min = finz['costo_revision_por_item_flag']['base'] / finz['costo_moderador_por_hora']['base'] * 60
    w_exc = (lat - piso).mean()
    lam = 1 / ts_min - 1 / w_exc
    marcadas_dia = df[df.moderation_flag == 1].cascade_id.nunique() / dias
    escala = lam * 1440 / marcadas_dia
    cuentas = df.dropna(subset=['actor_follower_count']).groupby('actor_id').actor_follower_count.max()

    # --- cascadas observadas (para la validación)
    marcada = df.groupby('cascade_id').moderation_flag.max()
    obs = {'tamano': pieza.tamano.values.astype(int),
           'profundidad': df.groupby('cascade_id').cascade_depth.max().reindex(pieza.index).values.astype(int),
           'duracion_h': (df.groupby('cascade_id').edad_min.max() / 60).reindex(pieza.index).values,
           'marcada': marcada.reindex(pieza.index).values.astype(int),
           'nociva': pieza.nociva.values.astype(int),
           'tope': tope, 'n_completas': len(completas), 'retardos': x,
           'latencias': lat, 'p_marca_aparicion': float(df.moderation_flag.mean()), 'marcadas_por_dia': float(marcadas_dia), 'lambda_casos_min': float(lam),
           'mu_casos_min': float(1 / ts_min)}

    return Insumos(
        tasa_dia_hora=tasa, dias_muestra=dias, semillas_por_dia=len(sem) / dias,
        p_nociva=float(pieza.nociva.mean()), p_c2=float(pieza.c2.mean()),
        nivel_semilla={int(c): niv_sem.loc[c].values for c in niv_sem.index},
        formatos=list(form.index), p_formato=form.values,
        vm_sigma=float(s_vm), vm_mediana=float(esc_vm),
        p_nivel_hijo=p_hijo, superficies=list(surf.index), p_superficie=surf.values, p_verificada=p_ver,
        coef=coef, r_nb=1 / ALFA_NB,
        m_reenvio=float(dfc[dfc.seed_flag == 0].hijos.mean()), m_semilla=float(dfc[dfc.seed_flag == 1].hijos.mean()),
        corte_retardo=corte, w_rapida=len(rap) / len(x), media_rapida=float(rap.mean()),
        media_diferida=float(dif.mean()), media_exceso_diferida=float(dif.mean() - corte),
        seguidores=seg, frac_c=float(c_fr), frac_escala=float(esc_fr),
        precision_ev=float(prec), recall_ev=float(rec), pi_ev=float(df.nociva.mean()),
        piso_latencia=piso, escala=float(escala), ndeg=int((cuentas >= 100_000).sum()), finanzas=finz,
        observado=obs, fases_retardo=ajustar_mezcla_retardo(x, corte)[0])
