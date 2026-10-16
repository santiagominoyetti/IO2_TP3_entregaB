"""Validación del simulador RedBaires contra la muestra de cascadas.

Prueba de bondad de ajuste por Monte Carlo (bootstrap paramétrico):
  1. Una simulación de referencia grande estima la distribución del modelo.
  2. Se calcula un estadístico de discrepancia entre la muestra observada y la referencia.
  3. Se simulan R muestras independientes del mismo tamaño que la observada y se calcula el mismo
     estadístico para cada una: es su distribución cuando el modelo es correcto.
  4. Valor p = (1 + cantidad de muestras simuladas con estadístico >= observado) / (R + 1).

Estadísticos: chi cuadrado por intervalos para el tamaño y la profundidad, Kolmogorov-Smirnov para la
duración y diferencia absoluta de proporciones para la fracción de piezas marcadas.
"""
import numpy as np
import pandas as pd

from simulador_redbaires import Supuestos, en_paralelo

INTERVALOS = {
    'tamano': [1, 2, 3, 4, 6, 11, 21, 51, 101, 500, 10 ** 9],
    'profundidad': [0, 1, 2, 3, 4, 5, 6, 8, 11, 10 ** 9],
}
ETIQUETAS = {
    'tamano': ['1', '2', '3', '4-5', '6-10', '11-20', '21-50', '51-100', '101-499', '500 (tope)'],
    'profundidad': ['0', '1', '2', '3', '4', '5', '6-7', '8-10', '11 o más'],
}

def _lote(sim, tarea):
    semilla, n, tope, retardo = tarea
    return sim.simular_cascadas(n, semilla=semilla, tope=tope, sup=Supuestos(retardo=retardo))


def simular_lotes(sim, semillas, n, tope, retardo, procesos=2):
    return en_paralelo(sim, _lote, [(s, n, tope, retardo) for s in semillas], procesos)


def _chi2(x, p, bins):
    o = np.histogram(x, bins=bins)[0]
    e = len(x) * p
    return float(((o - e) ** 2 / e).sum())


def _ks(x, ref_ordenada):
    """Distancia de Kolmogorov-Smirnov entre la muestra x y la referencia, evaluada en todos los saltos."""
    x = np.sort(x)
    puntos = np.unique(np.concatenate([x, ref_ordenada]))
    fx = np.searchsorted(x, puntos, side='right') / len(x)
    fr = np.searchsorted(ref_ordenada, puntos, side='right') / len(ref_ordenada)
    return float(np.abs(fx - fr).max())


def estadisticos(muestra, ref):
    """Estadísticos de discrepancia de una muestra (dict de arreglos) contra la referencia."""
    return {
        'tamano': _chi2(muestra['tamano'], ref['p_tamano'], INTERVALOS['tamano']),
        'profundidad': _chi2(muestra['profundidad'], ref['p_profundidad'], INTERVALOS['profundidad']),
        'duracion': _ks(muestra['duracion_h'], ref['duracion']),
        'marcada': abs(float(np.mean(muestra['marcada'])) - ref['p_marcada']),
    }


def prueba_bondad(sim, observado, retardo='h2', n_ref=200_000, R=200, lote=5_000, semilla=1, procesos=2):
    """Devuelve (tabla de pruebas, distribuciones comparadas, referencia).
    Las muestras simuladas tienen el mismo tamaño que la observada (sin las duraciones faltantes)."""
    tope = int(observado['tope'])
    # referencia
    k = n_ref // lote
    ref_df = pd.concat(simular_lotes(sim, [semilla * 100_000 + i for i in range(k)], lote, tope, retardo, procesos),
                       ignore_index=True)
    ref = {'p_tamano': np.histogram(ref_df.tamano, INTERVALOS['tamano'])[0] / len(ref_df),
           'p_profundidad': np.histogram(ref_df.profundidad, INTERVALOS['profundidad'])[0] / len(ref_df),
           'duracion': np.sort(ref_df.duracion_h.values), 'p_marcada': float(ref_df.marcada.mean())}
    # observado
    ok = ~np.isnan(observado['duracion_h'])
    obs = {'tamano': observado['tamano'], 'profundidad': observado['profundidad'],
           'duracion_h': observado['duracion_h'][ok], 'marcada': observado['marcada']}
    n = len(observado['tamano'])
    t_obs = estadisticos(obs, ref)
    # muestras simuladas bajo el modelo
    muestras = simular_lotes(sim, [semilla * 100_000 + 50_000 + i for i in range(R)], n, tope, retardo, procesos)
    t_sim = pd.DataFrame([estadisticos({'tamano': m.tamano.values, 'profundidad': m.profundidad.values,
                                         'duracion_h': m.duracion_h.values[:ok.sum()], 'marcada': m.marcada.values}, ref)
                          for m in muestras])
    filas = []
    nombres = {'tamano': 'Tamaño (chi cuadrado, 10 intervalos)', 'profundidad': 'Profundidad (chi cuadrado, 9 intervalos)',
               'duracion': 'Duración (Kolmogorov-Smirnov)', 'marcada': 'Proporción de piezas marcadas (diferencia absoluta)'}
    for v in t_obs:
        filas.append({'variable': nombres[v], 'estadístico observado': t_obs[v],
                      'percentil 95 simulado': float(np.quantile(t_sim[v], 0.95)),
                      'valor p': (1 + int((t_sim[v] >= t_obs[v]).sum())) / (R + 1)})
    tabla = pd.DataFrame(filas).set_index('variable')
    comparacion = {}
    for v in ('tamano', 'profundidad'):
        o = np.histogram(obs[v], INTERVALOS[v])[0] / n
        comparacion[v] = pd.DataFrame({'observado': o, 'simulado': ref['p_' + v]}, index=ETIQUETAS[v])
    return tabla, comparacion, ref_df
