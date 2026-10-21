# %% Importación de paquetes ============================
from pathlib import Path

# Módulos de numpy, pandas, matplotlib y scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
#import ruptures as rpt
#(Solo util para detección de quiebres estructurales, a veces no viene instalada en anaconda)
from scipy.stats import jarque_bera, probplot

# Módulos de statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss


# %% Cargar bases de datos en python usando rutas relativas =========================

# Obtener la ruta del directorio raíz
BASE_DIR = Path(__file__).resolve().parents[2]

# Obtener la ruta del directorio con los datos
DATA_DIR = BASE_DIR / "Poster prueba" / "datos"

# Rutas de las bases de datos 
ruta_exp = DATA_DIR / "base_1.xlsx" # Base de datos 

# %% limpiar la base de datos, pues los datos originales se descargaron de forma conjunta =========================

# Base de datos xlsx
expo_base = pd.read_excel(ruta_exp, 
                          header=0, 
                          skiprows=[1],
                          na_values=["-", "—", " "]
                          )

#Eliminar la columna no deseada 
columna_a_borrar = expo_base.columns[2]

base_limpia = expo_base.dropna().drop(columns=[columna_a_borrar])

#Eliminar filas con cualquier dato faltante (NA)
base_NA = base_limpia.dropna().reset_index(drop=True)
#Guardar el resultado en un nuevo archivo Excel
base_NA.to_excel(DATA_DIR / "base_NA.xlsx", index=False)



print(base_NA.head()) # Primeras observaciones
print(base_NA.tail()) # Últimas observaciones, Son DATAframe

#Indexar la fecha como índice de la Serie de tiempo
base_NA["Fecha"] = pd.to_datetime(base_NA["Fecha"], format="%d/%m/%Y")

#Crear la Serie, Pandas.series (type)
serie_tiempo = pd.Series(
    data=base_NA["Tasa global de participación, según rango de edad: de 26 a 45 años(Dato fin de mes)"].values,
    index=base_NA["Fecha"],
    name="TGP_26_45"
)

#Asegurar que los valores de la serie sean numéricos (remplazando comas por puntos si existen)
serie_tiempo = pd.to_numeric(
    serie_tiempo.astype(str).str.replace(',', '.'), 
    errors='coerce'
)

print(serie_tiempo.head())
# %%==========================
# Paso 1: Identificación
# ============================
# Gráfica de la serie de tiempo 
plt.figure(figsize=(10, 5))
plt.plot(serie_tiempo)
plt.title("Tasa global de participación(26-45), 2007-2020")
plt.xlabel("Fecha")
plt.ylabel("%")
plt.grid(True)
plt.show()
# %% FAC y FACP de la serie original

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    serie_tiempo,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC tasa global de participación")

plot_pacf(
    serie_tiempo,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP de la serie original")

plt.tight_layout()
plt.show()

# %% Tests de Raíz Unitaria 

# Test de Augmented Dickey Fuller (ADF)
adf_result = adfuller(serie_tiempo) # Test de Dickey Fuller sin intercepto y tendencia determínistica

# Nota: En el test de ADF si no rechazo la H0 la serie no es estacionaria
#       y si rechazo la H0 la serie es estacionaria. 

print("=== Test ADF ===")
print("Estadístico ADF:", adf_result[0])
print("p-valor:", adf_result[1])
print("Rezagos usados:", adf_result[2]) 
print("Observaciones:", adf_result[3])
print("Valores críticos:")
for nivel, valor in adf_result[4].items():
    print(f"{nivel}: {valor}")

if adf_result[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")

# Test KPSS
kpss_result = kpss(serie_tiempo, regression="c", nlags="auto")

# Nota: En prueba KPSS se interpreta al contario que una prueba ADF.
#       Si no rechazo la H0 la serie es estacionaria
#       y si rechazo la H0 la serie es no estacionaria. 

print("\n=== Test KPSS ===")
print("Estadístico KPSS:", kpss_result[0])
print("p-valor:", kpss_result[1])
print("Rezagos usados:", kpss_result[2])
print("Valores críticos:")
for nivel, valor in kpss_result[3].items():
    print(f"{nivel}: {valor}")

if kpss_result[1] < 0.05:
    print("KPSS: rechazamos H0. Según el test, la serie es no estacionaria.")
else:
    print("KPSS: no rechazamos H0. Según el test, La serie es estacionaria.")

# Nota: Según los resultados de la prueba ADF y KPSS, hay que 
#       diferenciar la serie. 
# %% diferenciar la serie de tiempo =========================
# Diferenciar la serie de tiempo para hacerla estacionaria
serie_tiempo_diff = serie_tiempo.diff().dropna()

# Gráfica de la serie de tiempo diferenciada
plt.figure(figsize=(10, 5))
plt.plot(serie_tiempo_diff)
plt.title("Serie diferenciada")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()
#No aplico log, pues la naturalidad de la serie es que no tiene tendencia exponencial, sino lineal.
# %% FAC y FACP de la serie diferenciada
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    serie_tiempo_diff,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC de diferenciada")

plot_pacf(
    serie_tiempo_diff,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP de diferenciada")

plt.tight_layout()
plt.show()
# %% Qué modelo es mejor para la serie diferenciada? =========================
# Para determinar el mejor modelo para la serie diferenciada, podemos utilizar criterios de información como A
# Definir la lista de modelos a evaluar (p, d, q) x (P, D, Q, s)
# Se basan en los picos de los rezagos 1, 3 y 12
candidatos = [
    ((1, 1, 1), (0, 0, 1, 12)),#sobreidentificación
    ((0, 1, 3), (0, 0, 1, 12)),#sobreidentificación
    ((0, 1, 1), (0, 0, 1, 12)), # Es de los mejores pero no todos sus estimadores son significativos, se descarta por parsimonia 
    ((1, 1, 1), (1, 0, 1, 12)),#sobreidentificación
    ((0, 1, 1), (0, 1, 1, 12)),#sobrediferenciación
    ((1, 1, 1), (1, 1, 1, 12)),#sobreidentificación y sobrediferenciación
    ((0, 1, 0), (0, 0, 1, 12)),# El unico que minimiza criterios de información, hubo convergencia en el metodo de estimación (estimadores coherentes y significativos)
    ((1, 1, 3), (1, 1, 1, 12)),#sobreidentificación
    ((0, 1, 3), (0, 1, 3, 12)),#sobreidentificación
    ((0, 1, 3), (0, 1, 2, 12)),#sobreidentificación
    ((1, 1, 1), (1, 1, 1, 3)), #Correlación en los residuales
    ((0, 1, 1), (0, 1, 1, 3)),
    ((1, 1, 1), (1, 0, 1, 3)),
    ((1, 1, 1), (1, 0, 1, 1)),
    ((0, 1, 1), (0, 1, 1, 1)),
    ((1, 1, 1), (1, 1, 1, 1))
]

resultados = []

for order, seasonal_order in candidatos:
    try:
        model = sm.tsa.statespace.SARIMAX(
            serie_tiempo,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False)
        resultados.append({
            'Modelo': f"SARIMA{order}x{seasonal_order}",
            'AIC': res.aic,
            'BIC': res.bic
        })
    except Exception as e:
        continue

# Convertir a DataFrame y ordenar por menor AIC
df_res = pd.DataFrame(resultados).sort_values('AIC').reset_index(drop=True)
print(df_res)

# %% Estimación del modelo SARIMA seleccionado =========================
# Seleccionar el modelo con menor AIC y BIC
modelo_ma1 = SARIMAX(
    serie_tiempo,
    order=(0, 1, 0),
    seasonal_order=(0, 0, 1, 12),
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False
)

# Objeto tipo "SARIMAX" del paquete statmodels
type(modelo_ma1)

#  Acá es donde se realiza la estimación del modelo MA(1)
estimacion_ma1 = modelo_ma1.fit(disp=False)

# Objeto tipo "SARIMAXResultsWrapper"
type(estimacion_ma1)

print(estimacion_ma1.summary())

################################################
#Uso de la libreria ruptures para detectar quiebres estructurales en la serie de tiempo, y así determinar fechas sugeridas para colocar dummies.
################################################

# Convertir la serie a un array NumPy
#points = serie_tiempo.values

# Aplicar el algoritmo PELT (Pruned Exact Linear Time) para detectar cambios en la media/varianza
#algo = rpt.Pelt(model="rbf").fit(points)
#result = algo.predict(pen=3) # Ajusta 'pen' (penalización): a mayor penalización, detecta menos quiebres

#fechas_quiebre = [serie_tiempo.index[i-1] for i in result[:-1]]
#print("Fechas sugeridas para colocar Dummies:", fechas_quiebre)

# Graficar los segmentos detectados
#rpt.display(points, result, figsize=(10, 5))
#plt.title("Puntos de Quiebre Estructural Detectados")
#plt.show()

dummies_list = []

d1 = pd.Series(0, index=serie_tiempo.index, name="dummy_2007_07_31")
if "2007-07-31" in d1.index:
    d1.loc["2007-07-31"] = 1
dummies_list.append(d1)

# Dummy 2: 2009-03-31
d2 = pd.Series(0, index=serie_tiempo.index, name="dummy_2009_03_31")
if "2009-03-31" in d2.index:
    d2.loc["2009-03-31"] = 1
dummies_list.append(d2)

# Dummy 3: 2009-08-31
d3 = pd.Series(0, index=serie_tiempo.index, name="dummy_2009_08_31")
if "2009-08-31" in d3.index:
    d3.loc["2009-08-31"] = 1
dummies_list.append(d3)

# Dummy 4: 2011-09-30
d4 = pd.Series(0, index=serie_tiempo.index, name="dummy_2011_09_30")
if "2011-09-30" in d4.index:
    d4.loc["2011-09-30"] = 1
dummies_list.append(d4)

# Dummy 5: 2015-01-31
d5 = pd.Series(0, index=serie_tiempo.index, name="dummy_2015_01_31")
if "2015-01-31" in d5.index:
    d5.loc["2015-01-31"] = 1
dummies_list.append(d5)

# Dummy 6: 2018-10-31
d6 = pd.Series(0, index=serie_tiempo.index, name="dummy_2018_10_31")
if "2018-10-31" in d6.index:
    d6.loc["2018-10-31"] = 1
dummies_list.append(d6)

exog_X = pd.concat(dummies_list, axis=1)

#Despues de estimar el modelo con todas las dummys solo d1 es significativa, por lo que se estima un modelo SARIMA con d1 como exógena.

modelo_dummies = sm.tsa.statespace.SARIMAX(
    serie_tiempo,
    exog=d1,
    order=(0, 1, 0),
    seasonal_order=(0, 0, 1, 12),
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

# Objeto tipo "SARIMAXResultsWrapper"
type(modelo_dummies)

print(modelo_dummies.summary())

# %% Validación de supuestos del modelo estimado =========================
# Residuales del modelo MA(1)
residuales1 = estimacion_ma1.resid.dropna()

# Los residuales son un objeto tipo Pandas.Series
type(residuales1)

# Gráfica de los residuales (deberían comportarse como ruido blanco)
plt.figure(figsize=(10, 5))
plt.plot(residuales1.iloc[1:-1])
plt.title("Residuales del modelo MA(1)")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

# Descripción de los residuales 
print(residuales1.describe())

# Residuals del modelo SARIMA con dummy 
residuales2 = modelo_dummies.resid.dropna()

# Los residuales son un objeto tipo Pandas.Series
type(residuales2)

# Gráfica de los residuales (deberían comportarse como ruido blanco)
plt.figure(figsize=(10, 5))
plt.plot(residuales2.iloc[1:-1])
plt.title("Residuales del modelo SARIMA con dummy")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

# Descripción de los residuales 
print(residuales2.describe())
# %% FAC y FACP de los residuales

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    residuales1,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC de los residuales")

plot_pacf(
    residuales1,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP de los residuales")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    residuales2,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC de los residuales (Dummy)")

plot_pacf(
    residuales2,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP de los residuales (Dummy)")

plt.tight_layout()
plt.show()

# %%Prueba Ljung-Box

ljung_box = acorr_ljungbox(
    residuales1,
    lags=[6, 12, 18, 24],
    model_df=1,  # Ajuste por el número de parámetros del modelo Sma
    return_df=True
)

print("Prueba Ljung-Box")
print(ljung_box)

ljung_box2 = acorr_ljungbox(
    residuales2,
    lags=[6, 12, 18, 24],
    model_df=2,  # Ajuste por el número de parámetros del modelo Sma + d1
    return_df=True
)

print("Prueba Ljung-Box con dummy")
print(ljung_box2)
# H0: no hay autocorrelación en los residuos.
# Si p-value > 0.05, no se rechaza H0.

# %% FAC y FACP de los residuales al cuadrado

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    residuales1**2,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC de los residuales al cuadrado")

plot_pacf(
    residuales1**2,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP de los residuales al cuadrado")

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    residuales2**2,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC de los residuales al cuadrado con dummy")

plot_pacf(
    residuales2**2,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP de los residuales al cuadrado con dummy")

plt.tight_layout()
plt.show()

# %%Prueba ARCH de heterocedasticidad

arch_test = het_arch(residuales1, nlags=12) 

print("Prueba ARCH")
print("LM statistic:", arch_test[0])
print("LM p-value:", arch_test[1])

arch_test = het_arch(residuales2, nlags=12) 

print("Prueba ARCH")
print("LM statistic:", arch_test[0])
print("LM p-value:", arch_test[1])

# H0: no hay efectos ARCH.
# Si p-value > 0.05, no se rechaza H0.

# %%Q-Q plot de los residuos

plt.figure(figsize=(6, 6))
probplot(residuales1.iloc[1:-1], dist="norm", plot=plt)
plt.title("Q-Q plot de los residuos")
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 6))
probplot(residuales2.iloc[1:-1], dist="norm", plot=plt)
plt.title("Q-Q plot de los residuos con dummy")
plt.grid(True)
plt.show()

# %%Prueba de normalidad Jarque-Bera

jb_test = jarque_bera(residuales1.iloc[1:-1])

print("Prueba Jarque-Bera")
print("Jarque-Bera statistic:", jb_test.statistic)
print("Jarque-Bera p-value:", jb_test.pvalue)

jb_test = jarque_bera(residuales2.iloc[1:-1])

print("Prueba Jarque-Bera")
print("Jarque-Bera statistic:", jb_test.statistic)
print("Jarque-Bera p-value:", jb_test.pvalue)

# H0: los residuos no siguen una distribución normal.
# Si p-value < 0.05, se rechaza H0.

# %%# %%PRONOSTICO

#Definir horizonte de pronóstico 
pasos_futuros = 12

#Generar las fechas futuras correspondientes a la frecuencia de la serie
fechas_futuras = pd.date_range(
    start=serie_tiempo.index[-1] + pd.offsets.MonthEnd(1),
    periods=pasos_futuros,
    freq="ME"
)

#Obtener el pronóstico y sus intervalos de confianza del modelo sin dummy
get_forecast_sin_dummy =estimacion_ma1.get_forecast(steps=pasos_futuros)

#Extraer la media predicha y sus intervalos de confianza al 95%
pronostico_media = get_forecast_sin_dummy.predicted_mean
intervalos_conf = get_forecast_sin_dummy.conf_int(alpha=0.05)

#Consolidar en un DataFrame
df_pronostico = pd.DataFrame({
    'Pronóstico': pronostico_media,
    'Límite Inferior (95%)': intervalos_conf.iloc[:, 0],
    'Límite Superior (95%)': intervalos_conf.iloc[:, 1]
}, index=fechas_futuras)

print("=== Pronóstico para los próximos 12 meses (Sin Dummy) ===")
print(df_pronostico)

#Gráfica del pronóstico
plt.figure(figsize=(12, 6))

#Serie histórica 
plt.plot(serie_tiempo.iloc[-36:], label="Histórico", color="blue", linewidth=1.5)

#Línea del pronóstico
plt.plot(pronostico_media.index, pronostico_media, label="Pronóstico SARIMA (sin dummy)", color="red", linestyle="--", marker="o")

#Intervalos de confianza
plt.fill_between(
    intervalos_conf.index,
    intervalos_conf.iloc[:, 0],
    intervalos_conf.iloc[:, 1],
    color="red",
    alpha=0.2,
    label="Intervalo de Confianza (95%)"
)

plt.title("Pronóstico TGP (26-45 años) - Modelo sin Exógena")
plt.xlabel("Fecha")
plt.ylabel("%")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()


#=============================================================================================
#Definir horizonte de pronóstico
pasos_futuros = 12

#Generar las fechas futuras correspondientes a la frecuencia de tu serie (fin de mes)
fechas_futuras = pd.date_range(
    start=serie_tiempo.index[-1] + pd.offsets.MonthEnd(1),
    periods=pasos_futuros,
    freq="ME"
)

#Crear las variables exógenas para el periodo de pronóstico
# Como la dummy 'd1' representa un evento puntual (2007-07-31), en el futuro toma valor 0
exog_futuro = pd.DataFrame(
    0, 
    index=fechas_futuras, 
    columns=d1.to_frame().columns
)

#Obtener el pronóstico y sus intervalos de confianza
get_forecast = modelo_dummies.get_forecast(steps=pasos_futuros, exog=exog_futuro)

# Extraer la media del pronóstico e intervalos de confianza del 95%
pronostico_media = get_forecast.predicted_mean
intervalos_conf = get_forecast.conf_int(alpha=0.05)

#Crear DataFrame con los resultados organizados
df_pronostico = pd.DataFrame({
    'Pronóstico': pronostico_media,
    'Límite Inferior (95%)': intervalos_conf.iloc[:, 0],
    'Límite Superior (95%)': intervalos_conf.iloc[:, 1]
}, index=fechas_futuras)

print("=== Pronóstico para los próximos 12 meses ===")
print(df_pronostico)

#Gráfica de la serie y el pronóstico con abanico de confianza
plt.figure(figsize=(12, 6))

#Dibujar histórico
plt.plot(serie_tiempo.iloc[-36:], label="Histórico", color="blue", linewidth=1.5)

#Dibujar pronóstico
plt.plot(pronostico_media.index, pronostico_media, label="Pronóstico SARIMAX", color="red", linestyle="--", marker="o")

#Sombreado del intervalo de confianza
plt.fill_between(
    intervalos_conf.index,
    intervalos_conf.iloc[:, 0],
    intervalos_conf.iloc[:, 1],
    color="red",
    alpha=0.2,
    label="Intervalo de Confianza (95%)"
)

plt.title("Pronóstico de la Tasa Global de Participación Usando Dummy (26-45 años)")
plt.xlabel("Fecha")
plt.ylabel("%")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% Evaluación Fuera de Muestra (Backtesting - Recorte de 12 Períodos) =========================

# 1. Separar la serie en datos de entrenamiento (Train) y evaluación (Test)
n_test = 12
train_serie = serie_tiempo.iloc[:-n_test]
test_serie = serie_tiempo.iloc[-n_test:]

# Ajustar las variables exógenas (dummy d1) para train y test
train_exog = d1.iloc[:-n_test]
test_exog = d1.iloc[-n_test:]

# 2. Reestimar el modelo con los datos de entrenamiento
modelo_backtest = SARIMAX(
    train_serie,
    exog=train_exog,
    order=(0, 1, 1),
    seasonal_order=(0, 0, 1, 12),
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

# 3. Generar el pronóstico sobre los 12 períodos recortados
forecast_backtest = modelo_backtest.get_forecast(steps=n_test, exog=test_exog)
pred_media = forecast_backtest.predicted_mean
pred_ic = forecast_backtest.conf_int(alpha=0.05)

# 4. Tabla comparativa entre el valor observado vs pronosticado
df_comparacion = pd.DataFrame({
    'Valor Real': test_serie,
    'Pronóstico': pred_media,
    'Error Absoluto': np.abs(test_serie - pred_media),
    'Error Porcentual (%)': np.abs((test_serie - pred_media) / test_serie) * 100,
    'Límite Inferior (95%)': pred_ic.iloc[:, 0],
    'Límite Superior (95%)': pred_ic.iloc[:, 1]
})

print("=== Comparación: Pronóstico vs Valor Real (Últimos 12 períodos) ===")
print(df_comparacion)

# 5. Métricas de precisión del pronóstico
rmse = np.sqrt(np.mean((test_serie - pred_media)**2))
mae = np.mean(np.abs(test_serie - pred_media))
mape = np.mean(np.abs((test_serie - pred_media) / test_serie)) * 100

print("\n=== Métricas de Error del Pronóstico ===")
print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.4f}")
print(f"MAE  (Error Absoluto Medio):           {mae:.4f}")
print(f"MAPE (Error Porcentual Absoluto Medio): {mape:.2f}%")

# 6. Gráfica de comparación
plt.figure(figsize=(12, 6))

# Mostrar los últimos 24 períodos de train para dar contexto visual
plt.plot(train_serie.iloc[-24:], label="Histórico (Entrenamiento)", color="blue", linewidth=1.5)

# Graficar los valores reales del periodo de prueba
plt.plot(test_serie.index, test_serie, label="Valor Real (Observado)", color="black", marker="o", linewidth=2)

# Graficar el pronóstico
plt.plot(pred_media.index, pred_media, label="Pronóstico SARIMAX", color="red", linestyle="--", marker="s", linewidth=2)

# Sombreado del intervalo de confianza
plt.fill_between(
    pred_ic.index,
    pred_ic.iloc[:, 0],
    pred_ic.iloc[:, 1],
    color="red",
    alpha=0.2,
    label="Intervalo de Confianza (95%)"
)

plt.title("Evaluación Fuera de Muestra: Pronóstico SARIMAX vs Datos Reales")
plt.xlabel("Fecha")
plt.ylabel("%")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()
