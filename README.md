# Edgebet Demo

**Demo publico del proyecto edgebet** — prediccion de futbol con modelos de ML.

> Este repository es PUBLICO. Contiene solo un demo simplificado.
> El codigo completo del sistema de produccion esta en un repository PRIVADO.

## Que es esta demo

Una aplicacion web que permite:
1. Probar predicciones con un modelo ONNX ligero
2. Entender la metodologia del proyecto
3. Configurar tu propio entorno local

## Limitaciones del demo

| Aspecto | Demo | Produccion |
|---|---|---|
| Features | ~10 basicas | ~245 de contrato |
| Modelo | ONNX demo generico | LightGBM por liga/mercado |
| Datos | Simulados / user-provided | Understat + API-Football + FDCO |
| Mercados | 1X2, BTTS, O/U | +AH, corners, exactos |

## Correr localmente

```bash
pip install -r requirements.txt
python convert_model.py --demo --output models/demo.onnx
uvicorn server:app --host 0.0.0.0 --port 8000
```

## GitHub Pages

Sirve en: `https://cacelass.github.io/edgebet-demo/`

## Models ONNX

Se actualizan semanalmente via GitHub Actions (lunes 06:00 UTC).

## Privacidad

- Las API keys que introduces **NO se recopilan ni almacenan**
- Se usan exclusivamente en tu navegador

## Aviso

Las predicciones son solo para fines educativos. No constituyen consejo de apuestas.
