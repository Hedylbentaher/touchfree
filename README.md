# TouchFree ✋ : contrôle sans contact par gestes de la main

[![CI - qualité du code](https://github.com/Hedylbentaher/touchfree/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/Hedylbentaher/touchfree/actions/workflows/ci-tests.yml)
[![CI - images Docker](https://github.com/Hedylbentaher/touchfree/actions/workflows/docker-build.yml/badge.svg)](https://github.com/Hedylbentaher/touchfree/actions/workflows/docker-build.yml)

TouchFree reconnaît **18 gestes de la main + « main au repos »** devant une simple webcam
et les transforme en actions : diapositive suivante, zoom, pause d'une vidéo… Il sert partout
où l'on ne peut pas (ou ne veut pas) toucher le clavier : une présentation, un lecteur vidéo,
ou un chirurgien ganté au bloc opératoire.

Projet de module **Deep Learning avancé (axe 3 : détection d'objets)** et **MLOps** :
un seul projet, un seul jeu de données, un seul modèle, industrialisé de bout en bout.

*Hedyl Ben Taher, 5e année ingénieur Data Science & IA, TEK-UP. Encadrant : M. Haythem Ghazouani.*

---

## Démarrage

Prérequis : **Docker Desktop**, **DVC** et un jeton **DagsHub**.

```bash
# 1. Récupérer le modèle de production (versionné avec DVC)
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <utilisateur_dagshub>
dvc remote modify origin --local password <jeton_dagshub>
dvc pull models.dvc              # connexion lente : dvc pull models.dvc -j 1

# 2. Lancer toute l'application
docker compose up --build        # la première fois
docker compose up                # les fois suivantes (images déjà construites)
```

| Service | Adresse |
|---|---|
| Interface web (React + nginx) | http://localhost:8081 |
| API + documentation Swagger | http://localhost:8000/docs |
| Métriques Prometheus | http://localhost:8000/metrics |

Le port 8080 étant déjà utilisé sur le poste de développement, l'interface est publiée sur
le port **8081** (ligne `"8081:80"` de `docker-compose.yml`).

---

## Architecture

```
               ┌──────────── Entraînement (Google Colab, GPU T4) ────────────┐
 HaGRID ──► sous-ensemble ──► DVC ──► YOLO11 / Faster R-CNN ──► MLflow (DagsHub)
 (lecture partielle)       (DagsHub)        │                       │
                                            ▼                       ▼
                                export ONNX FP32 / FP16 / INT8 ──► benchmark
                                                                    │
                                   porte qualité (promote.py) ──► Model Registry
                                                                    │ alias production
                     ┌──────────── Service (Docker Compose) ────────┴──────────┐
 Navigateur ──► React (nginx :8081) ──/api──► FastAPI (:8000) ──► ONNX Runtime
```

## Structure du dépôt

| Dossier / fichier | Rôle |
|---|---|
| `params.yaml` | tous les réglages : données, entraînement, export, seuils de la porte qualité |
| `scripts/` | extraction des annotations, construction du sous-ensemble, vérification visuelle, porte qualité (`promote.py`) |
| `src/data/dataset.py` | conversion COCO → YOLO et contrôle anti-fuite (`user_id`) |
| `src/models/` | entraînement YOLO11 et Faster R-CNN, export ONNX, benchmark |
| `data/subset.dvc`, `models.dvc` | pointeurs DVC vers les données et les modèles (stockés sur DagsHub) |
| `reports/` | métriques JSON et tableau de comparaison (`comparison.md`) |
| `backend/` | API FastAPI, inférence ONNX en NumPy, 12 tests Pytest, Dockerfile multi-stage |
| `frontend/` | interface React (photo, webcam, boîtes, gestes, performances), nginx |
| `docker-compose.yml` | lancement de l'API et de l'interface en une commande |
| `.github/workflows/` | CI : style Python, tests, ESLint + build React, images Docker |

## Pipeline Deep Learning

1. **Données** : HaGRID, 18 gestes + `no_gesture`, sous-ensemble équilibré de **7 980 images**
   (300 / 60 / 60 par classe), 5 images au plus par personne, 512 px. Aucune personne
   (`user_id`) dans deux ensembles : vérifié automatiquement.
2. **Augmentation raisonnée** : miroir horizontal autorisé, **miroir vertical interdit**
   (`peace` deviendrait `peace_inverted`).
3. **Modèles** : YOLO11n (une étape, modèle avancé) contre Faster R-CNN MobileNetV3-Large FPN
   (deux étapes, baseline), pré-entraînés sur COCO puis fine-tunés.
4. **Évaluation** : mAP@[.5:.95] et mAP@.5 sur le jeu de **test**, jamais vu à l'entraînement.
5. **Optimisation** : export ONNX FP32, FP16 et INT8 (quantification statique calibrée sur
   200 images d'entraînement, tête Detect gardée en flottant).

## Résultats (jeu de test, 1 140 images)

| Modèle | mAP50-95 | mAP50 | Taille | Latence |
|---|---|---|---|---|
| Faster R-CNN MobileNetV3 (baseline, PyTorch) | 0,771 | 0,978 | – | 24,9 ms (GPU T4) |
| YOLO11n (PyTorch) | **0,849** | **0,991** | 5,5 Mo | 2,2 ms (GPU T4, lots de 16) |
| **YOLO11n ONNX FP32 — en production** | 0,842 | 0,991 | 10,6 Mo | 64,2 ms (CPU) |
| YOLO11n ONNX FP16 | 0,843 | 0,991 | 5,4 Mo | 80,5 ms (CPU) |
| YOLO11n ONNX INT8 (refusé par la porte qualité) | 0,696 | 0,867 | 3,2 Mo | 76,0 ms (CPU) |

Latence CPU : Colab Intel Xeon 2,00 GHz, de bout en bout (préparation + modèle + NMS),
moyenne sur 100 images. Dans Docker sur un PC portable (Intel i5-12500H), l'API répond en
**45 à 55 ms** par image.

- YOLO11n dépasse la baseline de **0,078** de mAP50-95 et est environ **10 fois plus rapide**.
  Faster R-CNN n'a reçu que 8 époques (contre 50) : la comparaison vaut à budget de temps
  comparable, pas à convergence.
- **Porte qualité** : l'INT8 perd 0,146 de mAP (seuil : 0,02) et n'accélère pas ce CPU ;
  il est refusé et le **FP32** est servi (`touchfree-detector` version 2, alias `production`).

## MLOps

| Besoin | Outil |
|---|---|
| Versionner les données et les modèles | DVC + stockage DagsHub |
| Tracer chaque expérience | MLflow (serveur DagsHub) : paramètres, courbes, artefacts, version Git |
| Gérer les versions du modèle | MLflow Model Registry, alias `production` |
| Ne promouvoir qu'un bon modèle | `scripts/promote.py` : mAP minimale, latence maximale, perte INT8 maximale |
| Servir le modèle | FastAPI asynchrone, typée (Pydantic), documentée (Swagger), sans PyTorch |
| Déployer | Dockerfiles multi-stage (utilisateur non-root, HEALTHCHECK) + Docker Compose |
| Vérifier à chaque push | GitHub Actions : black, flake8, Pytest, ESLint, build React, build Docker |
| Observer | endpoint `/metrics` (Prometheus) : requêtes, latence, gestes détectés |

Liens : [données et modèles (DagsHub)](https://dagshub.com/Hedylbentaher/touchfree) ·
[expériences MLflow](https://dagshub.com/Hedylbentaher/touchfree.mlflow)

## Jalons

| Jalon | Contenu | Validation |
|---|---|---|
| 1 | Données versionnées | suppression des données et du cache, puis `dvc pull` : 15 965 fichiers identiques |
| 2 | Entraînement suivi | YOLO11n 0,849 contre Faster R-CNN 0,771, runs dans MLflow |
| 3 | Service du modèle | API + 12 tests + image Docker de 168 Mo, conteneur `healthy`, résultat identique hors et dans Docker |
| 4 | Application complète | interface React (photo + webcam) et API lancées par `docker compose up`, CI complète verte |

## Développement local (sans Docker)

```bash
# API (depuis la racine du dépôt)
pip install -r backend/requirements-dev.txt
pytest backend/tests -v                               # 12 tests
uvicorn app.main:app --app-dir backend --reload       # http://localhost:8000/docs

# Interface (dans un 2e terminal ; /api est redirigé vers :8000)
cd frontend && npm ci && npm run dev                  # http://localhost:5173
```

## Limites

- Sous-ensemble de 7 980 images (environ 0,8 % de HaGRID), choisi pour tenir dans le temps
  et l'espace disponibles.
- Faster R-CNN non convergé (8 époques).
- Quantification INT8 statique min-max trop dégradante ; pistes : calibration par percentiles,
  plus d'images de calibration, entraînement conscient de la quantification (QAT).
- HaGRID contient surtout des mains d'adultes : confusions observées sur de petites mains
  (« Quatre » contre « Paume »).
