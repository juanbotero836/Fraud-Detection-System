# Fraud Detection System (MLOps API)

Este repositorio contiene un sistema extremo a extremo (**End-to-End**) para la detección automatizada de fraudes en transacciones bancarias utilizando técnicas avanzadas de Machine Learning y una arquitectura moderna de **MLOps**.

El proyecto expone un modelo entrenado mediante una API , empaquetada en Docker y desplegada con integración y despliegue continuos (**CI/CD**) en la nube de Microsoft Azure.

🌐 **API en Vivo (Producción):** [https://fraud-detection-app.jollyfield-cfd35558.eastus.azurecontainerapps.io/docs](https://fraud-detection-app.jollyfield-cfd35558.eastus.azurecontainerapps.io/docs)

---

## Tecnologías y Herramientas Utilizadas

- **Core:** Python 3.12, Pandas, Scikit-Learn
- **Machine Learning:** XGBoost (Árboles de decisión de gradiente aumentado)
- **API Framework:** FastAPI, Uvicorn, Swagger UI
- **Pruebas Unitarias:** Pytest
- **Contenerización:** Docker
- **CI/CD Pipeline:** GitHub Actions
- **Cloud Computing:** Microsoft Azure (Azure Container Registry & Azure Container Apps)

---

## Arquitectura del Sistema (Flujo MLOps)

El proyecto implementa un ciclo de vida automatizado que garantiza la estabilidad del sistema en cada cambio de código:

1. **Desarrollo Local:** Se entrena el modelo XGBoost y se empaqueta junto a la API en un entorno local controlado por Docker.
2. **Integración Continua (CI):** Cada `git push` a la rama `main` dispara un flujo en GitHub Actions que instala las dependencias y ejecuta las pruebas unitarias automatizadas con `pytest`.
3. **Despliegue Continuo (CD):** Si los tests pasan con éxito, GitHub Actions construye la imagen de Docker (2.31 GB) y la sube encriptada a **Azure Container Registry (ACR)**.
4. **Orquestación en la Nube:** **Azure Container Apps** detecta la nueva imagen del almacén, la descarga de forma transparente y actualiza la API en vivo en un microservicio con recursos optimizados (0.5 CPU y 1 Gi RAM).

---

