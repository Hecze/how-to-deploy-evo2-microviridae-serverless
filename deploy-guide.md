# How to Deploy Evo 2 Microviridae on Modal

Guía para desplegar el modelo [evo-design/evo-2-7b-8k-microviridae](https://huggingface.co/evo-design/evo-2-7b-8k-microviridae) como endpoint HTTPS serverless (pago por uso, escala a cero).

**Paper original:** [Generative design of novel bacteriophages with genome language models](https://www.biorxiv.org/content/10.1101/2025.09.12.675911v1)

## ¿Por qué Modal y no otra opción?

Antes de llegar a Modal evaluamos varias alternativas. Este es el resumen:

| Opción | Modelo de pago | Problema |
|---|---|---|
| **Hugging Face Inference Endpoints** | Por hora de GPU | El modelo no tiene inference provider configurado. Además, es más caro que Modal. |
| **AWS SageMaker + NVIDIA NIM** | Por hora de instancia | Usa el NIM oficial de NVIDIA, pero dejas una instancia encendida pagando aunque no la uses. |
| **Replicate** | Por inferencia | No tiene el modelo pre-configurado. Habría que buildearlo con Cog, similar a lo que hacemos acá. |
| **Modal** | **Por segundo de GPU** | Escala a cero. Pagas ~$0.001 por genoma. Crédito gratis de $30/mes. |

**Elegimos Modal porque es el único que combina:**
- **Pago por uso real** (no por hora). Si nadie llama al endpoint, pagas $0.
- **Infraestructura configurable** (elegir GPU, imagen Docker, etc.).
- **Sin vendor lock-in**: el código usa Python estándar con la librería oficial `evo2`. Si mañana quieres migrar a otra plataforma, el script de generación es el mismo; solo cambia cómo lo sirves.

Esta guía se enfoca en Modal, pero los conceptos clave se aplican a cualquier proveedor serverless:
- Separar el **router HTTP** (liviano, sin GPU) del **worker de inferencia** (GPU pesada).
- La lógica de generación (`Evo2(...).generate(...)`) es portátil. Solo cambia el decorador o la forma de exponer el endpoint.
- El dolor de cabeza son las **dependencias nativas** (flash-attn, CUDA). Una vez resuelto acá, migrar a otro lado es cuestión de copiar la imagen Docker.

## Requisitos

- Python 3.10+
- Una cuenta gratuita en [Modal](https://modal.com) (dan $30/mes de crédito)

## Paso a paso

### 1. Crear el proyecto

En tu terminal:

```bash
mkdir evo-phage-api
cd evo-phage-api
python -m venv .venv
```

En Windows:
```bash
.venv\Scripts\activate
```

En Mac/Linux:
```bash
source .venv/bin/activate
```

### 2. Instalar Modal

```bash
pip install modal
```

### 3. Autenticarte en Modal

```bash
python -m modal setup
```

Esto abrirá el navegador para que inicies sesión. Necesitas una cuenta en [modal.com](https://modal.com).

### 4. Crear el archivo `evo_api.py`

Copia exactamente este código en un archivo llamado `evo_api.py`:

```python
import modal
from modal import Image, App, fastapi_endpoint

gpu_image = (
    Image.from_registry(
        "pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel",
        add_python="3.11"
    )
    .run_commands(
        "MAX_JOBS=1 pip install flash-attn --no-build-isolation",
        "pip install git+https://github.com/arcinstitute/evo2.git",
        "pip install fastapi[standard]"
    )
)

app = App("evo-phage-api")

@app.function(
    image=gpu_image,
    gpu="A100",
    timeout=1800,
    min_containers=0,
    scaledown_window=300
)
def generate_genome(prompt: str, length: int, temperature: float):
    from evo2 import Evo2
    model = Evo2('evo2_7b_microviridae')
    output = model.generate(
        prompt_seqs=[prompt],
        n_tokens=length,
        temperature=temperature,
        top_k=4
    )
    return output.sequences[0]

api_image = Image.debian_slim().pip_install("fastapi[standard]")

@app.function(image=api_image)
@fastapi_endpoint(method="POST")
def api(data: dict):
    prompt = data.get("prompt", "ATGTTTAG")
    length = data.get("length", 5386)
    temperature = data.get("temperature", 0.7)
    genome = generate_genome.remote(prompt, length, temperature)
    return {"status": "ok", "genome": genome, "length": len(genome)}
```

### 5. Desplegar

```bash
modal deploy evo_api.py
```

**El deploy tarda entre 5 y 10 minutos.** Modal:
1. Descarga la imagen base de PyTorch con CUDA (~3 min)
2. Compila Flash Attention con `MAX_JOBS=1` para no agotar memoria (~3 min)
3. Instala la librería `evo2` desde GitHub (~30s)
4. Despliega el endpoint HTTPS (~2s)

Al terminar, Modal te mostrará la URL del endpoint.

### 6. Probar

```bash
curl -X POST https://tu-usuario--evo-phage-api-api.modal.run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "ATGTTTAG", "length": 5386, "temperature": 0.7}'
```

La primera llamada tarda ~30-60s (descarga del modelo de 14 GB). Las siguientes, ~2-5s.

## Parámetros del endpoint

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `prompt` | string | `"ATGTTTAG"` | Semilla de nucleótidos (4-9 óptimo) |
| `length` | int | `5386` | Largo del genoma a generar |
| `temperature` | float | `0.7` | 0.3 conservador, 0.9 diverso |

## Respuesta

```json
{
  "status": "ok",
  "genome": "ATGTTTAG...",
  "length": 5386
}
```

## Costos

- GPU A100: ~$0.00055/s → ~$0.0011/genoma (~2s de generación)
- GPU L4 (más lenta): ~$0.00023/s → ~$0.007/genoma (~30s)
- Crédito gratuito: $30/mes (~27,000 genomas en A100)
- **Sin uso = sin costo.** El endpoint escala a cero.

Si quieres cambiar a GPU L4 (más barata pero más lenta), reemplaza `gpu="A100"` por `gpu="L4"` en el código.

## Ver logs

```bash
modal app logs evo-phage-api
```

## Notas técnicas

- Este código usa la imagen oficial `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel` de Docker Hub, que ya trae CUDA y PyTorch preinstalados.
- Instala `evo2` desde GitHub (no desde PyPI) para evitar una dependencia rota de `transformer-engine`.
- Usa `MAX_JOBS=1` al compilar flash-attn para no agotar la memoria RAM durante el build.
- La función `api` (recibir peticiones) corre en una imagen liviana sin GPU, mientras que `generate_genome` usa la GPU A100.
- Todo lo que instala Modal queda cacheado para deploys futuros. Si solo cambias el código, el redeploy es instantáneo.
