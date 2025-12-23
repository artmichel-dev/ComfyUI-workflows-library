# 🎬 Workflow ComfyUI: Generación de Video con Self-Forcing + IPAdapter

## 📋 Descripción General

Este workflow de ComfyUI implementa un pipeline completo para la generación de videos de alta calidad utilizando técnicas avanzadas de difusión, condicionamiento visual mediante IPAdapter, interpolación de frames con RIFE, y upscaling con modelos especializados.

---

## 🤖 Modelos Utilizados

| Tipo              | Modelo                                   | Descripción                                                  |
| ----------------- | ---------------------------------------- | ------------------------------------------------------------ |
| **UNET**          | `self_forcing_dmd.pt`                    | Modelo de difusión Self-Forcing DMD para generación de video |
| **CLIP Text**     | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | Encoder de texto WAN (UMT5-XXL) en formato FP8               |
| **CLIP Vision**   | `clip_vision_h.safetensors`              | Encoder visual para procesamiento de imágenes de referencia  |
| **IPAdapter**     | `ip-adapter_sd15.safetensors`            | Adaptador para inyección de características visuales         |
| **VAE**           | `wan_2.1_vae.safetensors`                | Autoencoder variacional para decodificación de latentes      |
| **Upscaler**      | `4x-ClearRealityV1.pth`                  | Modelo de super-resolución 4x                                |
| **Interpolación** | `rife47.pth`                             | RIFE v4.7 para interpolación de frames                       |

---

## 🔄 Pipeline Técnico

```
┌─────────────────┐
│ Imagen          │
│ Referencia      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│ CLIP Vision     │─────▶│ IPAdapter    │
└─────────────────┘      └──────┬───────┘
                                │
         ┌──────────────────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│ UNET            │◀─────│ CLIP Text    │
│ Self-Forcing    │      │ Encoder      │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐
│ KSampler        │
│ (LCM, 8 steps)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ VAE Decode      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ RIFE VFI        │
│ (2x frames)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Upscale 4x      │
│ ClearReality    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Resize Final    │
│ 1440×960        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Export MP4      │
│ H.264/H.265     │
└─────────────────┘
```

---

## ⚙️ Configuración Detallada

### 1. **Condicionamiento Visual (IPAdapter)**

**Nodo:** `IPAdapterClipVisionEnhancer`

- **Imagen de referencia:** Cargada desde `LoadImage`
- **Weight:** 0.85 (influencia del 85% de la imagen de referencia)
- **Weight type:** Linear
- **Combine embeds:** Concat
- **Embeds scaling:** V only
- **Enhance tiles:** 1
- **Enhance ratio:** 1.0

El IPAdapter permite inyectar características visuales de una imagen de referencia directamente en el proceso de generación, guiando el estilo y composición del video resultante.

---

### 2. **Generación de Frames (Self-Forcing DMD)**

**Configuración del Latente:**

- **Resolución base:** 540×960 pixels
- **Frames iniciales:** 41 frames
- **Batch size:** 1

**Model Sampling (SD3):**

- **Shift:** 8 (control de distribución temporal del ruido)

**KSampler:**

- **Seed:** Randomize
- **Steps:** 8 (optimizado para Self-Forcing DMD)
- **CFG:** 1.0 (guidance scale mínimo)
- **Sampler:** LCM (Latent Consistency Model)
- **Scheduler:** Simple
- **Denoise:** 1.0 (denoising completo)

**Prompt de ejemplo:**

```
A young man sits confidently and calmly on an ornate gold baroque-style throne
inside a modern, minimalist interior. He wears sleek urban clothing: a black
jacket, light gray t-shirt, black pants, and beige sneakers, along with dark
sunglasses that convey confidence, power, and control.

Behind him, through a large floor-to-ceiling window, a deep blue nighttime sky
is visible with palm trees in silhouette. A black blimp slowly drifts across
the sky, displaying a glowing white message: "THE WORLD IS YOURS…".

The camera starts with a centered medium frontal shot and performs a slow,
cinematic dolly-in toward the subject. Warm lighting highlights the gold details
of the throne, while cool blue lighting fills the exterior background, creating
a dramatic color contrast.

The atmosphere feels powerful, ambitious, and aspirational. The man remains still,
breathing calmly, while the blimp moves smoothly and the text glows subtly.
Cinematic style, ultra-realistic, shallow depth of field, professional lighting,
modern luxury aesthetic, epic tone, 4K quality, subtle motion blur, contemporary
dramatic film look.
```

---

### 3. **Interpolación de Frames (RIFE)**

**Nodo:** `RIFE VFI`

- **Modelo:** `rife47.pth`
- **Clear cache after n frames:** 8
- **Multiplier:** 2x (duplica el número de frames)
- **Fast mode:** Habilitado
- **Ensemble:** Habilitado (mejora calidad)
- **Scale factor:** 1.0

**Resultado:** 41 frames → 82 frames interpolados

La interpolación con RIFE genera frames intermedios usando redes neuronales, creando transiciones más suaves y fluidas en el video final.

---

### 4. **Upscaling y Resize**

**Upscaling (4x):**

- **Modelo:** `4x-ClearRealityV1.pth`
- **Resolución intermedia:** 2160×3840 (4x de 540×960)

**Resize Final:**

- **Método:** Bicubic
- **Resolución final:** 1440×960
- **Crop:** Center

**Cálculo de dimensiones:**

```
Width:  540 × 2 (upscale factor) = 1080 → resize → 1440
Height: 960 × 2 (upscale factor) = 1920 → resize → 960
```

---

### 5. **Export de Video**

**Output 1 - Video Raw (16fps):**

- **Formato:** H.265 (HEVC) MP4
- **Frame rate:** 16 fps
- **CRF:** 22 (calidad alta)
- **Pixel format:** yuv420p10le (10-bit)
- **Prefix:** `Self_forcing`

**Output 2 - Video Upscaled (24fps):**

- **Formato:** H.264 MP4
- **Frame rate:** 24 fps
- **CRF:** 19 (calidad muy alta)
- **Pixel format:** yuv420p (8-bit)
- **Prefix:** `self_Forcing_upscale`

---

## 🔧 Nodos Especializados

### Gestión de Memoria

- **`easy cleanGpuUsed`:** Limpia VRAM después de interpolación
- **`LayerUtility: PurgeVRAM`:** Purga cache y modelos al finalizar
  - `purge_cache`: true
  - `purge_models`: true

### Utilidades Matemáticas

- **`SimpleMath+`:** Cálculo dinámico de dimensiones para upscaling
  - Width: `b * a` (donde a = ancho base, b = factor)
  - Height: `a * b` (donde a = altura base, b = factor)

---

## 📊 Especificaciones Técnicas

| Parámetro           | Valor Inicial | Valor Final                    |
| ------------------- | ------------- | ------------------------------ |
| **Resolución**      | 540×960       | 1440×960                       |
| **Frames**          | 41            | 82 (post-RIFE)                 |
| **Frame Rate**      | -             | 16fps (raw) / 24fps (upscaled) |
| **Duración aprox.** | -             | ~2.5s (raw) / ~3.4s (upscaled) |
| **Formato**         | Latent        | MP4 (H.264/H.265)              |

---

## 🎯 Características Destacadas

### ✅ Ventajas del Pipeline

1. **Condicionamiento visual preciso** mediante IPAdapter con CLIP Vision
2. **Generación rápida** con Self-Forcing DMD (solo 8 steps)
3. **Interpolación neural** para movimiento fluido
4. **Upscaling de alta calidad** con modelo especializado
5. **Gestión automática de VRAM** para prevenir OOM errors
6. **Dual output** (raw + upscaled) para comparación

### 🎨 Casos de Uso

- Generación de videos cinematográficos cortos
- Animación de imágenes estáticas con control de estilo
- Prototipado rápido de conceptos visuales
- Producción de contenido para redes sociales

---

## 📝 Notas Importantes

1. **VRAM requerida:** Mínimo 12GB recomendado (RTX 3060 o superior)
2. **Tiempo de generación:** ~2-5 minutos dependiendo de hardware
3. **Prompt negativo:** Vacío (el modelo Self-Forcing no lo requiere)
4. **LoRA:** Slot disponible en `Power Lora Loader` (actualmente deshabilitado)

---

## 🚀 Cómo Usar

1. **Cargar imagen de referencia** en el nodo `LoadImage`
2. **Ajustar prompt** en `CLIP Text Encode (Positive Prompt)`
3. **Configurar dimensiones** en nodos `INTConstant` (Width/Height/Length)
4. **Ejecutar workflow** (Queue Prompt)
5. **Revisar outputs** en carpeta `ComfyUI/output/`

---

## 🔗 Dependencias de Custom Nodes

- `comfy-core` (v0.3.34)
- `comfyui-videohelpersuite`
- `comfyui-frame-interpolation`
- `comfyui-easy-use`
- `comfyui_layerstyle`
- `comfyui_essentials`
- `rgthree-comfy`
- `comfyui-kjnodes`

---

## 📄 Licencia y Créditos

**Workflow creado por:** artmichel  
**Versión:** v33  
**Fecha:** Diciembre 2025

---

## 🐛 Troubleshooting

**Error: Out of Memory**

- Reducir resolución base (ejemplo: 480×720)
- Disminuir número de frames iniciales
- Desactivar upscaling o RIFE

**Frames con artefactos**

- Aumentar CFG (probar con 1.5-2.0)
- Ajustar IPAdapter weight (probar 0.6-0.9)
- Verificar calidad de imagen de referencia

**Video demasiado rápido/lento**

- Ajustar frame_rate en nodos `VHS_VideoCombine`
- Modificar número de frames iniciales
- Cambiar multiplier de RIFE

---

**¿Preguntas o mejoras?** Contacto: artmichel@protonmail.ch
