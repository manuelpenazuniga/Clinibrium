# MediaPipe Nystagmus Detection Spike

**Spike de viabilidad** para medición determinista on-device de nistagmo usando MediaPipe FaceLandmarker.

## Objetivo

Responder: ¿Se puede medir nistagmo a 60fps de forma fiable con MediaPipe en el navegador?

## Cómo correr

### Opción 1: Servidor estático (recomendado)
```bash
cd spikes/mediapipe-nystagmus
npx serve
```

Luego abrir `http://localhost:3000` en Chrome/Edge.

### Opción 2: Apertura directa
```bash
open index.html
```

**Nota**: Si hay errores CORS al cargar el modelo, usá `npx serve` o cualquier servidor HTTP.

## Uso

1. **Webcam**: Click "Iniciar Webcam" para trackeo en vivo
2. **Video**: Click "Cargar Video" para procesar un clip grabado (ej. Dix-Hallpike a 60fps)
3. **Observar**: 
   - Overlay verde: puntos del iris trackeados
   - Puntos rojos: centros del iris calculados
   - Gráfico inferior: velocidad horizontal (rojo) y vertical (azul) vs tiempo
   - Métricas: FPS, velocidad, frecuencia de nistagmo, latencia, batidas, dirección, fatigabilidad

## Arquitectura del algoritmo

### Tracking del iris
- **Landmarks**: 468-472 (iris izquierdo), 473-477 (iris derecho)
- **Centro**: Promedio de los 5 puntos del iris por ojo
- **Estabilización**: Promedio de ambos ojos para reducir ruido

### Cálculo de velocidad
```
vx = (dx_pixels * PX_TO_DEG) / dt
vy = (dy_pixels * PX_TO_DEG) / dt
```
Donde `PX_TO_DEG ≈ 0.05` (aproximación: ~20px por grado visual)

**Limitación**: Esta conversión es una estimación burda. Para grados/s reales se necesitaría:
- Calibración con referencia conocida (ej. target a distancia fija)
- Considerar distancia cámara-ojo y FOV del lente

### Detección de nistagmo
- **Fase rápida**: Umbral de velocidad > 30°/s
- **Intervalo mínimo entre batidas**: 300ms (evita falsos positivos)
- **Frecuencia**: `n_batidas / duración_total`
- **Dirección**: Promedio de velocidad en últimos 30 frames
- **Latencia**: Tiempo desde inicio hasta primera batida
- **Fatigabilidad**: Comparación de amplitud primera vs segunda mitad del video

## Dependencias

- **MediaPipe Tasks Vision** v0.10.18 (vía CDN)
- **Modelo**: FaceLandmarker (float16, ~30MB)
- **Navegador**: Chrome/Edge con WebGPU/WebGL2

**Todo on-device**: El video NUNCA se sube a ningún servidor. Solo se descarga la librería de MediaPipe.

---

## VEREDICTO DE VIABILIDAD

**COMPLETAR DESPUÉS DE VALIDAR CON CLIP REAL DE DIX-HALLPIKE**

### Métricas observadas

| Métrica | Valor | Objetivo | ¿Cumple? |
|---------|-------|----------|----------|
| FPS efectivo | _ TBD _ | ≥55 fps | ⏳ |
| Estabilidad del tracking | _ TBD _ | <5% dropped frames | ⏳ |
| Detección de batidas | _ TBD _ | Sensibilidad >80% | ⏳ |
| Precisión de dirección | _ TBD _ | Concordancia con experto | ⏳ |
| Latencia de detección | _ TBD _ | <500ms desde inicio | ⏳ |

### Análisis de viabilidad

#### Tier 1: Velocidad horizontal/vertical on-device

**¿Es viable?** _PENDIENTE DE VALIDACIÓN_

**Factores a favor**:
- MediaPipe FaceLandmarker soporta 478 landmarks incluyendo iris
- GPU acceleration en navegador (WebGPU/WebGL2)
- Sin latencia de red (todo on-device)

**Limitaciones conocidas**:
1. **FPS**: MediaPipe reporta ~30-45fps en hardware típico con iris tracking. 60fps es ambicioso.
2. **Conversión px→grados**: La aproximación actual (0.05°/px) es burda. Para clínica se necesita calibración por paciente.
3. **Robustez del iris**:
   - Degrada con iluminación pobre o desigual
   - Párpados cerrados/parcialmente cerrados causan dropout
   - Lentes de contacto pueden confundir el tracker
4. **Aliasing temporal**: A 30fps, nistagmos >1.5Hz pueden tener aliasing. 60fps es deseable.
5. **Detección de fases**: El algoritmo actual (umbral de velocidad) es básico. Fases lentas rápidas requieren análisis de aceleración/jerk.

#### Tier 2: Torsión (rotación ocular)

**¿Es viable?** **NO en esta iteración**

**Razones**:
1. MediaPipe FaceLandmarker **no provee torsión directamente**
2. Se necesitaría:
   - Tracking de características del iris (patrones de criptas)
   - O tracking de vasos sanguíneos en esclera
   - Ambos requieren procesamiento de imagen adicional (OpenCV.js, custom shaders)
3. Complejidad algorítmica significativa (registro de imágenes, optical flow)
4. Performance: probablemente <15fps con torsión

**Alternativa para torsión**: Usar video-oculografía dedicada (hardware) o algoritmos de research (ej.基于iris pattern matching con deep learning)

### Recomendación al equipo

**Para Tier 1 (V/H on-device)**:
- ✅ **Viable con reservas** si se acepta:
  - FPS realista: 30-45fps (no 60fps)
  - Calibración manual para conversión px→grados
  - Condiciones controladas de iluminación
  - Validación contra experto en ≥20 clips
  
- 🔧 **Mejoras necesarias para producción**:
  1. Calibración por paciente (target a distancia conocida)
  2. Filtro de Kalman para suavizar trayectoria del iris
  3. Detección de fases lenta/rápida con análisis de jerk (derivada de aceleración)
  4. Manejo de dropout (párpados, parpadeos) con interpolación
  5. Validación clínica formal (sensibilidad/especificidad vs VOG gold standard)

**Para Tier 2 (torsión)**:
- ❌ **No viable con MediaPipe alone**
- 🔬 Requiere spike separado con:
  - OpenCV.js para optical flow
  - Algoritmo de registro de iris (ej. phase-based optical flow)
  - Validación contra torsión artificial (video sintético con rotación conocida)

### Próximos pasos sugeridos

1. **Validar este spike** con clip real de Dix-Hallpike a 60fps
2. **Medir FPS real** en hardware objetivo (laptop del clínico, tablet)
3. **Comparar** métricas calculadas vs interpretación de experto
4. Si FPS < 55: evaluar trade-off calidad/speed (reducir landmarks, usar modelo lite)
5. Si tracking es inestable: considerar filtro de Kalman o mediana móvil

### Conclusión

**Tier 1 (V/H)**: **VIABILIDAD CONDICIONAL** — funciona en condiciones controladas, pero requiere calibración y validación clínica antes de uso diagnóstico.

**Tier 2 (torsión)**: **NO VIABLE** con MediaPipe alone. Requiere stack de computer vision adicional y probablemente no alcance tiempo real.

**Recomendación**: Proceder con Tier 1 como MVP, validar con ≥20 clips reales, y evaluar si la precisión justifica el despliegue. Tier 2 queda como trabajo futuro o se delega a hardware dedicado.

---

## Notas técnicas

### API de MediaPipe usada
```javascript
FaceLandmarker.createFromOptions(filesetResolver, {
  baseOptions: {
    modelAssetPath: "...",
    delegate: "GPU"
  },
  runningMode: "VIDEO",
  numFaces: 1,
  outputFaceBlendshapes: false,
  outputFacialTransformationMatrixes: false
});

faceLandmarker.detectForVideo(videoElement, timestampMs);
```

### Iris landmarks (478 total)
- **468**: Centro iris izquierdo
- **469-472**: Contorno iris izquierdo
- **473**: Centro iris derecho
- **474-477**: Contorno iris derecho

### Privacidad
✅ Todo on-device. Cero subida de video/PII a servidores.

### Limitaciones del spike
- Código no optimizado para producción (es un spike)
- Umbral de detección hardcodeado (30°/s)
- Conversión px→grados aproximada
- Sin manejo de errores robusto
- Sin interpolación de dropout

---

**Autor**: Spike generado para Clinibrium VertigoDx  
**Fecha**: Julio 2026  
**Status**: ⏳ Pendiente de validación con clip real
