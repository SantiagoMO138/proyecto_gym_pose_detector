"""
Script para podar el modelo ViT de 22 clases a solo 2 clases relevantes:
  - bicep_curl  (original: "barbell biceps curl", id 0)
  - lateral_raise (original: "lateral raises", id 9)

El modelo podado se guarda en models/exercise_cnn_v5/
"""

import os
import shutil
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

SRC_MODEL = "models/workout_model_vit"
DST_MODEL = "models/exercise_cnn_v5"

# Índices de las clases originales que queremos conservar
IDX_BICEP_CURL = 0   # "barbell biceps curl"
IDX_LATERAL_RAISE = 9  # "lateral raises"


def main():
    print(f"Cargando modelo original desde: {SRC_MODEL}")
    processor = AutoImageProcessor.from_pretrained(SRC_MODEL)
    model = AutoModelForImageClassification.from_pretrained(SRC_MODEL)

    # El head del clasificador en ViT es model.classifier (nn.Linear)
    old_classifier = model.classifier  # Linear(768, 22)
    in_features = old_classifier.in_features  # 768

    # Crear nuevo head con solo 2 salidas
    new_classifier = nn.Linear(in_features, 2, bias=True)

    # Copiar los pesos y bias de las 2 clases seleccionadas
    with torch.no_grad():
        new_classifier.weight[0] = old_classifier.weight[IDX_BICEP_CURL].clone()
        new_classifier.weight[1] = old_classifier.weight[IDX_LATERAL_RAISE].clone()
        new_classifier.bias[0] = old_classifier.bias[IDX_BICEP_CURL].clone()
        new_classifier.bias[1] = old_classifier.bias[IDX_LATERAL_RAISE].clone()

    # Reemplazar el head
    model.classifier = new_classifier

    # Actualizar la configuración del modelo
    model.config.num_labels = 2
    model.config.id2label = {
        "0": "bicep_curl",
        "1": "lateral_raise",
    }
    model.config.label2id = {
        "bicep_curl": 0,
        "lateral_raise": 1,
    }

    # Guardar el modelo podado
    os.makedirs(DST_MODEL, exist_ok=True)
    model.save_pretrained(DST_MODEL)
    processor.save_pretrained(DST_MODEL)

    print(f"Modelo podado guardado en: {DST_MODEL}")
    print(f"Clases finales: {model.config.id2label}")

    # Copiar archivos auxiliares que faltan (si los hay)
    for aux_file in ["preprocessor_config.json"]:
        src_path = os.path.join(SRC_MODEL, aux_file)
        dst_path = os.path.join(DST_MODEL, aux_file)
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copiado: {aux_file}")

    print("\nPoda completada exitosamente.")


if __name__ == "__main__":
    main()
