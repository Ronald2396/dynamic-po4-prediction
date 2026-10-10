"""Red neuronal propuesta: perceptrón multicapa (MLP) multihorizonte.

Entrada: 153 variables causales estandarizadas.
Salida: cambio de PO4 (ΔPO4) a 10, 30 y 60 min; la predicción es y(t) + Δ̂.
Arquitectura: 3 capas ocultas de 512 neuronas (LayerNorm + GELU + Dropout).
Entrenamiento: AdamW + OneCycle, pérdida de Huber, detención temprana.
El modelo final es el promedio (ensamble) de 5 redes con distinta semilla.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

torch.set_num_threads(16)
SEMILLAS = [42, 7, 123, 2024, 31415]


class MLP(nn.Module):
    def __init__(self, n_entradas, n_salidas, ancho=512, profundidad=3, dropout=0.1):
        super().__init__()
        capas, d = [], n_entradas
        for _ in range(profundidad):
            capas += [nn.Linear(d, ancho), nn.LayerNorm(ancho), nn.GELU(), nn.Dropout(dropout)]
            d = ancho
        capas.append(nn.Linear(d, n_salidas))
        self.net = nn.Sequential(*capas)

    def forward(self, x):
        return self.net(x)


@dataclass
class Predictor:
    modelo: MLP
    mu: np.ndarray
    sd: np.ndarray
    ym: np.ndarray
    ys: np.ndarray
    historial: list[dict] = field(default_factory=list)

    def predecir_delta(self, X: np.ndarray) -> np.ndarray:
        Z = np.nan_to_num((X - self.mu) / self.sd).astype("float32")
        self.modelo.eval()
        with torch.no_grad():
            out = torch.cat([self.modelo(torch.from_numpy(Z[i:i + 65536]))
                             for i in range(0, len(Z), 65536)]).numpy()
        return out * self.ys + self.ym

    def guardar(self, ruta, columnas):
        torch.save({"state": self.modelo.state_dict(), "mu": self.mu, "sd": self.sd, "ym": self.ym,
                    "ys": self.ys, "cols": columnas, "n_out": len(self.ym)}, ruta)

    @staticmethod
    def cargar(ruta):
        ck = torch.load(ruta, weights_only=False)
        m = MLP(len(ck["cols"]), ck["n_out"])
        m.load_state_dict(ck["state"])
        return Predictor(m, ck["mu"], ck["sd"], ck["ym"], ck["ys"]), ck["cols"]


def fijar_semilla(semilla: int) -> None:
    random.seed(semilla); np.random.seed(semilla); torch.manual_seed(semilla)


def entrenar_mlp(X, D, tr, va, semilla=42, epocas=40, lote=4096, lr=2e-3, ancho=512, profundidad=3,
                 paciencia=6) -> Predictor:
    """X: variables de entrada; D: objetivos delta (n, n_horizontes).
    Normalización con estadísticos de entrenamiento; detención temprana con el
    MSE de validación."""
    fijar_semilla(semilla)
    mu, sd = np.nanmean(X[tr], 0), np.nanstd(X[tr], 0) + 1e-6
    Z = np.nan_to_num((X - mu) / sd).astype("float32")
    ym, ys = D[tr].mean(0), D[tr].std(0) + 1e-6
    ZY = ((D - ym) / ys).astype("float32")
    xtr, ytr = torch.from_numpy(Z[tr]), torch.from_numpy(ZY[tr])
    xva, yva = torch.from_numpy(Z[va]), torch.from_numpy(ZY[va])
    modelo = MLP(X.shape[1], D.shape[1], ancho, profundidad)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    pasos_epoca = math.ceil(len(tr) / lote)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epocas * pasos_epoca)
    mejor, estado, sin_mejora, hist = math.inf, None, 0, []
    for ep in range(epocas):
        modelo.train()
        perm = torch.randperm(len(tr))
        acum = 0.0
        for i in range(pasos_epoca):
            b = perm[i * lote:(i + 1) * lote]
            pred = modelo(xtr[b])
            loss = nn.functional.huber_loss(pred, ytr[b], delta=1.0)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            acum += float(((pred.detach() - ytr[b]) ** 2).mean()) * len(b)
        modelo.eval()
        with torch.no_grad():
            vl = float(((modelo(xva) - yva) ** 2).mean())
        hist.append({"epoca": ep + 1, "mse_entrenamiento": acum / len(tr), "mse_validacion": vl})
        if vl < mejor - 1e-4:
            mejor, sin_mejora, estado = vl, 0, {k: v.clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    modelo.load_state_dict(estado); modelo.eval()
    return Predictor(modelo, mu, sd, ym, ys, hist)
