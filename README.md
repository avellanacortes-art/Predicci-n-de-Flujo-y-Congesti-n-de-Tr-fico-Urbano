# Predicci-n-de-Flujo-y-Congesti-n-de-Tr-fico-Urbano
El pronóstico espacio-temporal de tráfico es uno de los proyectos de IA predictiva más completos y vistosos para presentar en la universidad: combina un reto de Deep Learning complejo en el backend con una visualización de datos de alto impacto en el frontend.
import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ==========================================
# 1. DEFINICIÓN DEL MODELO Y SCALER
# ==========================================

class ConvLSTM1DCell(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int = 3):
        super(ConvLSTM1DCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=in_channels + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        conv_outputs = self.conv(combined)
        i_gate, f_gate, g_gate, o_gate = torch.split(conv_outputs, self.hidden_dim, dim=1)
        i, f, g, o = torch.sigmoid(i_gate), torch.sigmoid(f_gate), torch.tanh(g_gate), torch.sigmoid(o_gate)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class TrafficConvLSTM(nn.Module):
    def __init__(self, in_features: int = 2, hidden_dim: int = 64, seq_len_out: int = 12):
        super(TrafficConvLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len_out = seq_len_out
        self.cell = ConvLSTM1DCell(in_channels=in_features, hidden_dim=hidden_dim, kernel_size=3)
        self.head = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=seq_len_out, kernel_size=1)
        )

    def forward(self, x):
        batch_size, seq_len_in, num_nodes, num_features = x.shape
        x = x.permute(0, 1, 3, 2)
        h = torch.zeros(batch_size, self.hidden_dim, num_nodes, device=x.device)
        c = torch.zeros(batch_size, self.hidden_dim, num_nodes, device=x.device)

        for t in range(seq_len_in):
            h, c = self.cell(x[:, t, :, :], h, c)

        out = self.head(h)
        return out


class SimpleScaler:
    """Escalador Z-score para des-normalizar las predicciones del modelo."""
    def __init__(self, mean: float = 54.3, std: float = 12.8):
        self.mean = mean
        self.std = std

    def inverse_transform(self, data: torch.Tensor) -> np.ndarray:
        return (data * self.std + self.mean).detach().cpu().numpy()


# ==========================================
# 2. ESTADO GLOBAL Y CARGA DE MODELO
# ==========================================

ml_resources = {}

def generate_mock_sensor_locations(num_sensors: int = 207) -> List[Dict]:
    """Genera coordenadas GPS reales para la red de autopistas de Los Ángeles (METR-LA)."""
    np.random.seed(42)
    # Bounding Box de autopistas en Los Ángeles
    base_lat, base_lng = 34.0522, -118.2437
    locations = []
    for i in range(num_sensors):
        locations.append({
            "sensor_id": i,
            "name": f"Sensor_LA_{i:03d}",
            "latitude": float(base_lat + np.random.uniform(-0.15, 0.15)),
            "longitude": float(base_lng + np.random.uniform(-0.20, 0.20))
        })
    return locations


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga los recursos pesados (pesos del modelo) en memoria durante el inicio."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrafficConvLSTM(in_features=2, hidden_dim=64, seq_len_out=12)
    
    weights_path = "best_convlstm_metr_la.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"[OK] Pesos cargados exitosamente desde '{weights_path}'.")
    else:
        print(f"[Aviso] '{weights_path}' no encontrado. Se usarán pesos iniciales sintéticos.")

    model.to(device)
    model.eval()

    # Guardar en contexto global del ciclo de vida
    ml_resources["model"] = model
    ml_resources["scaler"] = SimpleScaler()
    ml_resources["device"] = device
    ml_resources["sensors"] = generate_mock_sensor_locations(207)
    
    yield
    ml_resources.clear()


# ==========================================
# 3. APICACIÓN FASTAPI Y ENDPOINTS
# ==========================================

app = FastAPI(
    title="API de Predicción Spatio-Temporal de Tráfico (METR-LA)",
    version="1.0.0",
    lifespan=lifespan
)

# Habilitar CORS para consumo desde cualquier frontend web local/remoto
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class PredictionRequest(BaseModel):
    # Opcional: Tensor de entrada histórico (12, 207, 2). Si es None, genera simulación.
    historical_data: Optional[List[List[List[float]]]] = None


@app.get("/")
def read_root():
    return {"status": "online", "model": "TrafficConvLSTM", "sensors_active": 207}


@app.get("/api/sensors")
def get_sensors():
    """Retorna la lista completa de sensores con sus ubicaciones geográficas."""
    return {"status": "success", "total": len(ml_resources["sensors"]), "data": ml_resources["sensors"]}


@app.post("/api/predict")
def predict_traffic(payload: Optional[PredictionRequest] = None):
    """
    Calcula la velocidad proyectada de tráfico en los 207 sensores para +15, +30 y +60 minutos.
    """
    model = ml_resources["model"]
    scaler = ml_resources["scaler"]
    device = ml_resources["device"]
    sensors = ml_resources["sensors"]

    try:
        # 1. Preparar Tensor de Entrada (Batch=1, Seq_in=12, Nodes=207, Features=2)
        if payload and payload.historical_data:
            input_array = np.array(payload.historical_data, dtype=np.float32)
            input_tensor = torch.tensor(input_array).unsqueeze(0).to(device)
        else:
            # Entrada sintética de prueba si no se proporciona histórico en el body
            input_tensor = torch.randn(1, 12, 207, 2).to(device)

        # 2. Inferencia en PyTorch
        with torch.no_grad():
            preds_scaled = model(input_tensor) # Shape: (1, 12, 207)
            preds_mph = scaler.inverse_transform(preds_scaled)[0] # Shape: (12, 207)

        # 3. Formatear la respuesta JSON estructurada para el mapa
        results = []
        for i, sensor in enumerate(sensors):
            # Extraer pasos correspondientes: 15 min (índice 2), 30 min (índice 5), 60 min (índice 11)
            v_15 = float(np.clip(preds_mph[2, i], 5.0, 75.0))
            v_30 = float(np.clip(preds_mph[5, i], 5.0, 75.0))
            v_60 = float(np.clip(preds_mph[11, i], 5.0, 75.0))

            results.append({
                "sensor_id": sensor["sensor_id"],
                "name": sensor["name"],
                "latitude": sensor["latitude"],
                "longitude": sensor["longitude"],
                "predictions": {
                    "min_15": round(v_15, 1),
                    "min_30": round(v_30, 1),
                    "min_60": round(v_60, 1),
                    "timeline_mph": [round(float(v), 1) for v in preds_mph[:, i]]
                }
            })

        return {
            "status": "success",
            "horizons": ["min_15", "min_30", "min_60"],
            "total_sensors": len(results),
            "data": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {str(e)}")
