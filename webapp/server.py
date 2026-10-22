"""Web interface and REST API for the formation-energy predictor.

Run (from the project folder):
    python -m uvicorn webapp.server:app --port 8765
then open http://localhost:8765  (interactive API docs: http://localhost:8765/docs)

Endpoints
    GET  /api/overview            model scores, element list, data for the explorer map
    GET  /api/predict/{formula}   full prediction for one ABO3 oxide, e.g. /api/predict/SrTiO3
    POST /api/predict             same, body {"formula": "SrTiO3"}
    POST /api/batch               body {"formulas": ["SrTiO3", "BaZrO3", ...]} -> one row each
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project modules
from predictor import FormulaError, PerovskitePredictor  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
MAX_BATCH = 500
engine: PerovskitePredictor | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global engine
    engine = PerovskitePredictor()   # loads data + models once (~10-30 s)
    yield


app = FastAPI(
    title="HyQCGNN Perovskite Predictor",
    description="Formation energies of ABO3 oxide perovskites from XGBoost, a classical GNN "
                "and a hybrid quantum GNN. The first element of a formula is the A site.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def revalidate_ui_files(request, call_next):
    """Browsers must re-check the page and static files, so UI edits show up on reload."""
    response = await call_next(request)
    if not request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-cache"
    return response


class PredictRequest(BaseModel):
    formula: str = Field(..., examples=["SrTiO3"])


class BatchRequest(BaseModel):
    formulas: list[str] = Field(..., examples=[["SrTiO3", "BaZrO3", "LaAlO3"]])


def _predict(formula: str) -> dict:
    try:
        return engine.predict(formula)
    except FormulaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@app.get("/api/overview")
def overview() -> dict:
    return engine.overview


@app.get("/api/predict/{formula}")
def predict_get(formula: str) -> dict:
    return _predict(formula)


@app.post("/api/predict")
def predict_post(req: PredictRequest) -> dict:
    return _predict(req.formula)


@app.post("/api/batch")
def batch(req: BatchRequest) -> dict:
    if len(req.formulas) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"At most {MAX_BATCH} formulas per request.")
    rows = []
    for formula in req.formulas:
        if not formula.strip():
            continue
        try:
            r = engine.predict(formula)
        except FormulaError as exc:
            rows.append({"input": formula, "error": str(exc)})
            continue
        d = r["descriptors"]
        rows.append({
            "input": formula,
            "formula": r["formula"],
            "xgboost": r["predictions"]["xgboost"]["e_form"],
            "classical": r["predictions"]["classical"]["e_form"],
            "hybrid": r["predictions"]["hybrid"]["e_form"],
            "dft": r["dft"],
            "split": r["split"],
            "structure_source": r["structure_source"],
            "tolerance_factor": d["tolerance_factor"],
            "octahedral_factor": d["octahedral_factor"],
            "tau": d["tau"],
            "lattice_constant": d["lattice_constant"],
        })
    return {"rows": rows}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
