import os
import io
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from run_geosteering import process_well, SAVE_DIR

app = FastAPI(title="GeoSteering API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/process")
def process(
    las_file: UploadFile = File(..., description="LAS file to process"),
    gr_target: float = Form(60.0),
    gr_band: float = Form(15.0),
    show_samples: Optional[int] = Form(None),
    save_png: bool = Form(True),
    save_csv: bool = Form(True),
    save_json: bool = Form(True),
):
    if not las_file.filename.lower().endswith(".las"):
        raise HTTPException(status_code=400, detail="File must be a .las")

    try:
        # Persist uploaded LAS to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".las") as tmp:
            tmp.write(las_file.file.read())
            tmp_path = tmp.name

        result = process_well(
            tmp_path,
            GR_TARGET=gr_target,
            GR_BAND=gr_band,
            show_samples='all' if show_samples is None else show_samples,
            save_png=save_png,
            save_csv=save_csv,
            plan_md_by_window=None,
            save_json=save_json,
        )

        if result is None:
            raise HTTPException(status_code=422, detail="Processing did not produce results (missing curves?)")

        return JSONResponse(content=result)
    finally:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


# Optional: root doc
@app.get("/")
def root() -> dict:
    return {"name": "GeoSteering API", "endpoints": ["/health", "/process"], "outputs_dir": SAVE_DIR}


