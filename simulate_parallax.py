import os
import cv2
import glob
import math
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import gradio as gr

# ==================== Utilidades núcleo (warp / parallax) ====================

def imread_any(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"No se pudo leer: {path}")
    return img

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def list_pairs(rgb_dir, th_dir, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    def index_dir(d):
        idx = {}
        for e in exts:
            for p in glob.glob(os.path.join(d, f"*{e}")):
                idx[os.path.splitext(os.path.basename(p))[0]] = p
        return idx

    rgb_idx = index_dir(rgb_dir)
    th_idx  = index_dir(th_dir)
    common = sorted(set(rgb_idx.keys()) & set(th_idx.keys()))
    pairs = [(b, rgb_idx[b], th_idx[b]) for b in common]
    return pairs  # [(basename, rgb_path, th_path), ...]

def compose_homography(w, h, dx_px=0.0, dy_px=0.0, persp_strength=0.0, rot_deg=0.0, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    cx, cy = w * 0.5, h * 0.5
    theta = math.radians(rot_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    R = np.array([[cos_t, -sin_t, 0.0],
                  [sin_t,  cos_t, 0.0],
                  [0.0,    0.0,   1.0]], dtype=np.float32)

    T1 = np.array([[1, 0, -cx],
                   [0, 1, -cy],
                   [0, 0,  1]], dtype=np.float32)
    T2 = np.array([[1, 0, cx],
                   [0, 1, cy],
                   [0, 0,  1]], dtype=np.float32)
    H_rot = T2 @ R @ T1

    H_trans = np.array([[1, 0, dx_px],
                        [0, 1, dy_px],
                        [0, 0, 1]], dtype=np.float32)

    max_shift = persp_strength * 0.02 * (w + h)
    tl = np.array([0 + np.random.uniform(-max_shift, max_shift*0.2),
                   0 + np.random.uniform(-max_shift*0.2, max_shift)], dtype=np.float32)
    tr = np.array([w + np.random.uniform(-max_shift*0.2, max_shift),
                   0 + np.random.uniform(-max_shift*0.2, max_shift)], dtype=np.float32)
    br = np.array([w + np.random.uniform(-max_shift*0.2, max_shift),
                   h + np.random.uniform(-max_shift, max_shift*0.2)], dtype=np.float32)
    bl = np.array([0 + np.random.uniform(-max_shift, max_shift*0.2),
                   h + np.random.uniform(-max_shift, max_shift*0.2)], dtype=np.float32)

    src = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    dst = np.stack([tl,tr,br,bl], axis=0)
    H_persp = cv2.getPerspectiveTransform(src, dst)

    H = H_trans @ H_persp @ H_rot
    return H

def apply_homography(img, H, out_size=None, border=cv2.BORDER_CONSTANT, border_value=0):
    if out_size is None:
        out_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, H, out_size, flags=cv2.INTER_LINEAR,
                               borderMode=border, borderValue=border_value)

def emulate_resolution_mismatch(img, scale_factor=1.0, blur_sigma=0.0):
    if abs(scale_factor - 1.0) < 1e-6 and blur_sigma <= 0:
        return img.copy()
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale_factor)))
    new_h = max(1, int(round(h * scale_factor)))
    interp_down = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_LINEAR
    tmp = cv2.resize(img, (new_w, new_h), interpolation=interp_down)
    res = cv2.resize(tmp, (w, h), interpolation=cv2.INTER_LINEAR)
    if blur_sigma <= 0 and abs(scale_factor - 1.0) > 1e-3:
        blur_sigma = max(0.0, abs(scale_factor - 1.0) * 1.25)
    if blur_sigma > 0:
        k = max(3, int(blur_sigma * 4) | 1)
        res = cv2.GaussianBlur(res, (k, k), blur_sigma)
    return res

def simulate_pair(rgb_img, th_img, dx, dy, persp, rot, scale, blur,
                  apply_to="thermal", relative=False, res_mismatch=True, force_th_3ch=False, seed=42):
    # Convertir térmica a 3 canales si se desea visualizar/guardar homogéneo
    rgb = rgb_img.copy()
    th  = th_img.copy()
    if rgb.ndim == 2: rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    if th.ndim == 2 and force_th_3ch: th = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

    def to_px(val, size): return val*size if relative else val

    def warp(img):
        h, w = img.shape[:2]
        H = compose_homography(w, h,
                               dx_px=to_px(dx, w),
                               dy_px=to_px(dy, h),
                               persp_strength=persp,
                               rot_deg=rot,
                               seed=seed)
        return apply_homography(img, H, out_size=(w, h), border=cv2.BORDER_CONSTANT, border_value=0)

    out_rgb, out_th = rgb.copy(), th.copy()
    if apply_to in ("thermal","both"):
        out_th = warp(out_th)
    if apply_to in ("rgb","both"):
        out_rgb = warp(out_rgb)

    if res_mismatch:
        if apply_to in ("thermal","both"):
            out_th  = emulate_resolution_mismatch(out_th, scale_factor=scale, blur_sigma=blur)
        if apply_to in ("rgb","both"):
            out_rgb = emulate_resolution_mismatch(out_rgb, scale_factor=scale, blur_sigma=blur)

    return out_rgb, out_th

def bgr_to_rgb(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def side_by_side(a, b, max_width=1200):
    # Redimensiona manteniendo proporción para que el total no exceda max_width
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    target_w_each = max_width // 2
    def resize_keep(img, target_w):
        h, w = img.shape[:2]
        scale = target_w / w
        return cv2.resize(img, (target_w, int(round(h*scale))), interpolation=cv2.INTER_AREA)
    a2 = resize_keep(a, target_w_each)
    b2 = resize_keep(b, target_w_each)
    h = max(a2.shape[0], b2.shape[0])
    canvas = np.zeros((h, a2.shape[1]+b2.shape[1], 3), dtype=a2.dtype)
    canvas[:a2.shape[0], :a2.shape[1]] = a2
    canvas[:b2.shape[0], a2.shape[1]:] = b2
    return canvas

# ==================== Capa Gradio (UI) ====================

def refresh_pairs(rgb_dir, th_dir):
    pairs = list_pairs(rgb_dir, th_dir)
    choices = [p[0] for p in pairs]
    info = f"Pares encontrados: {len(choices)}"
    default = choices[0] if choices else None
    return gr.Dropdown(choices=choices, value=default), info

def load_images(rgb_dir, th_dir, basename, force_th_3ch=False):
    pairs = {b:(rp,tp) for b,rp,tp in list_pairs(rgb_dir, th_dir)}
    if basename not in pairs:
        return None, None, "Par no encontrado."
    rgb_path, th_path = pairs[basename]
    rgb = imread_any(rgb_path)
    th  = imread_any(th_path)
    if rgb.ndim == 2: rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    if th.ndim == 2 and force_th_3ch: th = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    return rgb, th, f"Cargado: {basename}"

def preview(rgb_dir, th_dir, basename, dx, dy, persp, rot, scale, blur,
            apply_to, relative, res_mismatch, force_th_3ch, seed):
    rgb, th, msg = load_images(rgb_dir, th_dir, basename, force_th_3ch)
    if rgb is None: 
        return None, None, msg
    out_rgb, out_th = simulate_pair(rgb, th, dx, dy, persp, rot, scale, blur,
                                    apply_to=apply_to, relative=relative,
                                    res_mismatch=res_mismatch,
                                    force_th_3ch=force_th_3ch, seed=seed)
    before = side_by_side(bgr_to_rgb(rgb), bgr_to_rgb(th))
    after  = side_by_side(bgr_to_rgb(out_rgb), bgr_to_rgb(out_th))
    return before, after, msg

def process_all(rgb_dir, th_dir, dx, dy, persp, rot, scale, blur,
                apply_to, relative, res_mismatch, force_th_3ch, seed,
                out_rgb_dir, out_th_dir, save_params):
    pairs = list_pairs(rgb_dir, th_dir)
    if not pairs:
        return "No se encontraron pares."
    ensure_dir(out_rgb_dir)
    ensure_dir(out_th_dir)
    params_log = {}

    for basename, rgb_path, th_path in tqdm(pairs, desc="Procesando"):
        try:
            rgb = imread_any(rgb_path)
            th  = imread_any(th_path)
            if rgb.ndim == 2: rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
            if th.ndim == 2 and force_th_3ch: th = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

            out_rgb, out_th = simulate_pair(rgb, th, dx, dy, persp, rot, scale, blur,
                                            apply_to=apply_to, relative=relative,
                                            res_mismatch=res_mismatch,
                                            force_th_3ch=force_th_3ch, seed=seed)

            cv2.imwrite(os.path.join(out_rgb_dir, f"{basename}.png"), out_rgb)
            cv2.imwrite(os.path.join(out_th_dir,  f"{basename}.png"), out_th)

            if save_params:
                params_log[basename] = dict(dx=dx, dy=dy, persp=persp, rot=rot,
                                            scale=scale, blur=blur, apply_to=apply_to,
                                            relative=relative, res_mismatch=res_mismatch, seed=seed)
        except Exception as e:
            print(f"[WARN] Falló {basename}: {e}")

    if save_params and params_log:
        with open(os.path.join(out_th_dir, "_parallax_params.json"), "w", encoding="utf-8") as f:
            json.dump(params_log, f, indent=2, ensure_ascii=False)

    return f"Listo. Guardado en:\n- {out_rgb_dir}\n- {out_th_dir}"

with gr.Blocks(title="Simulador de Parallax RGB–T") as demo:
    gr.Markdown("## Simulador de Parallax RGB–T\nAjusta los parámetros y observa el efecto en un par de ejemplo.")

    with gr.Row():
        rgb_dir     = gr.Textbox(value="./rgb", label="Carpeta RGB")
        th_dir      = gr.Textbox(value="./thermal", label="Carpeta Térmica")
        refresh_btn = gr.Button("🔄 Buscar pares")
        info_pairs  = gr.Markdown("")

    basename_dd = gr.Dropdown(choices=[], label="Selecciona un par (basename)")

    with gr.Row():
        dx   = gr.Slider(-80, 80, value=8, step=1, label="dx (px)")
        dy   = gr.Slider(-40, 40, value=0, step=1, label="dy (px)")
        rot  = gr.Slider(-3.0, 3.0, value=0.0, step=0.1, label="Rotación (°)")

    with gr.Row():
        persp = gr.Slider(0.0, 1.0, value=0.2, step=0.01, label="Perspectiva (0..1)")
        scale = gr.Slider(0.8, 1.1, value=0.90, step=0.01, label="Escala (mismatch resolución)")
        blur  = gr.Slider(0.0, 3.0, value=0.0, step=0.1, label="Blur σ extra")

    with gr.Row():
        apply_to = gr.Radio(choices=["thermal","rgb","both"], value="thermal", label="Aplicar a")
        relative = gr.Checkbox(False, label="dx/dy relativos al tamaño")
        res_mismatch = gr.Checkbox(True, label="Simular mismatch de resolución")
        force_th_3ch = gr.Checkbox(False, label="Forzar térmica a 3 canales (visualización)")
        seed = gr.Number(value=42, precision=0, label="Semilla (reproducibilidad)")

    preview_btn = gr.Button("👁️ Previsualizar")
    with gr.Row():
        before_img = gr.Image(label="Antes (RGB | Thermal)", interactive=False)
        after_img  = gr.Image(label="Después (RGB | Thermal)", interactive=False)
    status = gr.Markdown("")

    gr.Markdown("---")
    gr.Markdown("### Procesar todo el dataset (opcional)")
    with gr.Row():
        out_rgb_dir = gr.Textbox(value="./rgb_out", label="Salida RGB")
        out_th_dir  = gr.Textbox(value="./thermal_out", label="Salida Térmica")
        save_params = gr.Checkbox(True, label="Guardar JSON de parámetros")
        process_btn = gr.Button("💾 Procesar todo")

    # Lógica de eventos
    refresh_btn.click(refresh_pairs, inputs=[rgb_dir, th_dir], outputs=[basename_dd, info_pairs])
    preview_btn.click(
        preview,
        inputs=[rgb_dir, th_dir, basename_dd, dx, dy, persp, rot, scale, blur,
                apply_to, relative, res_mismatch, force_th_3ch, seed],
        outputs=[before_img, after_img, status]
    )
    process_btn.click(
        process_all,
        inputs=[rgb_dir, th_dir, dx, dy, persp, rot, scale, blur,
                apply_to, relative, res_mismatch, force_th_3ch, seed,
                out_rgb_dir, out_th_dir, save_params],
        outputs=[status]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )


