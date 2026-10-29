"""Coverage app: a small local server that runs the Coverage workflow on Comfy Cloud through the Comfy API.

Run:  python app/server.py      then open http://localhost:8800
Needs COMFY_API_KEY in the environment or in ~/.config/comfy/.env, and `pip install comfy-sdk`.

The page sends a selfie plus a place (a tile from the library, your own place photo, or a few typed words).
This server patches workflows/coverage.api.json, submits it with the Comfy SDK, and hands each output
back to the page the moment it lands.
"""
import base64, copy, json, os, re, threading, time, uuid
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from comfy_sdk import Comfy

APP = Path(__file__).resolve().parent
ROOT = APP.parent
WORKFLOW = json.loads((ROOT / "workflows" / "coverage.api.json").read_text())
PLACES = {p["id"]: p for p in json.loads((APP / "places.json").read_text())}
RUNS = Path(os.environ.get("COVERAGE_RUNS", APP / "runs"))
PORT = int(os.environ.get("PORT", 8800))

# Node ids in coverage.api.json
WHO = "405"            # LoadImage: your photo
SITUATION = "430"      # CustomCombo: "place | moment"
PLACE_OUT = "415:421"  # where the built place comes out (FLUX branch)
PLACE_SAVE = "427"
OUTPUTS = {            # node id -> what the page calls it, in the order they land
    "438": "full_body", "414": "hero",
    "8": "still_1", "14": "still_2", "20": "still_3", "26": "still_4", "32": "still_5",
    "11": "shot_1", "17": "shot_2", "23": "shot_3", "29": "shot_4", "35": "shot_5",
    "5": "scene",
}
PLACE_HEAD = ("Straight out of camera iPhone photo of a real place, completely empty of people, wide view at eye level, "
              "natural light, real textures, lived in, no text, no logos, no watermarks. "
              "The place (show only the place, no person): ")
DEFAULT_MOMENT = "taking in the place, calm and natural"


def api_key():
    if os.environ.get("COMFY_API_KEY"):
        return os.environ["COMFY_API_KEY"]
    env = Path.home() / ".config" / "comfy" / ".env"
    for line in env.read_text().splitlines():
        if line.startswith("COMFY_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("Put COMFY_API_KEY in your environment or ~/.config/comfy/.env")


client = Comfy(api_key=api_key())
runs = {}


def qwen_place(graph, text):
    """Build the place from typed words with Qwen Image 2512 (open weights) instead of a library tile."""
    graph.update({
        "901": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2512_bf16.safetensors", "weight_dtype": "default"}},
        "902": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "903": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "904": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["901", 0], "shift": 3.1}},
        "905": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["902", 0], "text": PLACE_HEAD + text}},
        "906": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["902", 0], "text": "plastic, CGI, render, illustration, oversaturated, blurry, people, person, text"}},
        "907": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1344, "height": 768, "batch_size": 1}},
        "908": {"class_type": "KSampler", "inputs": {"model": ["904", 0], "positive": ["905", 0], "negative": ["906", 0], "latent_image": ["907", 0],
                                                     "seed": int(time.time()) % 2**31, "steps": 30, "cfg": 4, "sampler_name": "euler", "scheduler": "simple", "denoise": 1}},
        "909": {"class_type": "VAEDecode", "inputs": {"samples": ["908", 0], "vae": ["903", 0]}},
    })
    return "909"


def build(place_id=None, place_text=None, have_place_photo=False):
    g = copy.deepcopy(WORKFLOW)
    if place_id:
        p = PLACES[place_id]
        situation = f"{p['place']} | {p['moment']}"
    else:
        situation = f"{place_text or 'Your own place'} | {DEFAULT_MOMENT}"
    g[SITUATION]["inputs"].update({"choice": situation, "option1": situation, "index": 0})

    if place_id or have_place_photo:
        g["900"] = {"class_type": "LoadImage", "inputs": {"image": "place.png"}}
        new_place = "900"
    else:
        new_place = qwen_place(g, place_text or "a quiet city street")
    for node in g.values():
        for k, v in node["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and str(v[0]) == PLACE_OUT:
                node["inputs"][k] = [new_place, v[1]]
    g.pop(PLACE_SAVE, None)

    # keep only what the outputs need (drops the old FLUX branch)
    keep, todo = set(), [n for n in OUTPUTS if n in g]
    while todo:
        n = todo.pop()
        if n in keep or n not in g:
            continue
        keep.add(n)
        todo += [str(v[0]) for v in g[n]["inputs"].values() if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int)]
    return {k: v for k, v in g.items() if k in keep}


def save_data_url(data_url, path):
    path.write_bytes(base64.b64decode(data_url.split(",", 1)[1]))


def run_job(rid, body):
    r = runs[rid]
    folder = RUNS / rid
    try:
        selfie = folder / "selfie.png"
        save_data_url(body["selfie"], selfie)
        place_photo = None
        if body.get("place_id"):
            place_photo = APP / "places" / f"{body['place_id']}.jpg"
        elif body.get("place_image"):
            place_photo = folder / "place.png"
            save_data_url(body["place_image"], place_photo)

        wf = client.workflows.from_json(build(body.get("place_id"), body.get("place_text"), bool(place_photo)))
        wf.set_input(WHO, "image", client.assets.from_file(selfie))
        if place_photo:
            wf.set_input("900", "image", client.assets.from_file(place_photo))
        job = client.submit(wf)
        r.update(status="running", job=job.id)

        while True:
            job.refresh()
            for out in job.outputs:
                key = OUTPUTS.get(out.node_id)
                if key and key not in r["files"]:
                    ext = Path(out.name).suffix or (".mp4" if "video" in out.content_type else ".png")
                    dest = folder / f"{key}{ext}"
                    out.to_file(dest)
                    r["files"][key] = f"/runs/{rid}/{dest.name}"
            if job.status in ("succeeded", "failed", "cancelled", "canceled"):
                r["status"] = "done" if job.status == "succeeded" else "failed"
                if job.status != "succeeded":
                    r["error"] = str(job.error)[:500]
                return
            time.sleep(4)
    except Exception as e:
        r.update(status="failed", error=str(e)[:500])


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(APP), **kw)

    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        m = re.match(r"^/api/run/([\w-]+)$", self.path)
        if m:
            r = runs.get(m.group(1))
            return self.send_json(r or {"status": "unknown"}, 200 if r else 404)
        m = re.match(r"^/runs/([\w-]+)/([\w.]+)$", self.path)
        if m:
            f = RUNS / m.group(1) / m.group(2)
            if not f.exists():
                return self.send_error(404)
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4" if f.suffix == ".mp4" else "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)
        if self.path == "/api/places":
            return self.send_json(list(PLACES.values()))
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/run":
            return self.send_error(404)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not body.get("selfie"):
            return self.send_json({"error": "Add a photo first"}, 400)
        rid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        (RUNS / rid).mkdir(parents=True, exist_ok=True)
        runs[rid] = {"id": rid, "status": "starting", "files": {}, "started": time.time()}
        threading.Thread(target=run_job, args=(rid, body), daemon=True).start()
        self.send_json({"id": rid})


if __name__ == "__main__":
    RUNS.mkdir(parents=True, exist_ok=True)
    print(f"Coverage is running at http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
