"""Builds workflows/coverage.app.json from the two official Comfy templates.

One hero frame in. Five camera angles out (Qwen Image Edit 2511 + multiple angles LoRA),
each one animated with LTX 2.5 as a handheld shot, then cut together into one scene.

Usage: python3 tools/build_coverage.py
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REF = os.path.join(ROOT, "templates")
OUT = os.path.join(ROOT, "workflows", "coverage.app.json")

# The coverage a camera crew shoots. Angles feed the CameraAngle node:
# horizontal 0 front, 90 right, 180 back; vertical negative looks up; zoom 0 wide, 10 close.
SHOTS = [
    {"name": "01 WIDE", "h": 0, "v": 12, "zoom": 0.5, "text": "Wide establishing shot."},
    {"name": "02 MEDIUM", "h": 340, "v": 5, "zoom": 6, "text": "Medium shot."},
    {"name": "03 CLOSE UP", "h": 10, "v": 0, "zoom": 9, "text": "Close up shot."},
    {"name": "04 REVERSE", "h": 180, "v": 5, "zoom": 4, "text": "Reverse angle, looking back the other way."},
    {"name": "05 LOW ANGLE", "h": 330, "v": -25, "zoom": 6, "text": "Low angle hero shot."},
]

# The operator. Locked off AI shots read as AI, so every shot gets a human on the camera.
CAMERA_OP = (
    "Shot handheld by a camera operator: gentle breathing float, small drifting reframes, "
    "natural motion blur on movement, subtle focus breathing. Real 35mm film texture, natural light. "
    "Same characters, same wardrobe, same location and same view out of the windows as the first frame."
)

DEFAULT_SCENE = "The scene comes to life. Keep the same characters, the same light and the same location."

SECONDS = 4
FPS = 24


def load(name):
    with open(os.path.join(REF, name)) as f:
        return json.load(f)


class Graph:
    def __init__(self):
        self.nodes, self.links = [], []
        self.next_node, self.next_link = 1, 1

    def add(self, node, pos):
        node = copy.deepcopy(node)
        node["id"] = self.next_node
        self.next_node += 1
        node["pos"] = list(pos)
        node["order"] = len(self.nodes)
        for i in node.get("inputs", []):
            i["link"] = None
        for o in node.get("outputs", []):
            o["links"] = []
        self.nodes.append(node)
        return node

    def new(self, type_, pos, widgets, inputs=(), outputs=(), size=(320, 120), title=None, props=None):
        node = {
            "id": 0, "type": type_, "pos": pos, "size": list(size), "flags": {}, "order": 0, "mode": 0,
            "inputs": [dict(i) for i in inputs], "outputs": [dict(o) for o in outputs],
            "properties": {"Node name for S&R": type_, **(props or {})}, "widgets_values": widgets,
        }
        if title:
            node["title"] = title
        return self.add(node, pos)

    def link(self, src, out_slot, dst, in_name):
        in_slot = next(k for k, i in enumerate(dst["inputs"]) if i["name"] == in_name)
        typ = src["outputs"][out_slot]["type"]
        lid = self.next_link
        self.next_link += 1
        self.links.append([lid, src["id"], out_slot, dst["id"], in_slot, typ])
        src["outputs"][out_slot]["links"].append(lid)
        dst["inputs"][in_slot]["link"] = lid
        return lid


def widget_input(name, typ):
    return {"name": name, "type": typ, "widget": {"name": name}, "link": None}


def build():
    angles_t = load("template_qwen_multiangle.app.json")
    ltx_t = load("video_ltx2_5_i2v.json")
    by_id = lambda t, i: next(n for n in t["nodes"] if n["id"] == i)
    qwen_inst, cam_t = by_id(angles_t, 219), by_id(angles_t, 221)
    ltx_inst, res_t = by_id(ltx_t, 398), by_id(ltx_t, 403)

    g = Graph()
    hero = g.new("LoadImage", [0, 0], ["hero.png", "image"],
                 inputs=[{**widget_input("image", "COMBO"), "label": "hero frame"}],
                 outputs=[{"name": "IMAGE", "type": "IMAGE"}, {"name": "MASK", "type": "MASK"}],
                 size=[340, 420], title="HERO FRAME")
    scene = g.new("PrimitiveStringMultiline", [0, 480], [DEFAULT_SCENE],
                  inputs=[{**widget_input("value", "STRING"), "label": "what happens"}],
                  outputs=[{"name": "STRING", "type": "STRING"}], size=[340, 180],
                  title="WHAT HAPPENS IN THE SCENE")
    res = g.add(res_t, [0, 720])
    res["widgets_values"] = ["16:9 (Widescreen)", 0.5, 32]

    concat = g.new("ConcatenateVideo", [2900, 0], ["auto"],
                   inputs=[{"name": f"videos.video{k}", "localized_name": f"video{k}", "type": "VIDEO",
                            **({"shape": 7} if k else {})} for k in range(len(SHOTS))]
                   + [{"name": "complete_audio", "type": "AUDIO", "shape": 7},
                      widget_input("codec", "COMBO")],
                   outputs=[{"name": "VIDEO", "type": "VIDEO"}], size=[300, 200], title="CUT THE SCENE")
    scene_out = g.new("SaveVideo", [3260, 0], ["coverage/scene", "auto", "auto"],
                      inputs=[{"name": "video", "type": "VIDEO"}], size=[640, 420], title="THE SCENE")
    g.link(concat, 0, scene_out, "video")

    shot_saves = []
    for k, s in enumerate(SHOTS):
        y = k * 760
        cam = g.add(cam_t, [420, y])
        cam["widgets_values"] = [s["h"], s["v"], s["zoom"], "camera"]
        cam["widgets_values_named"] = {"horizontal_angle": s["h"], "vertical_angle": s["v"],
                                       "zoom": s["zoom"], "view": "camera"}
        cam["title"] = s["name"] + " angle"
        g.link(hero, 0, cam, "image")

        angle = g.add(qwen_inst, [880, y])
        angle["title"] = s["name"] + " frame"
        angle["widgets_values"][-1] = 1000 + k
        angle["widgets_values_named"]["seed"] = 1000 + k
        g.link(hero, 0, angle, "image")
        g.link(cam, 1, angle, "prompt")

        still = g.new("SaveImage", [1320, y], [f"coverage/stills/{s['name'].replace(' ', '_')}"],
                      inputs=[{"name": "images", "type": "IMAGE"}], size=[340, 300], title=s["name"] + " still")
        g.link(angle, 0, still, "images")

        prompt = g.new("StringConcatenate", [1320, y + 340], ["", s["text"] + " " + CAMERA_OP, " "],
                       inputs=[widget_input("string_a", "STRING"), widget_input("string_b", "STRING"),
                               widget_input("delimiter", "STRING")],
                       outputs=[{"name": "STRING", "type": "STRING"}], size=[340, 200],
                       title=s["name"] + " camera op")
        g.link(scene, 0, prompt, "string_a")

        shot = g.add(ltx_inst, [1720, y])
        shot["title"] = s["name"] + " shot"
        wv = shot["widgets_values"]
        wv[0], wv[1], wv[2], wv[5], wv[6] = "", False, SECONDS, 2000 + k, FPS
        g.link(angle, 0, shot, "input")
        g.link(prompt, 0, shot, "value")
        g.link(res, 0, shot, "value_3")
        g.link(res, 1, shot, "value_4")

        save = g.new("SaveVideo", [2280, y], [f"coverage/shots/{s['name'].replace(' ', '_')}", "auto", "auto"],
                     inputs=[{"name": "video", "type": "VIDEO"}], size=[540, 360], title=s["name"])
        g.link(shot, 0, save, "video")
        g.link(shot, 0, concat, f"videos.video{k}")
        shot_saves.append(save["id"])

    subgraphs = angles_t["definitions"]["subgraphs"] + ltx_t["definitions"]["subgraphs"]
    wf = {
        "id": "c0e7a6e0-c0de-4c0e-8a6e-c00e2a9e0001",
        "revision": 0,
        "last_node_id": g.next_node - 1,
        "last_link_id": g.next_link - 1,
        "nodes": g.nodes,
        "links": g.links,
        "groups": [],
        "definitions": {"subgraphs": subgraphs},
        "config": {},
        "extra": {
            "linearMode": True,
            "linearData": {
                "inputs": [[hero["id"], "image"], [scene["id"], "value"]],
                "outputs": [scene_out["id"]] + shot_saves,
            },
            "ds": {"scale": 0.45, "offset": [100, 100]},
        },
        "version": 0.4,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(wf, f, indent=1)
    print("wrote", OUT, len(g.nodes), "nodes", len(g.links), "links")


if __name__ == "__main__":
    build()
