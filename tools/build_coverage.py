"""Builds workflows/coverage.app.json from the two official Comfy templates.

A person and a place in. A hero frame, then five camera angles (Qwen Image Edit 2511 + multiple angles LoRA),
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
    {"name": "04 INSERT", "h": 0, "v": 35, "zoom": 10, "text": "Insert close up on the hands and what they are holding."},
    {"name": "05 LOW ANGLE", "h": 330, "v": -25, "zoom": 6, "text": "Low angle hero shot."},
]

# The rule book: what must never happen. These models run at CFG 1 (fast distilled), where a negative
# prompt is ignored, so the rules are written into every positive prompt instead.
RULES = (
    "Rules: every person appears only once, no duplicates, no twins, no extra people. "
    "Natural human anatomy, two hands with five fingers, true to life scale between people and objects. "
    "Phones, screens and papers are seen from the back or edge on and held the right way round. "
    "No text, no logos, no watermarks. No morphing faces, no melting objects."
)

# The operator. Locked off AI shots read as AI, so every shot gets a human on the camera.
CAMERA_OP = (
    "Shot handheld by a camera operator: gentle breathing float, small drifting reframes, "
    "natural motion blur on movement, subtle focus breathing. Real 35mm film texture, natural light. "
    "The camera holds this framing, one single continuous take, no big zooms, no pull backs, no cuts to another place. "
    "Same characters, same wardrobe, same location and same view out of the windows as the first frame. " + RULES
)

DEFAULT_SCENE = "She pours coffee from the moka pot into her mug, takes a first sip and smiles to herself before heading out."

# Puts the person from WHO into the place from WHERE. The scene line is added after it.
HERO_TEXT = (
    "Put the person from image 2 into the place in image 1, full body in the frame, naturally lit by the light "
    "of the place. Keep the place exactly the same. Same face, hair and wardrobe as image 2, the person appears "
    "only once. Real candid photo, not posed, not looking at the camera. " + RULES + " In this moment:"
)
HERO_LORA = "Qwen-Image-Edit-2511_Consistency.safetensors"

# Tells the angle model to keep the hero's people and check the place against WHERE.
REFERENCE_TEXT = (
    "Keep the people in image 1 exactly the same (same faces, hair, wardrobe, same number of people). "
    "Keep the place exactly as in image 2 (same walls, furniture, windows, street). Keep the light of image 1. " + RULES
)

SECONDS = 3
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
    def photo(label, title, y):
        return g.new("LoadImage", [0, y], [title.lower() + ".png", "image"],
                     inputs=[{**widget_input("image", "COMBO"), "label": label}],
                     outputs=[{"name": "IMAGE", "type": "IMAGE"}, {"name": "MASK", "type": "MASK"}],
                     size=[340, 360], title=title)

    who = photo("who (a photo of the person)", "WHO", 0)
    where = photo("where (a photo of the place)", "WHERE", 420)
    scene = g.new("PrimitiveStringMultiline", [0, 840], [DEFAULT_SCENE],
                  inputs=[{**widget_input("value", "STRING"), "label": "what happens"}],
                  outputs=[{"name": "STRING", "type": "STRING"}], size=[340, 180],
                  title="WHAT HAPPENS")

    # Step 1: put the person in the place, doing the thing. This is the hero frame.
    hero_prompt = g.new("StringConcatenate", [420, -900], [HERO_TEXT, "", " "],
                        inputs=[widget_input("string_a", "STRING"), widget_input("string_b", "STRING"),
                                widget_input("delimiter", "STRING")],
                        outputs=[{"name": "STRING", "type": "STRING"}], size=[340, 200], title="HERO PROMPT")
    g.link(scene, 0, hero_prompt, "string_b")
    hero = g.add(qwen_inst, [880, -900])
    hero["title"] = "MAKE THE HERO FRAME"
    hero["widgets_values"][6] = HERO_LORA
    hero["widgets_values"][-1] = 777
    hero["widgets_values_named"].update({"lora_name_1": HERO_LORA, "seed": 777})
    g.link(where, 0, hero, "image")
    g.link(who, 0, hero, "image2")
    g.link(hero_prompt, 0, hero, "prompt")
    hero_still = g.new("SaveImage", [1320, -900], ["coverage/stills/00_HERO"],
                       inputs=[{"name": "images", "type": "IMAGE"}], size=[340, 300], title="HERO FRAME")
    g.link(hero, 0, hero_still, "images")

    res = g.add(res_t, [0, 1440])
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
        angle_prompt = g.new("StringConcatenate", [880, y + 700], ["", REFERENCE_TEXT, " "],
                             inputs=[widget_input("string_a", "STRING"), widget_input("string_b", "STRING"),
                                     widget_input("delimiter", "STRING")],
                             outputs=[{"name": "STRING", "type": "STRING"}], size=[340, 160],
                             title=s["name"] + " match the place")
        g.link(cam, 1, angle_prompt, "string_a")
        g.link(hero, 0, angle, "image")
        # The person is already in the hero frame; passing their photo again makes clones.
        g.link(where, 0, angle, "image2")
        g.link(angle_prompt, 0, angle, "prompt")

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
        wv[0], wv[1], wv[2], wv[5], wv[6] = "", True, SECONDS, 2000 + k, FPS  # prompt enhance looks at each frame
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
                "inputs": [[who["id"], "image"], [where["id"], "image"], [scene["id"], "value"]],
                "outputs": [scene_out["id"], hero_still["id"]] + shot_saves,
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
