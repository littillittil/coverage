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
    {"name": "04 PROFILE", "h": 60, "v": 0, "zoom": 6, "text": "Side profile medium shot."},
    {"name": "05 LOW ANGLE", "h": 330, "v": -25, "zoom": 6, "text": "Low angle hero shot."},
]

# The rule book: what must never happen. These models run at CFG 1 (fast distilled), where a negative
# prompt is ignored, so the rules are written into every positive prompt instead.
RULES = (
    "Rules: every person appears only once, no duplicates, no twins, no extra people. "
    "Natural human anatomy, two hands with five fingers, true to life scale between people and objects. "
    # Never name an object you do not want: "phones held the right way round" made the video model add phones.
    "Hands stay relaxed and empty unless the moment needs them. "
    "No text, no logos, no watermarks. No morphing faces, no melting objects."
)

# The operator. Locked off AI shots read as AI, so every shot gets a human on the camera, shooting on an iPhone.
CAMERA_OP = (
    "Shot on an iPhone held in someone's hands: gentle breathing float, small drifting reframes, "
    "natural motion blur on movement, the focus softly shifting between subject and background, "
    "auto exposure gently breathing. Real iPhone video look, natural light, no cinematic grade. "
    "The camera holds this framing, one single continuous take, no big zooms, no pull backs, no cuts to another place. "
    "Same characters, same wardrobe, same location and same view out of the windows as the first frame. " + RULES
)


# FLUX 2 builds the place from the situation. The situation is added after this.
PLACE_TEXT = (
    "Straight out of camera iPhone photo of a real place, completely empty of people, wide view at eye level, "
    "natural light, real textures, lived in, no text, no logos, no watermarks. "
    "The place where this happens (show only the place, no person):"
)
# The situation menu, written as "place | moment". FLUX 2 only ever sees the place (if it sees the moment it
# draws a stranger into the room); the hero frame gets the moment; the shots get both.
# Simple lifestyle moments with the face visible work best; prop actions (pouring coffee) and
# facing away (looking out a window) go wrong.
SITUATIONS = [
    "A sunny East Village walk up apartment with a red brick wall and a green velvet couch | relaxing on the couch, smiling to themselves",
    "A leafy Brooklyn street of brownstones at golden hour | walking slowly down the sidewalk, glancing around",
    "A path through Central Park in autumn with old benches | sitting on a bench, watching people go by, thoughtful",
    "A Manhattan rooftop at sunset with the skyline behind | leaning on the railing, the wind in their hair, calm",
    "A neon lit Hong Kong diner at night with mint green booths | sitting in a booth, laughing softly",
    "A Paris cafe by a rainy window | sitting at a table by the window, watching the street with a soft smile",
]

# The hero frame: the person's photo (image 1) seen from further away, inside the empty place FLUX built
# (image 2). It runs through the multiple angles LoRA with a pulled back CameraAngle. That pull back is
# what moves a close up selfie into a room; a plain edit keeps the selfie's own wall and framing.
HERO_CAMERA = {"h": 15, "v": 5, "zoom": 2.5}
HERO_TEXT = (
    "The same person from image 1, now seen from further away, inside the place from image 2. The whole "
    "background is the place from image 2, nothing of the plain wall in image 1 remains. Keep exactly the same "
    "face shape, skin tone, eyes, hair, facial hair and headwear, the same clothes; continue the clothes "
    "naturally if only the head and shoulders are visible. The face is clearly visible, turned about three "
    "quarters toward the camera, not looking into the lens. Real candid photo, not posed. " + RULES + " In this moment:"
)
# Step 0: most people upload a selfie, not a full body photo. This step shows the same person head to toe
# against a plain wall, keeping their face and top and completing the outfit, so every later step has a
# whole person to work with (head only photos gave near identical, cropped angles).
FULL_BODY_TEXT = (
    "Show this exact same person from head to toe, standing straight against a plain white wall, the whole body "
    "in the frame with space above the head and below the shoes, arms relaxed at their sides, hands empty, facing "
    "the camera. Keep exactly the same face, face shape, skin tone, eyes, hair, facial hair and headwear, and the "
    "same top they are wearing. Complete the outfit with trousers and shoes that naturally match their style. "
    "Straight out of camera iPhone photo, soft natural window light, real skin texture. Only one person."
)
FULL_BODY_LORA = "Qwen-Image-Edit-2511_Consistency.safetensors"

ANGLE_UNET = "qwen_image_edit_2511_bf16.safetensors"

# Tells the angle model to keep the hero's people and check the place against WHERE.
REFERENCE_TEXT = (
    "Keep the people in image 1 exactly the same: the same face shape, skin tone, eyes, hair and headwear, "
    "the same wardrobe, the same number of people. "
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
    flux_t = load("image_flux2_text_to_image.json")
    by_id = lambda t, i: next(n for n in t["nodes"] if n["id"] == i)
    qwen_inst, cam_t = by_id(angles_t, 219), by_id(angles_t, 221)
    ltx_inst, res_t = by_id(ltx_t, 398), by_id(ltx_t, 403)
    flux_inst = by_id(flux_t, 98)

    g = Graph()
    def photo(label, title, y):
        return g.new("LoadImage", [0, y], [title.lower() + ".png", "image"],
                     inputs=[{**widget_input("image", "COMBO"), "label": label}],
                     outputs=[{"name": "IMAGE", "type": "IMAGE"}, {"name": "MASK", "type": "MASK"}],
                     size=[340, 360], title=title)

    who = photo("Your photo (a selfie is enough)", "WHO", 0)
    situation = g.new("CustomCombo", [-900, 0], [SITUATIONS[0], 0] + SITUATIONS + [""],
                      inputs=[{**widget_input("choice", "COMBO"), "label": "Situation"}],
                      outputs=[{"name": "STRING", "type": "STRING"}, {"name": "INDEX", "type": "INT"}],
                      size=[420, 400], title="SITUATION")

    # Step 0: FLUX 2 builds the place from one line.
    place_prompt = g.new("StringConcatenate", [-900, 700], [PLACE_TEXT, "", " "],
                         inputs=[widget_input("string_a", "STRING"), widget_input("string_b", "STRING"),
                                 widget_input("delimiter", "STRING")],
                         outputs=[{"name": "STRING", "type": "STRING"}], size=[340, 200], title="PLACE PROMPT")
    def regex(title, pattern, y):
        node = g.new("RegexReplace", [-900, y], ["", pattern, ""],
                     inputs=[widget_input("string", "STRING"), widget_input("regex_pattern", "STRING"),
                             widget_input("replace", "STRING")],
                     outputs=[{"name": "STRING", "type": "STRING"}], size=[320, 160], title=title)
        g.link(situation, 0, node, "string")
        return node

    just_place = regex("JUST THE PLACE", r"\s*\|.*$", 500)
    just_moment = regex("JUST THE MOMENT", r"^.*?\|\s*", 800)
    g.link(just_place, 0, place_prompt, "string_b")
    where = g.add(flux_inst, [-450, 420])
    where["title"] = "MAKE THE PLACE"
    where["widgets_values"] = ["", 1344, 768, "flux2_dev_fp8mixed.safetensors", "mistral_3_small_flux2_bf16.safetensors",
                               "full_encoder_small_decoder.safetensors", "Flux_2-Turbo-LoRA_comfyui.safetensors", True, 4242]
    g.link(place_prompt, 0, where, "text")
    place_still = g.new("SaveImage", [-450, 900], ["coverage/stills/00_PLACE"],
                        inputs=[{"name": "images", "type": "IMAGE"}], size=[340, 300], title="THE PLACE")
    g.link(where, 0, place_still, "images")
    # Step 1: put the person in the place, doing the thing. This is the hero frame.
    hero_prompt = g.new("StringConcatenate", [420, -900], [HERO_TEXT, "", " "],
                        inputs=[widget_input("string_a", "STRING"), widget_input("string_b", "STRING"),
                                widget_input("delimiter", "STRING")],
                        outputs=[{"name": "STRING", "type": "STRING"}], size=[340, 200], title="HERO PROMPT")
    g.link(just_moment, 0, hero_prompt, "string_b")
    hero_cam = g.add(cam_t, [0, -1300])
    hero_cam["title"] = "HERO CAMERA"
    hero_cam["widgets_values"] = [HERO_CAMERA["h"], HERO_CAMERA["v"], HERO_CAMERA["zoom"], "camera"]
    hero_cam["widgets_values_named"] = {"horizontal_angle": HERO_CAMERA["h"], "vertical_angle": HERO_CAMERA["v"],
                                        "zoom": HERO_CAMERA["zoom"], "view": "camera"}
    # The edit keeps image 1's shape, so the person's photo is padded to widescreen first.
    widescreen = g.new("ResizeAndPadImage", [-450, -1300], [1344, 768, "white", "area"],
                       inputs=[{"name": "image", "type": "IMAGE"}, widget_input("target_width", "INT"),
                               widget_input("target_height", "INT"), widget_input("padding_color", "COMBO"),
                               widget_input("interpolation", "COMBO")],
                       outputs=[{"name": "IMAGE", "type": "IMAGE"}], size=[320, 160], title="WIDESCREEN THE PHOTO")
    full_body = g.add(qwen_inst, [-900, -1700])
    full_body["title"] = "MAKE THE FULL BODY"
    wv = full_body["widgets_values"]
    wv[0], wv[1], wv[2], wv[6], wv[-1] = FULL_BODY_TEXT, False, ANGLE_UNET, FULL_BODY_LORA, 555
    full_body["widgets_values_named"].update({"prompt": FULL_BODY_TEXT, "value": False, "unet_name": ANGLE_UNET,
                                              "lora_name_1": FULL_BODY_LORA, "seed": 555})
    g.link(who, 0, full_body, "image")
    full_body_still = g.new("SaveImage", [-450, -1700], ["coverage/stills/00_FULL_BODY"],
                            inputs=[{"name": "images", "type": "IMAGE"}], size=[340, 300], title="FULL BODY")
    g.link(full_body, 0, full_body_still, "images")
    g.link(full_body, 0, widescreen, "image")
    g.link(widescreen, 0, hero_cam, "image")
    hero_full = g.new("StringConcatenate", [420, -1300], ["", "", " "],
                      inputs=[widget_input("string_a", "STRING"), widget_input("string_b", "STRING"),
                              widget_input("delimiter", "STRING")],
                      outputs=[{"name": "STRING", "type": "STRING"}], size=[340, 160], title="HERO CAMERA + PROMPT")
    g.link(hero_cam, 1, hero_full, "string_a")
    g.link(hero_prompt, 0, hero_full, "string_b")
    hero = g.add(qwen_inst, [880, -900])
    hero["title"] = "MAKE THE HERO FRAME"
    hero["widgets_values"][-1] = 777
    hero["widgets_values_named"].update({"seed": 777})
    g.link(widescreen, 0, hero, "image")
    g.link(where, 0, hero, "image2")
    hero["widgets_values"][2] = ANGLE_UNET
    hero["widgets_values_named"]["unet_name"] = ANGLE_UNET
    g.link(hero_full, 0, hero, "prompt")
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
        # Full size weights hold faces better across new camera positions than the int8 file.
        angle["widgets_values"][2] = ANGLE_UNET
        angle["widgets_values_named"]["unet_name"] = ANGLE_UNET
        # Speed mode (4 step Lightning) smooths skin into plastic; full steps keep real texture.
        angle["widgets_values"][1] = False
        angle["widgets_values_named"]["value"] = False
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
        g.link(situation, 0, prompt, "string_a")

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

    subgraphs = (angles_t["definitions"]["subgraphs"] + ltx_t["definitions"]["subgraphs"]
                 + flux_t["definitions"]["subgraphs"])
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
                "inputs": [[who["id"], "image"], [situation["id"], "choice"]],
                "outputs": [scene_out["id"], full_body_still["id"], place_still["id"], hero_still["id"]] + shot_saves,
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
