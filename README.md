# Coverage

**One photo. Pick a situation. Get the whole scene, shot like a film crew.**

On a film set, the director never shoots just one angle. The crew shoots *coverage*: a wide, a medium, a close up, a profile, a low angle. That is what lets an editor cut a scene.

AI video gives you one shot at a time, and every new shot drifts: a new face, new light, a different room. Coverage fixes that. Give it one photo of a person (a selfie is enough), pick a situation, press Run. It builds the place, puts the person in it, plans five camera positions, keeps every shot matched, gives each one a human camera operator, and cuts them into a scene.

Built for the ComfyUI Open Source Higgsfield competition by a working cinematographer. **Every model is open weights.** LTX is the video model.

## How it works

```
your photo ─────────────────────────────┐
                                         ▼
situation ──► FLUX 2: build the place ──► Qwen Edit 2511: put the person in the place (the hero frame)
                     │                                   │
                     │          CameraAngle ×5 ──► Qwen Edit 2511 + multiple angles LoRA ──► 5 matching frames
                     └──────────── the place keeps every angle honest ──┘                       │
                                                                                                 ▼
                           situation + shot type + camera operator + rule book ──► LTX 2.5 ×5 ──► 5 handheld shots
                                                                                                 ▼
                                                                         ConcatenateVideo ──► the scene
```

| Step | Job | Model (all open weights) |
|---|---|---|
| 0 | Build the place from the situation | FLUX 2 dev |
| 1 | Put the person in the place: the hero frame | Qwen Image Edit 2511 + Consistency LoRA |
| 2 | Five camera angles, each checked against the place | Qwen Image Edit 2511 + multiple angles LoRA |
| 3 | Motion prompt written from what is in each frame | Gemma 4 (LTX prompt helper) |
| 4 | Five 3 second handheld shots with sound | LTX 2.5 |
| 5 | The cut | ConcatenateVideo |

**The camera operator.** Locked off AI shots read as AI. Every shot is told: handheld, breathing float, small drifting reframes, motion blur, focus breathing, one continuous take.

**The rule book.** These fast models ignore negative prompts, so the never do rules are written into every prompt: each person appears once, natural anatomy and scale, phones and papers held the right way round, no text, no morphing. Every failure found in testing became a rule.

**App mode.** The person using it sees two controls: a photo and a situation menu.

## Run it

**Comfy Cloud (fastest):** open [cloud.comfy.org](https://cloud.comfy.org), drag `workflows/coverage.app.json` onto the page, add a photo, pick a situation, press Run. Every model is already on Comfy Cloud.

**Your own ComfyUI:** update to the latest ComfyUI (it uses the core `CameraAngle`, `CustomCombo`, `RegexReplace`, `ResolutionSelector` and `ConcatenateVideo` nodes) and install one custom node pack from the Manager, **ComfyUI-Image-Filters** (for `BetterFilmGrain`), drag in the workflow, and download the models listed in the official templates `image_flux2_text_to_image`, `template_qwen_multiangle.app` and `video_ltx2_5_i2v` (Comfy's template browser links every file).

**No photo handy?** Try one of the sample people in `examples/people`: Maya, Kenji, Priya and Mateo. They were made with FLUX 2 inside ComfyUI, so they are open weights too. Each is a plain selfie, because that is all the app needs.

## Make it yours

Everything lives at the top of `tools/build_coverage.py`:
- `SITUATIONS`: the menu. Add a line, get a new situation.
- `SHOTS`: the shot list. `h` is the camera's position around the subject (0 front, 90 side), `v` its height (negative looks up), `zoom` from 0 wide to 10 close.
- `CAMERA_OP`: the operator's direction.
- `RULES`: the rule book.

Rebuild the workflow after an edit:

```bash
python3 tools/build_coverage.py
```

The builder reads the official Comfy templates in `templates/` and writes `workflows/coverage.app.json`.

## What's new here

Most image to video apps make one shot. Coverage makes a **scene**. It thinks like a camera crew: it plans the shot list, keeps every shot matched to one hero frame and one place, and gives every shot a human camera operator. One photo and one tap for the thing every AI filmmaker spends hours fixing by hand.

## What's next

Swipe to keep or reject shots (keepers cut the edit, rejects grow the rule book) · your voice drives the performance with LTX audio to video · two people per scene · story mode across shots · music and looks.

## License

MIT. See [LICENSE](LICENSE).

Made by Littil Swayamp, cinematographer and director, Los Angeles.
