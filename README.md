# Coverage

**One frame in. A whole scene out. Same face, same light, same room.**

On a film set, the director never shoots just one angle. The crew shoots *coverage*: a wide, a medium, a close up, a reverse, a low angle. That is what lets an editor cut a scene.

AI video gives you one shot at a time, and every new shot drifts: a new face, new light, a different room. Coverage fixes that. You drop in one hero frame, write one line about what happens, and press Run. You get back five matching shots, each one moving like a real operator is holding the camera, already cut together into a scene.

Built for the ComfyUI Open Source Higgsfield competition. Open weights only, with LTX as the video model.

![coverage board](examples/board.jpg)

## How it works

```
hero frame ──► CameraAngle ×5 ──► Qwen Image Edit 2511 + multiple angles LoRA ──► 5 matching frames
                                                                                      │
"what happens" ──► + shot type + camera operator direction ──────────────────────────┤
                                                                                      ▼
                                                     LTX 2.5 image to video ×5 ──► 5 handheld shots
                                                                                      │
                                                                ConcatenateVideo ──► the scene
```

1. **Angles.** Five `CameraAngle` nodes place a virtual camera around the subject (wide, medium, close up, reverse, low angle). Qwen Image Edit 2511 with the multiple angles LoRA redraws the hero frame from each position. Every angle is made from the same hero frame, so the character, wardrobe, light and location stay locked.
2. **The operator.** Locked off AI shots read as AI. Each shot's prompt adds the shot type plus a camera operator direction: breathing float, small drifting reframes, motion blur, focus breathing.
3. **Motion.** LTX 2.5 animates each frame into a 3 second shot with sound.
4. **The cut.** `ConcatenateVideo` joins the five shots into one scene, ready to post.

It runs in Comfy's **App mode**, so the person using it only sees two controls: the hero frame and what happens.

## Run it

**Comfy Cloud (fastest):** open [cloud.comfy.org](https://cloud.comfy.org), drag `workflows/coverage.app.json` onto the canvas, add your hero frame, describe the scene, press Run. Every model is already on Comfy Cloud.

**Your own ComfyUI:** update ComfyUI to the latest version (it needs the `CameraAngle`, `ResolutionSelector` and `ConcatenateVideo` core nodes), drag in `workflows/coverage.app.json`, and download the models below into the matching folders.

| Folder | Model |
|---|---|
| `diffusion_models` | `qwen_image_edit_2511_int8_convrot.safetensors` |
| `diffusion_models` | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` |
| `text_encoders` | `qwen_2.5_vl_7b_fp8_scaled.safetensors` |
| `text_encoders` | `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` |
| `text_encoders` | `gemma4_e2b_it_int8_convrot.safetensors` |
| `vae` | `qwen_image_vae.safetensors` |
| `vae` | `ltx-2.5-video-vae-bf16.safetensors`, `ltx-2.5-audio-vae-bf16.safetensors` |
| `loras` | `qwen-image-edit-2511-multiple-angles-lora.safetensors` |
| `loras` | `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` |
| `latent_upscale_models` | `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` |

All of these are open weights. The official Comfy templates `template_qwen_multiangle.app` and `video_ltx2_5_i2v` list the download links.

## Change the coverage

The shot list lives at the top of `tools/build_coverage.py`:

```python
SHOTS = [
    {"name": "01 WIDE", "h": 0, "v": 12, "zoom": 0.5, "text": "Wide establishing shot."},
    ...
]
```

`h` is the camera's position around the subject (0 front, 90 right, 180 behind), `v` its height (negative looks up), `zoom` goes from 0 wide to 10 close. Edit it, then rebuild the workflow:

```bash
python3 tools/build_coverage.py
```

The builder reads the two official templates in `templates/` and writes `workflows/coverage.app.json`.

## What's new here

Most image to video apps make one shot. Coverage makes a **scene**: it thinks like a camera crew, plans the shot list, keeps every shot matched to one hero frame, and gives every shot a human camera operator. It is one button for the thing every AI filmmaker spends hours fixing by hand.

## License

MIT. See [LICENSE](LICENSE).

Made by Littil Swayamp, cinematographer and director, Los Angeles.
