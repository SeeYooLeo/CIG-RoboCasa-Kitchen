# Homepage figures and videos

[Project overview](../../README.md) · [Paper and implementation notes](../paper.md)

The homepage now uses original figures from the supplied manuscript. Their image content is preserved; filenames describe their role in the release.

| Homepage section | Asset | Manuscript source |
|---|---|---|
| Method | `docs/media/method.png` | Method overview, updated `fig2.png` |
| robocasa-kitchen evaluation | `docs/media/results.png` | PnPMicrowaveToCounter trajectory comparison, original `fig3.png` |
| Stretch 3 experiments | `docs/media/real_robot_rollouts.png` | Main-text real-world rollout figure, original `real_robot.png` |
| Counter-to-rack video | `docs/media/counter-to-rack-placement.mp4`, `docs/media/counter-to-rack-placement.jpg` | Supplied autonomous demonstration and poster |
| Washer-to-basin video | `docs/media/washer-to-basin-retrieval.mp4`, `docs/media/washer-to-basin-retrieval.jpg` | Supplied autonomous demonstration and poster |
| Faucet video | `docs/media/faucet-operation.mp4`, `docs/media/faucet-operation.jpg` | Supplied autonomous demonstration and poster |
| VR video | `docs/media/vr-data-collection.mp4`, `docs/media/vr-data-collection.jpg` | 27-second headset recording with a face mosaic around 20 s, and poster |

To replace a figure, save the final image here and update its `src`, alternative text and caption in the root README.

## Real-robot video

The homepage includes three supplied videos, preserved without re-encoding or trimming: [counter-to-rack placement](counter-to-rack-placement.mp4), [washer-to-basin retrieval](washer-to-basin-retrieval.mp4) and [faucet operation](faucet-operation.mp4). Each retains its original “Autonomous 5x” annotation. Posters are extracted from the corresponding videos.

## VR video

The homepage embeds a GitHub-hosted video attachment with native playback controls. The recording is trimmed to 27 seconds at 1920 × 1080, retains its original 5× speed, and shows headset-based teleoperation for data collection.

A mosaic follows the face visible around 20 seconds. No other regions are masked. The poster is extracted from the released video. The untrimmed VR original and intermediate review frames remain outside the release directory.

## Paper metadata

The final homepage section contains a BibTeX entry using the camera-ready title, year and author list. Add the proceedings identifier when it is available. Add paper/project links near the title once their destinations exist.

The method figure shows the two experimental families: CR (parallel continuous-action regression, instantiated by OpenVLA-OFT) and FM (instantiated by π0.5). Its labels, parallel action arrows and CR score match the manuscript.

## GitHub playback

The homepage uses native GitHub video attachments, with playback, seeking and fullscreen controls directly in the README. The three policy-rollout attachments preserve the released MP4 files. The VR attachment is a 1280 × 720 H.264 copy under 10 MB, retaining the complete duration, face mosaic and absence of audio tracks. The original 1920 × 1080 VR file remains available in this directory.

Attachments are retained in the repository's [demonstration video assets](https://github.com/SeeYooLeo/CIG-RoboCasa-Kitchen/issues/1). The [standalone video gallery](https://seeyooleo.github.io/CIG-RoboCasa-Kitchen/index.html) also plays the original MP4 files.
