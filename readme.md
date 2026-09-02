# Agent Playground

## Questions to explore

- Have you explored other agent harnesses, multi-agent systems, environment design, or agent evals yet?
- Do you have a bit of understanding or experience working with image/video generation models, video understanding, or computer vision?

## Agent harnesses
1. Use systems to scan barcode.
2. AI reads the number, but the number comes from matching digits against labels.
3. If higher or equal, keep the change. If lower, change the code.
[ Repeat step 1, 2, 3 again and again. ]

```
cd barcode-scanner && source .venv/bin/activate                                                                                      
export ANTHROPIC_API_KEY=sk-ant-...                                                                                                  
python run_eval.py --agents claude
```

## Multi-agent system
1. Book may have a lot of chapters.
2. Take all the chapters out.
3. Put back all the chapter with subject and description.

a. Spliting is 1 agent.
b. Describing are another agents. (Not one by one. Parallel to improve the speed)
c. Checking is another agent.
d. Combining is another agent.

```
cd video-crew && source .venv/bin/activate                                                                                           
python summarise.py data/clip1/video.mp4 --out runs/clip1
```

## Environment design

The file is environment.py

To make sure every run is fair. No hidden data left from pass run.

## Agent evals

The file is scorer.py

Give grading after every run.
Did the reader improves on hidden pictures, tests passed, file in scope

[Note: If the agent can see every exam question, it can pass without learning anything. So hidden is important here.]
[Note: Test passed means every run succeed]
[Note: File in scope: To prevent cheating by adding or removing file manually]

## Computer vision
Done. Explain in barcode system above

## Video understanding
Split the video into chunks that each mean one thing.

1. Summarise each frame as a colour histogram (histogram = count of colours, ignoring where they are).
2. Compare with the previous frame.
3. Call it a cut if the distance jumps above 0.5. (This means the colours changed a lot.)
4. Each cut starts a new scene. A scene is all the frames between two cuts.

```
cd video-crew
source .venv/bin/activate
python summarise.py data/clip1/video.mp4 --out runs/clip1
```

## Image generation
Uses sd-turbo, a fast version of Stable Diffusion. It is a model that paints a picture from a text description.

Here the text comes from the video summary. The video goes in, a poster comes out.

```
cd video-crew
source .venv/bin/activate
pip install torch diffusers transformers accelerate
python summarise.py data/clip1/video.mp4 --out runs/clip1_sd --diffusion
```

The poster is saved at `runs/clip1_sd/poster.png`. First run downloads the model, about 2.5 GB.
Without `--diffusion`, it makes a grid of keyframes instead. No model needed.
