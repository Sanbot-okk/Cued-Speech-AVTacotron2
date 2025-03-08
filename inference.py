import subprocess

script_synthese = "do_syn.py"
script_downsample = "do_inverse_sample.py"

# Run the first script to synthesize audio and visuals
subprocess.run(["python", script_synthese], check=True)
print("Mel and visual features have been synthesized")

# Run the second script to readjust the sampling for visuals
subprocess.run(["python", script_downsample], check=True)
print("Visual features have been resampled to original rate for comparison. \n Run Visualize.ipynb to visualize.")