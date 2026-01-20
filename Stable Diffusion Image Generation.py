import os
import torch
import time
import matplotlib.pyplot as plt

from diffusers import StableDiffusionXLPipeline

def generate_solar_panel_fault_images(
    num_images=50, 
    output_dir="generated_solar_faults",
    model_path="C:/path/to/stable-diffusion-xl-model"
):
    """
    Generate synthetic solar panel fault images using Stable Diffusion XL.
    Images are saved to the specified output_dir.
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Start timing and monitoring
    overall_start = time.time()
    if torch.cuda.is_available():
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        torch.cuda.reset_peak_memory_stats()

    # Load Stable Diffusion XL model (edit path as needed)
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")
    pipeline.enable_attention_slicing()

    base_prompt = "A solar panel with {fault_type}, under {weather_condition}"
    fault_types = [
        "severe dust accumulation",
        "moss growth and heavy dust efficiency",
        "shattered solar cells with dark spots",
        "a very few covered with dirt", 
        "bird droppings causing hot spots"
    ]
    weather_conditions = [
        "clear sky",
        "overcast condition",
        "wind with sand particles"
    ]

    # Image generation with timing per image
    per_image_times = []
    gen_start = time.time()
    for i in range(num_images):
        image_start = time.time()
        fault = fault_types[i % len(fault_types)]
        weather = weather_conditions[i % len(weather_conditions)]
        prompt = base_prompt.format(fault_type=fault, weather_condition=weather)
        output_path = os.path.join(output_dir, f"solar_fault_{i+1}.png")
        generated_image = pipeline(prompt=prompt, num_inference_steps=30).images[0]
        generated_image.save(output_path)
        image_end = time.time()
        per_image_time = image_end - image_start
        per_image_times.append(per_image_time)
        print(f"Generated image {i+1} saved at {output_path} - Time: {per_image_time:.2f} seconds")
        # Optionally display last image
        if i == num_images - 1:
            plt.imshow(generated_image)
            plt.axis('off')
            plt.show()
    gen_end = time.time()

    # Report stats
    overall_end = time.time()
    print(f"\nTotal generation time for {num_images} images: {gen_end - gen_start:.2f} seconds")
    print(f"Average time per image: {(gen_end - gen_start)/num_images:.2f} seconds")
    print(f"Total end-to-end execution time: {overall_end - overall_start:.2f} seconds")
    print(f"Fastest image: {min(per_image_times):.2f}s, Slowest image: {max(per_image_times):.2f}s")

    if torch.cuda.is_available():
        print(f"Peak GPU memory used: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
        print(f"Current GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    try:
        import psutil
        process = psutil.Process()
        print(f"Peak RAM usage: {process.memory_info().rss / (1024**2):.2f} MB")
    except ImportError:
        print("psutil not installed. Skipping RAM usage stats.")

# --- EDIT model_path to point to your local Stable Diffusion XL weights ---
generate_solar_panel_fault_images(
    num_images=162,
    output_dir="generated_solar_faults",
    model_path="C:/path/to/stable-diffusion-xl-model"  # Change this to your SDXL model path
)
