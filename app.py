import os
import sys
import torch
import gradio as gr
from PIL import Image

# ideogram4 모듈을 찾을 수 있도록 src 디렉터리를 sys.path에 추가합니다.
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from ideogram4 import (
    DEFAULT_MAGIC_PROMPT,
    MAGIC_PROMPTS,
    PRESETS,
    Ideogram4Pipeline,
    Ideogram4PipelineConfig,
    aspect_ratio_from_size,
)

# 0번 GPU의 사용 가능한 VRAM의 50%(0.5)만 사용하도록 한도를 설정합니다.
if torch.cuda.is_available():
    try:
        torch.cuda.set_per_process_memory_fraction(0.5, 0)
        print("GPU 0 VRAM limit set to 50% (0.5 fraction) successfully.")
    except Exception as e:
        print(f"Failed to set VRAM limit on GPU 0: {e}")
else:
    print("CUDA is not available. GPU VRAM limit was not set.")

# 싱글톤 패턴으로 모델 파이프라인을 유지합니다.
pipeline = None
current_quantization = None

def get_pipeline(quantization="nf4", device="cuda"):
    global pipeline, current_quantization
    if pipeline is None or current_quantization != quantization:
        repo = os.environ.get("WEIGHTS_REPO")
        if not repo:
            repo = "ideogram-ai/ideogram-4-nf4" if quantization == "nf4" else "ideogram-ai/ideogram-4-fp8"
        print(f"Loading Ideogram 4 pipeline from '{repo}' on {device}...")
        pipeline = Ideogram4Pipeline.from_pretrained(
            config=Ideogram4PipelineConfig(weights_repo=repo),
            device=device,
            dtype=torch.bfloat16,
        )
        current_quantization = quantization
    return pipeline

def generate_image(
    prompt,
    width,
    height,
    sampler_preset,
    seed,
    quantization,
    magic_prompt,
    magic_prompt_model,
    magic_prompt_key,
    warn_on_caption_issues
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Magic Prompt API Key 및 변환 처리
    if magic_prompt:
        if not magic_prompt_key:
            return None, "Error: Magic prompt is enabled but API key is missing. Please enter an API key or disable Magic Prompt."
        aspect_ratio = aspect_ratio_from_size(width, height)
        try:
            magic = MAGIC_PROMPTS[magic_prompt_model](api_key=magic_prompt_key)
            prompt = magic.expand(prompt, aspect_ratio=aspect_ratio)
            print(f"Expanded Caption (Magic Prompt):\n{prompt}")
        except Exception as e:
            return None, f"Error expanding prompt: {str(e)}"
    
    try:
        pipe = get_pipeline(quantization, device)
        preset = PRESETS[sampler_preset]
        
        # 난수 시드 처리
        if seed is None or seed < 0:
            seed = 0
            
        images = pipe(
            prompt,
            height=int(height),
            width=int(width),
            num_steps=preset.num_steps,
            guidance_schedule=preset.guidance_schedule,
            mu=preset.mu,
            std=preset.std,
            seed=int(seed),
            raise_on_caption_issues=not warn_on_caption_issues,
        )
        return images[0], f"Image generated successfully! (Seed: {seed})"
    except Exception as e:
        import traceback
        error_msg = f"Error generating image: {str(e)}\n\nDetailed Traceback:\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        return None, error_msg

# Gradio 테마 및 인터페이스 설정
with gr.Blocks(title="Ideogram 4 WebUI") as demo:
    gr.Markdown("# 🎨 Ideogram 4 로컬 이미지 생성 WebUI")
    gr.Markdown(
        "Ideogram 4의 로컬 가중치 모델을 활용하여 웹 인터페이스에서 이미지를 생성합니다. "
        "서버의 0번 GPU에 VRAM 0.5 할당량 제한을 적용하여 동작합니다."
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt (프롬프트)",
                placeholder="영문으로 생성하고자 하는 이미지 묘사를 입력하세요...",
                lines=4
            )
            
            with gr.Row():
                width = gr.Slider(minimum=256, maximum=2048, step=16, value=1024, label="Width (가로 크기)")
                height = gr.Slider(minimum=256, maximum=2048, step=16, value=1024, label="Height (세로 크기)")
            
            with gr.Row():
                sampler_preset = gr.Dropdown(
                    choices=sorted(PRESETS.keys()),
                    value="V4_QUALITY_48",
                    label="Sampler Preset (샘플러 프리셋)"
                )
                quantization = gr.Dropdown(
                    choices=["nf4", "fp8"],
                    value="nf4",
                    label="Quantization (양자화)"
                )
            
            seed = gr.Number(value=0, label="Seed (시드 번호)", precision=0)
            warn_on_caption_issues = gr.Checkbox(
                label="Warn on caption issues (캡션 이슈 발생 시 에러 대신 경고만 표시)", 
                value=True
            )
            
            with gr.Group():
                magic_prompt = gr.Checkbox(
                    label="Enable Magic Prompt (매직 프롬프트 활성화 - API Key 필요)", 
                    value=False
                )
                magic_prompt_model = gr.Dropdown(
                    choices=sorted(MAGIC_PROMPTS.keys()),
                    value=DEFAULT_MAGIC_PROMPT,
                    label="Magic Prompt Model (매직 프롬프트 모델)"
                )
                magic_prompt_key = gr.Textbox(
                    label="Magic Prompt / Ideogram API Key",
                    placeholder="API Key가 없으면 'Enable Magic Prompt'를 해제하고 일반 프롬프트로 사용하세요.",
                    type="password"
                )
                
            generate_btn = gr.Button("이미지 생성 (Generate)", variant="primary")
            
        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image (생성된 이미지)")
            status_text = gr.Textbox(label="Status / Log (상태 및 로그)", interactive=False)
            
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt,
            width,
            height,
            sampler_preset,
            seed,
            quantization,
            magic_prompt,
            magic_prompt_model,
            magic_prompt_key,
            warn_on_caption_issues
        ],
        outputs=[output_image, status_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
