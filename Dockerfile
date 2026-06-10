FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 비대화형 모드 설정
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# 시스템 라이브러리 업데이트 및 Python 3.10 + 빌드 도구 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && pip install --no-cache-dir --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# 종속성 파일 및 소스 디렉터리 복사
COPY pyproject.toml LICENSE.md README.md ./
COPY src/ ./src/

# PyTorch (CUDA 12.1) 및 프로젝트 종속성 설치
# pip이 torch>=2.11 요구사항을 만족하는 최신 호환 버전을 자동으로 해결합니다.
RUN pip install --no-cache-dir -e . --extra-index-url https://download.pytorch.org/whl/cu121

# Web UI, VLM 모델 구동용 추가 패키지 설치
RUN pip install --no-cache-dir gradio timm torchvision qwen-vl-utils decord av

# 프로젝트의 모든 나머지 코드 복사
COPY . .

# Gradio Web UI 포트 노출
EXPOSE 7860

# Web UI 서버 실행
CMD ["python", "app.py"]
